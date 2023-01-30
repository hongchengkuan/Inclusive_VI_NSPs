from sampler import PriorSampler, PosteriorSampler
import torch
import utils
import copy
import preprocess_dataset
import time

PAD = preprocess_dataset.PAD

F = torch.nn.functional


class EM:
    def __init__(
        self,
        PPs,
        events_embedding,
        initialize_batch_size,
        resample_only,
        record_virtual_samples,
        check_real_ll,
        device,
        predict=False,
    ) -> None:
        self.PPs = PPs
        self.events_embedding = PPs.events_embedding
        self.posterior_sampler = PosteriorSampler(
            PPs,
            self.events_embedding,
            resample_only,
            record_virtual_samples,
            check_real_ll,
            device,
            predict=predict,
        )
        self.device = device
        self.posterior_sampler.initialize_ll(initialize_batch_size, plot=False, device=device if predict else None)
        self.PPs.to(device)

    def get_parents_samples(
        self, thisPP, samples_events_times_dict, samples_events_mask_dict
    ):
        # get the events from the parents
        parents_ids_dict = thisPP.parents_ids_dict
        parents_real_events_times = torch.cat(
            [samples_events_times_dict[p_id] for p_id in parents_ids_dict],
            dim=-1,
        )
        parents_real_events_ids = torch.cat(
            [
                torch.tensor([p_id], device=self.device).expand_as(samples_events_times_dict[p_id])
                for p_id in parents_ids_dict
            ],
            dim=-1,
        )
        parents_real_events_mask = torch.cat(
            [samples_events_mask_dict[p_id] for p_id in parents_ids_dict],
            dim=-1,
        )
         
        parents_real_events_times[~parents_real_events_mask] = 1e20

        parents_real_events_times, indices = torch.sort(
            parents_real_events_times, dim=-1
        )
        parents_real_events_mask = torch.gather(
            parents_real_events_mask, dim=-1, index=indices
        )
        parents_real_events_ids = torch.gather(
            parents_real_events_ids, dim=-1, index=indices
        )
        return (
            parents_real_events_times,
            parents_real_events_ids,
            parents_real_events_mask,
        )

    def get_children_samples(
        self, thisPP, samples_events_times_dict, samples_events_mask_dict
    ):
        # get the events from the parents
        children_ids_dict = thisPP.children_ids_dict
        children_real_events_times = torch.cat(
            [samples_events_times_dict[c_id] for c_id in children_ids_dict],
            dim=-1,
        )
        children_real_events_ids = torch.cat(
            [
                torch.tensor([c_id], device=self.device).expand_as(samples_events_times_dict[c_id])
                for c_id in children_ids_dict
            ],
            dim=-1,
        )
        children_real_events_mask = torch.cat(
            [samples_events_mask_dict[c_id] for c_id in children_ids_dict],
            dim=-1,
        )

        children_real_events_times[~children_real_events_mask] = 1e20

        children_real_events_times, indices = torch.sort(
            children_real_events_times
        )
        children_real_events_mask = torch.gather(
            children_real_events_mask, dim=-1, index=indices
        )
        children_real_events_ids = torch.gather(
            children_real_events_ids, dim=-1, index=indices
        )
        return (
            children_real_events_times,
            children_real_events_ids,
            children_real_events_mask,
        )

    def em_step(
        self,
        batch_ids,  # observation ids for a batch
        burn_in_steps,  # burn in steps, the number of scans before selecting samples
        sample_size,  # the number of samples for the calculation of the statistics(e.g., gradient)
        sample_intervals,  # the number of samples to discard before choosing a sample
        log_folder,  # the path to save some results
        fig,  # the fig object to plot
        axs,  # the axes object to plot
        save_mcmc,  # whether to save samples sampled from MCMC
        mcmc_load_path=None, # the path to load saved MCMC samples
        calc_elbo=False,
        data_type="train",
        first_after_opt=False,
    ):
        # expectation
        self.PPs.eval()
        if data_type == "train":
            batch_end_time = self.PPs.end_time[:, batch_ids, :]
        elif data_type == "dev":
            batch_end_time = self.PPs.dev_end_time[:, batch_ids, :]
        elif data_type == "test":
            batch_end_time = self.PPs.test_end_time[:, batch_ids, :]

        record_virtual_samples = self.posterior_sampler.record_virtual_samples
        if record_virtual_samples:
            (
                samples_events_times_dict,
                samples_events_mask_dict,
                virtual_samples_events_times_dict,
                virtual_samples_events_mask_dict,
            ) = self.posterior_sampler.sample(
                batch_ids=batch_ids,
                burn_in_steps=burn_in_steps,
                sample_size=sample_size,
                sample_intervals=sample_intervals,
                batch_end_time=batch_end_time,
                first_after_opt=first_after_opt,
            )
        else:
            if mcmc_load_path is not None:
                (
                    samples_events_times_dict,
                    samples_events_mask_dict,
                ) = torch.load(mcmc_load_path)
            else:
                time1 = time.time()
                (
                    samples_events_times_dict,
                    samples_events_mask_dict,
                ) = self.posterior_sampler.sample(
                    batch_ids=batch_ids,
                    burn_in_steps=burn_in_steps,
                    sample_size=sample_size,
                    sample_intervals=sample_intervals,
                    data_type=data_type,
                    batch_end_time=batch_end_time,
                    first_after_opt=first_after_opt,
                )
                # print('sample function takes', time.time()-time1)
            if burn_in_steps > 0:
                return

            if calc_elbo:
                with torch.random.fork_rng(enabled=True):
                    with torch.no_grad():
                        elbo_sampler = PriorSampler(
                            copy.deepcopy(self.PPs), self.events_embedding,
                        )
                        elbo_batch_size = len(batch_ids)
                        elbo_sample_size = 160
                        (
                            pure_virtual_samples_events_times_dict,
                            pure_virtual_samples_events_mask_dict,
                        ) = elbo_sampler.sample(
                            batch_size=elbo_batch_size,
                            return_samples=True,
                            n_samples=elbo_sample_size,
                        )
                        for pp in self.PPs.PPs_lst:
                            if not pp.bottom:
                                virtual_max_len = max(
                                    virtual_sample.shape[-1]
                                    for virtual_sample in pure_virtual_samples_events_times_dict[
                                        pp.id
                                    ]
                                )
                                pure_virtual_samples_events_times_dict[
                                    pp.id
                                ] = torch.cat(
                                    [
                                        torch.cat(
                                            (
                                                virtual_sample,
                                                torch.ones(
                                                    elbo_batch_size,
                                                    virtual_max_len
                                                    - virtual_sample.shape[-1],
                                                )
                                                * PAD,
                                            ),
                                            dim=1,
                                        )
                                        for virtual_sample in pure_virtual_samples_events_times_dict[
                                            pp.id
                                        ]
                                    ], dim=0
                                )
                                pure_virtual_samples_events_mask_dict[
                                    pp.id
                                ] = torch.cat(
                                    [
                                        torch.cat(
                                            (
                                                virtual_sample,
                                                torch.zeros(
                                                    elbo_batch_size,
                                                    virtual_max_len
                                                    - virtual_sample.shape[-1],
                                                ).bool(),
                                            ),
                                            dim=1,
                                        )
                                        for virtual_sample in pure_virtual_samples_events_mask_dict[
                                            pp.id
                                        ]
                                    ], dim=0
                                )
                            else:
                                pure_virtual_samples_events_times_dict[
                                    pp.id
                                ] = pp.real_events_times.expand(
                                    elbo_sample_size, -1, -1
                                )
                                pure_virtual_samples_events_mask_dict[
                                    pp.id
                                ] = pp.real_events_mask.expand(
                                    elbo_sample_size, -1, -1
                                )
                        elbo = self.calc_elbo(
                            pure_virtual_samples_events_times_dict,
                            pure_virtual_samples_events_mask_dict,
                        )

            if fig is not None:
                if save_mcmc:
                    torch.save(
                        [samples_events_times_dict, samples_events_mask_dict],
                        log_folder + "mcmc_samples.pt",
                    )
                utils.plot_mcmc(
                    samples_events_times_dict,
                    samples_events_mask_dict,
                    self.PPs.PPs_lst,
                    int(1e4),
                    log_folder,
                    fig,
                    axs,
                )
                exit(0)

        # maximization
        if data_type == "train":
            self.PPs.train()
        else:
            self.PPs.eval()
        virtual_loglikelihood_lst = []
        real_loglikelihood_lst = []
        if record_virtual_samples:
            virtual_loglikelihood_wrt_virtual_lst = []
        PPs_lst = self.PPs.PPs_lst
        real_events_in_groups = {}
        real_ids_in_groups = {}
        real_mask_in_groups = {}
        real_embeddings_in_groups = {}
        bottom_real_loglikelihood = 0
        base_rate_dict = {}
        for thisPP in PPs_lst:
            if thisPP.top:
                thisPP_events_times_sample = samples_events_times_dict[
                    thisPP.id
                ]
                this_real_loglikelihood = thisPP.real_ll(
                    thisPP_events_times_sample,
                    samples_events_mask_dict[thisPP.id],
                    None,
                    None,
                    None,
                    batch_end_time=batch_end_time,
                    data_type=data_type,
                    batch_ids=batch_ids,
                )
                this_real_loglikelihood = torch.mean(
                    this_real_loglikelihood, dim=(0, 1), keepdim=True
                ).squeeze(0)
                real_loglikelihood_lst.append(this_real_loglikelihood)

                if thisPP.base_rate_opt:
                    base_rate_dict[thisPP.id] = (
                        torch.mean(
                            torch.sum(
                                samples_events_mask_dict[thisPP.id],
                                dim=-1, keepdim=True
                            ).float(), dim=0, keepdim=True
                        )
                        / batch_end_time
                    )

            else:
                parents_ids_string = thisPP.parents_ids_string
                if not parents_ids_string in real_events_in_groups:
                    (
                        parents_real_events_times,
                        parents_real_events_ids,
                        parents_real_events_mask,
                    ) = self.get_parents_samples(
                        thisPP,
                        samples_events_times_dict,
                        samples_events_mask_dict,
                    )
                    parents_real_events_embeddings = self.PPs.events_embedding(
                        parents_real_events_times, parents_real_events_ids
                    )
                    real_events_in_groups[
                        parents_ids_string
                    ] = parents_real_events_times
                    real_ids_in_groups[
                        parents_ids_string
                    ] = parents_real_events_ids
                    real_mask_in_groups[
                        parents_ids_string
                    ] = parents_real_events_mask
                    real_embeddings_in_groups[
                        parents_ids_string
                    ] = parents_real_events_embeddings
                else:
                    parents_real_events_times = real_events_in_groups[
                        parents_ids_string
                    ]
                    parents_real_events_ids = real_ids_in_groups[
                        parents_ids_string
                    ]
                    parents_real_events_mask = real_mask_in_groups[
                        parents_ids_string
                    ]
                    parents_real_events_embeddings = real_embeddings_in_groups[
                        parents_ids_string
                    ]

                thisPP_events_times_sample = samples_events_times_dict[
                    thisPP.id
                ]
                if thisPP.processes_type == "general":
                    parents_real_events_params = thisPP.calc_params(
                        neighbor_real_events_embeddings=parents_real_events_embeddings,
                        neighbor_mask=parents_real_events_mask.unsqueeze(-1),
                        virtual=False,
                    )
                    parents_real_events_stats_ids = None
                else:
                    parents_shape = parents_real_events_times.shape
                    parents_real_events_stats_ids = parents_real_events_ids.clone()
                    for key, val in thisPP.parents_ids_dict.items():
                        parents_real_events_stats_ids = parents_real_events_stats_ids.masked_fill(parents_real_events_stats_ids == key, val)
                    parents_real_events_params = thisPP.kernel_params.expand(parents_shape[0], parents_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
                    if parents_real_events_params.shape[-2] > 1:
                        parents_real_events_params = parents_real_events_params.gather(dim=-2, index=parents_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, parents_real_events_params.shape[-1]))
                this_real_loglikelihood = thisPP.real_ll(
                    thisPP_events_times_sample,
                    samples_events_mask_dict[thisPP.id],
                    parents_real_events_times,
                    parents_real_events_params,
                    parents_real_events_mask,
                    batch_end_time=batch_end_time,
                    data_type=data_type,
                    parents_stats_ids=parents_real_events_stats_ids,
                )
                this_real_loglikelihood = torch.mean(
                    this_real_loglikelihood, dim=(0, 1), keepdim=True
                ).squeeze(0)
                real_loglikelihood_lst.append(this_real_loglikelihood)
                if thisPP.bottom:
                    bottom_real_loglikelihood = bottom_real_loglikelihood + this_real_loglikelihood.item()

            if not thisPP.bottom:
                children_ids_string = thisPP.children_ids_string
                if not children_ids_string in real_events_in_groups:
                    (
                        children_real_events_times,
                        children_real_events_ids,
                        children_real_events_mask,
                    ) = self.get_children_samples(
                        thisPP,
                        samples_events_times_dict,
                        samples_events_mask_dict,
                    )
                    children_real_events_embeddings = self.PPs.events_embedding(
                        children_real_events_times, children_real_events_ids
                    )
                    real_events_in_groups[
                        children_ids_string
                    ] = children_real_events_times
                    real_ids_in_groups[
                        children_ids_string
                    ] = children_real_events_ids
                    real_mask_in_groups[
                        children_ids_string
                    ] = children_real_events_mask
                    real_embeddings_in_groups[
                        children_ids_string
                    ] = children_real_events_embeddings
                else:
                    children_real_events_times = real_events_in_groups[
                        children_ids_string
                    ]
                    children_real_events_ids = real_ids_in_groups[
                        children_ids_string
                    ]
                    children_real_events_mask = real_mask_in_groups[
                        children_ids_string
                    ]
                    children_real_events_embeddings = real_embeddings_in_groups[
                        children_ids_string
                    ]

                thisPP_virtual_events_times_sample = samples_events_times_dict[
                    thisPP.id
                ]
                if thisPP.virtual_processes_type == "general":
                    children_real_events_params = thisPP.calc_params(
                        neighbor_real_events_embeddings=children_real_events_embeddings,
                        neighbor_mask=children_real_events_mask.unsqueeze(-1),
                        virtual=True,
                    )
                    children_real_events_stats_ids = None
                else:
                    children_shape = children_real_events_times.shape
                    children_real_events_stats_ids = children_real_events_ids.clone()
                    for key, val in thisPP.children_ids_dict.items():
                        children_real_events_stats_ids = children_real_events_stats_ids.masked_fill(children_real_events_stats_ids == key, val)
                    children_real_events_params = thisPP.virtual_kernel_params.expand(children_shape[0], children_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
                    if children_real_events_params.shape[-2] > 1:
                        children_real_events_params = children_real_events_params.gather(dim=-2, index=children_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, children_real_events_params.shape[-1]))
                    
                this_virtual_loglikelihood = thisPP.virtual_ll(
                    thisPP_virtual_events_times_sample,
                    samples_events_mask_dict[thisPP.id],
                    children_real_events_times,
                    children_real_events_params,
                    children_real_events_mask,
                    batch_end_time=batch_end_time,
                    children_stats_ids=children_real_events_stats_ids,
                )
                this_virtual_loglikelihood = torch.mean(
                    this_virtual_loglikelihood, dim=(0, 1), keepdim=True
                ).squeeze(0)
                virtual_loglikelihood_lst.append(this_virtual_loglikelihood)

                if record_virtual_samples:
                    this_virtual_loglikelihood_wrt_virtual = thisPP.virtual_ll(
                        virtual_samples_events_times_dict[thisPP.id],
                        virtual_samples_events_mask_dict[thisPP.id],
                        children_real_events_times,
                        children_real_events_params,
                        children_real_events_mask,
                        batch_end_time=batch_end_time,
                        prop_virtual_background=True,
                        children_stats_ids=children_real_events_stats_ids,
                    )
                    this_virtual_loglikelihood_wrt_virtual = torch.mean(
                        this_virtual_loglikelihood_wrt_virtual,
                        dim=(0, 1),
                        keepdim=True,
                    ).squeeze(0)
                    virtual_loglikelihood_wrt_virtual_lst.append(
                        this_virtual_loglikelihood_wrt_virtual
                    )

        real_loglikelihood = torch.cat(real_loglikelihood_lst).sum()
        virtual_loglikelihood = torch.cat(virtual_loglikelihood_lst).sum()
        if record_virtual_samples:
            virtual_loglikelihood_wrt_virtual = torch.cat(
                virtual_loglikelihood_wrt_virtual_lst
            ).sum()
            return (
                real_loglikelihood,
                virtual_loglikelihood,
                virtual_loglikelihood_wrt_virtual,
            )
        if calc_elbo:
            return real_loglikelihood, virtual_loglikelihood, elbo
        return real_loglikelihood, virtual_loglikelihood, bottom_real_loglikelihood, base_rate_dict

    def calc_elbo(
        self,
        pure_virtual_samples_events_times_dict,
        pure_virtual_samples_events_mask_dict,
        batch_end_time,
    ):
        real_loglikelihood_lst = []
        entropy_lst = []
        PPs_lst = self.PPs.PPs_lst
        real_events_in_groups = {}
        real_ids_in_groups = {}
        real_mask_in_groups = {}
        for thisPP in PPs_lst:
            if thisPP.top:
                this_real_loglikelihood = thisPP.real_ll(
                    pure_virtual_samples_events_times_dict[thisPP.id],
                    pure_virtual_samples_events_mask_dict[thisPP.id],
                    None,
                    None,
                    None,
                    batch_end_time=batch_end_time,
                )
                this_real_loglikelihood = torch.mean(
                    this_real_loglikelihood, dim=0
                )
                real_loglikelihood_lst.append(this_real_loglikelihood)
            else:
                parents_ids_string = thisPP.parents_ids_string
                if not parents_ids_string in real_events_in_groups:
                    (
                        parents_real_events_times,
                        parents_real_events_ids,
                        parents_real_events_mask,
                    ) = self.get_parents_samples(
                        thisPP,
                        pure_virtual_samples_events_times_dict,
                        pure_virtual_samples_events_mask_dict,
                    )
                    real_events_in_groups[
                        parents_ids_string
                    ] = parents_real_events_times
                    real_ids_in_groups[
                        parents_ids_string
                    ] = parents_real_events_ids
                    real_mask_in_groups[
                        parents_ids_string
                    ] = parents_real_events_mask
                else:
                    parents_real_events_times = real_events_in_groups[
                        parents_ids_string
                    ]
                    parents_real_events_ids = real_ids_in_groups[
                        parents_ids_string
                    ]
                    parents_real_events_mask = real_mask_in_groups[
                        parents_ids_string
                    ]

                parents_real_events_embeddings = self.PPs.events_embedding(
                    parents_real_events_times, parents_real_events_ids
                )
                this_real_loglikelihood = thisPP.real_ll(
                    pure_virtual_samples_events_times_dict[thisPP.id],
                    pure_virtual_samples_events_mask_dict[thisPP.id],
                    parents_real_events_times,
                    parents_real_events_embeddings,
                    parents_real_events_mask,
                    batch_end_time=batch_end_time,
                )  # [n_samples, batch_size]
                this_real_loglikelihood = torch.mean(
                    this_real_loglikelihood.sum(dim=-1), dim=0, keepdim=True,
                )
                real_loglikelihood_lst.append(this_real_loglikelihood)

            if not thisPP.bottom:
                children_ids_string = thisPP.children_ids_string
                virtual_processes_type = thisPP.virtual_processes_type
                if not children_ids_string in real_events_in_groups:
                    (
                        children_real_events_times,
                        children_real_events_ids,
                        children_real_events_mask,
                    ) = self.get_children_samples(
                        thisPP,
                        pure_virtual_samples_events_times_dict,
                        pure_virtual_samples_events_mask_dict,
                    )
                    real_events_in_groups[
                        children_ids_string
                    ] = children_real_events_times
                    real_ids_in_groups[
                        children_ids_string
                    ] = children_real_events_ids
                    real_mask_in_groups[
                        children_ids_string
                    ] = children_real_events_mask
                else:
                    children_real_events_times = real_events_in_groups[
                        children_ids_string
                    ]
                    children_real_events_ids = real_ids_in_groups[
                        children_ids_string
                    ]
                    children_real_events_mask = real_mask_in_groups[
                        children_ids_string
                    ]

                children_real_events_embeddings = (
                    self.PPs.events_embedding(
                        children_real_events_times, children_real_events_ids,
                    )
                    if virtual_processes_type == "general"
                    else None
                )

                # virtual params
                (
                    children_diff_time_ll_events,
                    params,
                    virtual_kernel_params_ll_events,
                ) = thisPP.prepare_params_and_diff_time(
                    real_virtual_events_times=pure_virtual_samples_events_times_dict[
                        thisPP.id
                    ],
                    neighbor_events_times=children_real_events_times,
                    neighbor_events_embeddings=children_real_events_embeddings
                    if virtual_processes_type == "general"
                    else None,
                    neighbor_real_events_mask=children_real_events_mask,
                    neighbor="children",
                    processes_type=virtual_processes_type,
                )
                if thisPP.virtual_processes_type == "general":
                    virtual_ll_events = torch.log(
                        thisPP.virtual_kernel.forward(
                            children_diff_time_ll_events,
                            virtual_kernel_params_ll_events,
                        )
                        + F.softplus(thisPP.virtual_background_rate)
                    )
                else:
                    virtual_ll_events = torch.log(
                        torch.sum(
                            thisPP.virtual_kernel.forward(
                                children_diff_time_ll_events,
                                virtual_kernel_params_ll_events.unsqueeze(-2),
                            ),
                            dim=-1,
                        )
                        + F.softplus(thisPP.virtual_background_rate)
                    )
                this_entropy = torch.mean(
                    torch.sum(
                        (virtual_ll_events - 1)
                        * pure_virtual_samples_events_mask_dict[thisPP.id],
                        dim=(-1, -2),
                    ),
                    dim=0,
                    keepdim=True,
                )
                entropy_lst.append(this_entropy)

        real_loglikelihood = torch.cat(real_loglikelihood_lst).sum()
        n_entropy = torch.cat(entropy_lst).sum()
        return real_loglikelihood - n_entropy
