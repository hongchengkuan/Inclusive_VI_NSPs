import torch.nn as nn
import custom_attention
import kernel
import utils
import torch
import matplotlib.pyplot as plt
from utils import get_children_events, get_parents_params_mask
import math

F = nn.functional


class PPs(nn.Module):
    def __init__(
        self, train_obs_size, dev_obs_size, test_obs_size, processes_type, virtual_processes_type, end_time, dev_end_time, test_end_time,
    ) -> None:
        super().__init__()
        self.PPs_lst = []
        self.evidence_ids_set = frozenset()
        self.top_ids_set = frozenset()
        self.hidden_pps_ids_lst = []
        self.num_of_hidden_PPs = 0
        self.train_obs_size = train_obs_size
        self.dev_obs_size = dev_obs_size
        self.test_obs_size = test_obs_size
        self.processes_type = processes_type
        self.virtual_processes_type = virtual_processes_type
        self.dev = dev_obs_size != 0
        self.test = test_obs_size != 0
        self.register_buffer("end_time", end_time)
        self.register_buffer("dev_end_time", dev_end_time)
        self.register_buffer("test_end_time", test_end_time)

    def add_PP(
        self,
        id,
        end_time,
        top_base_rate,
        kernel_type,
        parents_ids_dict,
        parents_ids_string,
        children_ids_dict,
        children_ids_string,
        attn_n_layers,
        d_model,
        d_inner,
        n_head,
        top,
        bottom,
        kernel_params=None,
        background_rate=None,
        virtual_kernel_params=None,
        virtual_background_rate=None,
        virtual_prop_background_rate=None,
        evidence_events_times=None,
        evidence_events_mask=None,
        dev_evidence_events_times=None,
        dev_evidence_events_mask=None,
        test_evidence_events_times=None,
        test_evidence_events_mask=None,
        normalize_before=True,
        dropout=0,
        kernel_params_opt=False,
        base_rate_opt=False,
        background_rate_opt=False,
        predict=False,
    ):
        self.PPs_lst.append(
            PP(
                id,
                end_time,
                top_base_rate,
                self.train_obs_size,
                self.dev_obs_size,
                self.test_obs_size,
                kernel_type,
                parents_ids_dict,
                parents_ids_string,
                children_ids_dict,
                children_ids_string,
                attn_n_layers,
                d_model,
                d_inner,
                n_head,
                top,
                bottom,
                self.processes_type,
                self.virtual_processes_type,
                kernel_params,
                background_rate,
                virtual_kernel_params,
                virtual_background_rate,
                virtual_prop_background_rate,
                evidence_events_times=evidence_events_times,
                evidence_events_mask=evidence_events_mask,
                dev_evidence_events_times=dev_evidence_events_times,
                dev_evidence_events_mask=dev_evidence_events_mask,
                test_evidence_events_times=test_evidence_events_times,
                test_evidence_events_mask=test_evidence_events_mask,
                normalize_before=normalize_before,
                dropout=dropout,
                kernel_params_opt=kernel_params_opt,
                base_rate_opt=base_rate_opt,
                background_rate_opt=background_rate_opt,
                predict=predict,
            )
        )
        if bottom:
            self.evidence_ids_set = self.evidence_ids_set.union([id])
        else:
            self.num_of_hidden_PPs += 1
            self.hidden_pps_ids_lst.append(id)

        if top:
            self.top_ids_set = self.top_ids_set.union([id])

    def register_all_PPs(self, d_model):
        self.PPs_lst = nn.ModuleList(self.PPs_lst)
        self.events_embedding = utils.EventsEmbedding(
            d_model=d_model, numPPs=len(self.PPs_lst),
        )

    def plot_virtual_intensity(
        self,
        plot_num_points,
        posterior_sampler,
        batch_ids=[0],
        log_folder=None,
    ):
        PPs_lst = self.PPs_lst
        n_hidden = self.num_of_hidden_PPs
        fig, axs = plt.subplots(n_hidden)
        time_mesh = torch.linspace(0, 1, plot_num_points) * self.end_time[:, batch_ids, :]
        for thisPP in PPs_lst:
            if thisPP.id not in self.evidence_ids_set:
                virtual_processes_type = thisPP.virtual_processes_type

                children_ids_dict = thisPP.children_ids_dict
                multiple_sample = True
                for c_id in children_ids_dict:
                    if c_id in self.evidence_ids_set:
                        multiple_sample = False
                        break
                num_of_multiple_samples = 1000 if multiple_sample else 1
                virtual_intensity_lst = []
                for n in range(num_of_multiple_samples):
                    for pp in PPs_lst:
                        if pp.id not in self.evidence_ids_set:
                            pp.real_events_times = None
                            pp.real_events_mask = None
                            pp.virtual_events_times = None
                            pp.virtual_events_mask = None
                    posterior_sampler.initialize_ll(1, plot=True)

                    batch_real_events_times = {}
                    batch_real_events_mask = {}
                    batch_virtual_events_times = {}
                    batch_virtual_events_mask = {}
                    batch_real_loglikelihood = {}
                    batch_virtual_loglikelihood = {}
                    batch_base_rate = {}
                    for pp in PPs_lst:
                        batch_real_events_times[pp.id] = pp.real_events_times[:, batch_ids, ...]
                        batch_real_events_mask[pp.id] = pp.real_events_mask[:, batch_ids, ...]
                        if pp.virtual_events_times is not None:
                            batch_virtual_events_times[pp.id] = pp.virtual_events_times[:, batch_ids, ...]
                            batch_virtual_events_mask[pp.id] = pp.virtual_events_mask[:, batch_ids, ...]
                            batch_virtual_loglikelihood[pp.id] = pp.virtual_loglikelihood[:, batch_ids, ...]
                        batch_real_loglikelihood[pp.id] = pp.real_loglikelihood[:, batch_ids, ...]
                        if pp.top:
                            batch_base_rate[pp.id] = pp.base_rate[:, batch_ids, :]
                    (
                        children_real_events_times,  # [batch_size, seq_len]
                        children_real_events_mask,
                        children_real_events_ids,
                        _,
                    ) = get_children_events(thisPP, batch_real_events_times, batch_real_events_mask)
                    numel_children = torch.numel(children_real_events_times)
                    if numel_children != 0:
                        if virtual_processes_type == "general":
                            children_real_events_embeddings = self.events_embedding(
                                children_real_events_times,
                                children_real_events_ids,
                            )
                            params = thisPP.calc_params(
                                neighbor_real_events_embeddings=children_real_events_embeddings,
                                neighbor_mask=children_real_events_mask.unsqueeze(-1),
                                virtual=True,
                            )
                            children_real_events_stats_ids = None
                        else:
                            children_real_events_times_shape = children_real_events_times.shape
                            children_real_events_stats_ids = children_real_events_ids.clone()
                            for key, val in thisPP.children_ids_dict.items():
                                children_real_events_stats_ids = children_real_events_stats_ids.masked_fill(children_real_events_stats_ids == key, val)
                            params = thisPP.virtual_kernel_params.expand(children_real_events_times_shape[0], children_real_events_times_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
                            if params.shape[-2] > 1:
                                params = params.gather(dim=-2, index=children_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, params.shape[-1]))
                        (
                            children_diff_time_ll_events,
                            params_ll_events,
                        ) = thisPP.prepare_params_and_diff_time(
                            real_virtual_events_times=time_mesh.unsqueeze(
                                0
                            ).unsqueeze(0),
                            neighbor_events_times=children_real_events_times,
                            params=params,
                            neighbor_real_events_mask=children_real_events_mask,
                            neighbor="children",
                            processes_type=virtual_processes_type,
                            neighbor_stats_ids=children_real_events_stats_ids,
                        )

                    if self.virtual_processes_type == "general":
                        if numel_children != 0:
                            virtual_intensity = thisPP.virtual_kernel.forward(
                                children_diff_time_ll_events, params_ll_events,
                            ) + F.softplus(thisPP.virtual_background_rate)
                        else:
                            virtual_intensity = F.softplus(
                                thisPP.virtual_background_rate
                            ).expand_as(time_mesh.unsqueeze(0).unsqueeze(0))
                    else:
                        if numel_children != 0:
                            virtual_intensity = torch.sum(
                                thisPP.virtual_kernel.forward(
                                    children_diff_time_ll_events,
                                    params_ll_events.unsqueeze(-2),
                                ),
                                dim=-1,
                            ) + F.softplus(thisPP.virtual_background_rate)
                        else:
                            virtual_intensity = F.softplus(
                                thisPP.virtual_background_rate
                            ).expand_as(time_mesh.unsqueeze(0).unsqueeze(0))
                    virtual_intensity_lst.append(virtual_intensity)
                virtual_intensity = torch.mean(
                    torch.stack(virtual_intensity_lst, dim=0), dim=0
                )

                if n_hidden == 1:
                    axs.set_title(str(thisPP.id), fontsize=20)
                    axs.set_xlabel("time", fontsize=15)
                    axs.set_ylabel("intensity", fontsize=15)
                    axs.plot(time_mesh, virtual_intensity.squeeze().detach())
                else:
                    axs[n_hidden - thisPP.id].set_title(
                        str(thisPP.id), fontsize=20
                    )
                    axs[n_hidden - thisPP.id].set_xlabel("time", fontsize=15)
                    axs[n_hidden - thisPP.id].set_ylabel(
                        "intensity", fontsize=15
                    )
                    axs[n_hidden - thisPP.id].plot(
                        time_mesh, virtual_intensity.squeeze().detach()
                    )

        if log_folder is not None:
            fig.tight_layout()
            fig.savefig(log_folder + "hidden_test.pdf")
            return None, None
        else:
            return fig, axs

    def sample_first_event(self, batch_ids, device):
        first_event_dict = {"type":torch.tensor([-1], device=device).expand(len(batch_ids), -1).clone(), "time":torch.tensor([float('inf')], device=device).expand(len(batch_ids), -1).clone()}
        batch_size = len(batch_ids)
        forward_sampling_done_layers_set = set([])
        batch_real_events_times = {}
        batch_real_events_mask = {}
        for pp in self.PPs_lst:
            batch_real_events_times[pp.id] = pp.test_real_events_times[
                :, batch_ids, ...
            ]
            batch_real_events_mask[pp.id] = pp.test_real_events_mask[
                :, batch_ids, ...
            ]
        batch_end_time = self.test_end_time[:, batch_ids, :]
        for e_layer_id in self.evidence_ids_set:
            self.sample_parents_first_event(
                e_layer_id,
                first_event_dict,
                forward_sampling_done_layers_set,
                batch_ids,
                batch_real_events_times,
                batch_real_events_mask,
                batch_end_time,
                device,
            )

        top_next_sample_time = None
        while (first_event_dict["time"] >= 1e20).any():
            topPP_to_sample = self.PPs_lst[next(iter(self.top_ids_set))]
            if top_next_sample_time is None:
                top_next_sample_time = -torch.log1p(-torch.rand(1, batch_size, 1, device=device)) / topPP_to_sample.test_base_rate[:, batch_ids, :] + self.test_end_time[:, batch_ids, :]
            else:
                top_next_sample_time += -torch.log1p(-torch.rand(1, batch_size, 1, device=device)) / topPP_to_sample.test_base_rate[:, batch_ids, :]
            # assert (top_next_sample_time < 1e20).all()

            src_batch_real_events_times = {}
            src_batch_real_events_mask = {}
            for parentId in self.top_ids_set:
                if parentId == topPP_to_sample.id:
                    src_batch_real_events_times[parentId] = top_next_sample_time
                    src_batch_real_events_mask[parentId] = torch.ones_like(top_next_sample_time, device=device).bool()
                else:
                    src_batch_real_events_times[parentId] = torch.tensor([[[]]])
                    src_batch_real_events_mask[parentId] = torch.tensor([[[]]])
            self.forward_sampling_to_the_evidence(
                topPP_to_sample,
                first_event_dict,
                src_batch_real_events_times,
                src_batch_real_events_mask,
                batch_end_time,
                batch_ids,
                device,
            )
            if torch.max(first_event_dict["time"][(top_next_sample_time < 1e20).squeeze(0)]) < 1e20:
                break
        
        # avoid the case that top pp has many events -----------------------------------------------------------------------------
        if top_next_sample_time is not None:
            num_top_extend_to_the_end = torch.max((pp.test_base_rate[:, batch_ids, :] * (first_event_dict["time"] - top_next_sample_time)).masked_fill(first_event_dict["time"] <= top_next_sample_time, 0))
        else:
            num_top_extend_to_the_end = torch.max((pp.test_base_rate[:, batch_ids, :] * (first_event_dict["time"] - self.test_end_time[:, batch_ids, :])).masked_fill(first_event_dict["time"] <= self.test_end_time[:, batch_ids, :], 0))
        while num_top_extend_to_the_end > 1e3:
            topPP_to_sample = self.PPs_lst[next(iter(self.top_ids_set))]
            if top_next_sample_time is None:
                top_next_sample_time = -torch.log1p(-torch.rand(1, batch_size, 1, device=device)) / topPP_to_sample.test_base_rate[:, batch_ids, :] + self.test_end_time[:, batch_ids, :]
            else:
                top_next_sample_time += -torch.log1p(-torch.rand(1, batch_size, 1, device=device)) / topPP_to_sample.test_base_rate[:, batch_ids, :]
            # assert (top_next_sample_time < 1e20).all()

            src_batch_real_events_times = {}
            src_batch_real_events_mask = {}
            for parentId in self.top_ids_set:
                if parentId == topPP_to_sample.id:
                    src_batch_real_events_times[parentId] = top_next_sample_time
                    src_batch_real_events_mask[parentId] = torch.ones_like(top_next_sample_time, device=device).bool()
                else:
                    src_batch_real_events_times[parentId] = torch.tensor([[[]]])
                    src_batch_real_events_mask[parentId] = torch.tensor([[[]]])
            self.forward_sampling_to_the_evidence(
                topPP_to_sample,
                first_event_dict,
                src_batch_real_events_times,
                src_batch_real_events_mask,
                batch_end_time,
                batch_ids,
                device,
            )
            num_top_extend_to_the_end = torch.max((pp.test_base_rate[:, batch_ids, :] * (first_event_dict["time"] - top_next_sample_time)).masked_fill(first_event_dict["time"] <= top_next_sample_time, 0))
        # ---------------------------------------------------------------------------------------------------------------------------------

        for evidence_id in self.evidence_ids_set:
            evidence_PP = self.PPs_lst[evidence_id]
            evidence_next_time = -torch.log1p(-torch.rand(1, batch_size, 1, device=device)) / F.softplus(evidence_PP.background_rate) + batch_end_time
            potential_first_event_time = evidence_next_time.squeeze(0)
            new_min_time_mask = first_event_dict["time"][batch_ids, :] > potential_first_event_time
            first_event_dict["time"][batch_ids] = torch.where(
                new_min_time_mask,
                potential_first_event_time,
                first_event_dict["time"][batch_ids]
            )
            first_event_dict["type"][batch_ids] = torch.where(
                new_min_time_mask,
                evidence_id,
                first_event_dict["type"][batch_ids]
            )
                

        assert (first_event_dict["time"] < 1e20).all()
        for top_pp_id in self.top_ids_set:
            src_batch_real_events_times = {}
            src_batch_real_events_mask = {}
            if top_pp_id == next(iter(self.top_ids_set)) and top_next_sample_time is not None:
                pp = self.PPs_lst[top_pp_id]
                poisson_rate = (pp.test_base_rate[:, batch_ids, :] * (first_event_dict["time"] - top_next_sample_time)).masked_fill(first_event_dict["time"] <= top_next_sample_time, 0)
                poisson_num = torch.poisson(poisson_rate)
                max_poisson_num = torch.max(poisson_num).int()
                src_events_times = torch.rand(1, batch_size, max_poisson_num, device=device) * (first_event_dict["time"] - top_next_sample_time) + top_next_sample_time
                src_events_mask = torch.arange(max_poisson_num, device=device) < poisson_num
                src_events_times[~src_events_mask] = 1e20
            else:
                pp = self.PPs_lst[top_pp_id]
                poisson_rate = (pp.test_base_rate[:, batch_ids, :] * (first_event_dict["time"] - self.test_end_time[:, batch_ids, :])).masked_fill(first_event_dict["time"] <= self.test_end_time[:, batch_ids, :], 0)
                poisson_num = torch.poisson(poisson_rate)
                max_poisson_num = torch.max(poisson_num).int()
                src_events_times = torch.rand(1, batch_size, max_poisson_num, device=device) * (first_event_dict["time"] - self.test_end_time[:, batch_ids, :]) + self.test_end_time[:, batch_ids, :]
                src_events_mask = torch.arange(max_poisson_num, device=device) < poisson_num
                src_events_times[~src_events_mask] = 1e20
            for parentId in self.top_ids_set:
                if parentId == top_pp_id:
                    src_batch_real_events_times[parentId] = src_events_times
                    src_batch_real_events_mask[parentId] = src_events_mask
                else:
                    src_batch_real_events_times[parentId] = torch.tensor([[[]]], device=device)
                    src_batch_real_events_mask[parentId] = torch.tensor([[[]]], device=device)
            if torch.numel(src_events_times) != 0:
                self.forward_sampling_to_the_evidence(
                    pp,
                    first_event_dict,
                    src_batch_real_events_times,
                    src_batch_real_events_mask,
                    batch_end_time,
                    batch_ids,
                    device,
                )

        for hidden_pp_id in self.hidden_pps_ids_lst:
            if hidden_pp_id not in self.top_ids_set:
                hidden_pp = self.PPs_lst[hidden_pp_id]
                poisson_num = torch.poisson(F.softplus(hidden_pp.background_rate) * (first_event_dict["time"] - batch_end_time) * (first_event_dict["time"] > batch_end_time))
                max_poisson_num = torch.max(poisson_num).int()
                if max_poisson_num > 0:
                    src_events_times = torch.rand(1, batch_size, max_poisson_num, device=device) * (first_event_dict["time"] - batch_end_time) + batch_end_time
                    src_events_mask = torch.arange(max_poisson_num, device=device) < poisson_num
                    src_batch_real_events_times = {}
                    src_batch_real_events_mask = {}
                    for hidden_pp_id_inner in self.hidden_pps_ids_lst:
                        if hidden_pp_id_inner == hidden_pp_id:
                            src_batch_real_events_times[hidden_pp_id_inner] = src_events_times
                            src_batch_real_events_mask[hidden_pp_id_inner] = src_events_mask
                        else:
                            src_batch_real_events_times[hidden_pp_id_inner] = torch.tensor([[[]]], device=device)
                            src_batch_real_events_mask[hidden_pp_id_inner] = torch.tensor([[[]]], device=device)
                    self.forward_sampling_to_the_evidence(
                        hidden_pp,
                        first_event_dict,
                        src_batch_real_events_times,
                        src_batch_real_events_mask,
                        batch_end_time,
                        batch_ids,
                        device,
                    )

        return first_event_dict

    def sample_parents_first_event(
        self,
        c_layer_id,
        first_event_dict,
        forward_sampling_done_layers_set,
        batch_ids,
        batch_real_events_times,
        batch_real_events_mask,
        batch_end_time,
        device
    ):
        if self.PPs_lst[c_layer_id].parents_ids_dict:
            parents_ids_keys = self.PPs_lst[c_layer_id].parents_ids_dict.keys()
            for p_id in parents_ids_keys:
                if p_id not in forward_sampling_done_layers_set:
                    parentPP = self.PPs_lst[p_id]
                    forward_sampling_done_layers_set.update([p_id])
                    self.forward_sampling_to_the_evidence(
                        parentPP,
                        first_event_dict,
                        batch_real_events_times,
                        batch_real_events_mask,
                        batch_end_time,
                        batch_ids,
                        device,
                    )
                self.sample_parents_first_event(
                    p_id,
                    first_event_dict,
                    forward_sampling_done_layers_set,
                    batch_ids,
                    batch_real_events_times,
                    batch_real_events_mask,
                    batch_end_time,
                    device,
                )

    def forward_sampling_to_the_evidence(
        self,
        pp,
        first_event_dict,
        batch_real_events_times,
        batch_real_events_mask,
        batch_end_time,
        batch_ids,
        device,
    ):
        if pp.children_ids_dict:
            for c_id in pp.children_ids_dict:
                childPP = self.PPs_lst[c_id]
                (
                    src_params, # [1, batch_size, seq_len, n_params]
                    src_real_events_times, 
                    src_real_events_mask,
                ) = get_parents_params_mask(
                    self, 
                    childPP, 
                    batch_real_events_times, 
                    batch_real_events_mask, 
                    None, 
                    device
                )
                sampling_start_time = torch.maximum(
                    src_real_events_times,
                    batch_end_time.expand_as(src_real_events_times),
                )
                mu = torch.empty_like(sampling_start_time).exponential_()
                src_real_events_times_childPP = childPP.kernel.integral_inv(
                    mu + childPP.kernel.integral(
                        (sampling_start_time - src_real_events_times) * src_real_events_mask, src_params
                    ), src_params
                ) + src_real_events_times
                src_real_events_times_childPP[~src_real_events_mask] = 1e20
                if c_id in self.evidence_ids_set:
                    if torch.numel(src_real_events_times_childPP) == 0:
                        continue
                    potential_first_event_time = torch.min(src_real_events_times_childPP, dim=-1, keepdim=True).values.squeeze(0)
                    new_min_time_mask = first_event_dict["time"][batch_ids, :] > potential_first_event_time
                    first_event_dict["time"][batch_ids] = torch.where(
                        new_min_time_mask,
                        potential_first_event_time,
                        first_event_dict["time"][batch_ids]
                    )
                    first_event_dict["type"][batch_ids] = torch.where(
                        new_min_time_mask,
                        c_id,
                        first_event_dict["type"][batch_ids]
                    )
                    continue

                src_real_events_mask = torch.logical_and(
                    src_real_events_mask,
                    src_real_events_times_childPP < first_event_dict["time"][batch_ids, :]
                )
                src_batch_real_events_times = {}
                src_batch_real_events_mask = {}
                if src_real_events_mask.any():
                    for grand_child_Id in childPP.children_ids_dict:
                        for parentId in self.PPs_lst[grand_child_Id].parents_ids_dict:
                            if parentId == c_id:
                                src_batch_real_events_times[parentId] = src_real_events_times_childPP
                                src_batch_real_events_mask[parentId] = src_real_events_mask
                            else:
                                src_batch_real_events_times[parentId] = torch.tensor([[[]]])
                                src_batch_real_events_mask[parentId] = torch.tensor([[[]]])
                    self.forward_sampling_to_the_evidence(
                        childPP,
                        first_event_dict,
                        src_batch_real_events_times,
                        src_batch_real_events_mask,
                        batch_end_time,
                        batch_ids,
                        device,
                    )

class PP(nn.Module):
    def __init__(
        self,
        id,
        end_time,
        top_base_rate,
        train_obs_size,
        dev_obs_size,
        test_obs_size,
        kernel_type,
        parents_ids_dict,
        parents_ids_string,
        children_ids_dict,
        children_ids_string,
        attn_n_layers,
        d_model,
        d_inner,
        n_head,
        top,
        bottom,
        processes_type,
        virtual_processes_type,
        kernel_params=None,
        background_rate=None,
        virtual_kernel_params=None,
        virtual_background_rate=None,
        virtual_prop_background_rate=None,
        evidence_events_times=None,
        evidence_events_mask=None,
        dev_evidence_events_times=None,
        dev_evidence_events_mask=None,
        test_evidence_events_times=None,
        test_evidence_events_mask=None,
        normalize_before=True,
        dropout=0,
        synthetic_end=False,
        kernel_params_opt=False,
        base_rate_opt=False,
        background_rate_opt=False,
        predict=False,
    ) -> None:
        super().__init__()
        self.id = id
        self.end_time = end_time

        self.top = top
        self.bottom = bottom
        self.processes_type = processes_type
        self.virtual_processes_type = virtual_processes_type

        self.base_rate_opt = base_rate_opt

        self.synthetic_end = synthetic_end
        self.num_params_per_kernel = (
            3
            if kernel_type == "Weibull" or kernel_type == "Gompertz" or kernel_type == "Normal"
            else
            4
            if kernel_type == "WeibullShift" or kernel_type == "GompertzShift"
            else 2
        )

        if top:
            self.register_buffer('base_rate', torch.ones(1, train_obs_size, 1) * top_base_rate)
            self.register_buffer('dev_base_rate', torch.ones(1, dev_obs_size, 1) * top_base_rate)
            self.register_buffer('test_base_rate', torch.ones(1, test_obs_size, 1) * top_base_rate)

        if not bottom:
            if virtual_processes_type == "general":
                self.attn_encoder_virtualPP = custom_attention.Encoder(
                    n_layers=attn_n_layers,
                    d_model=d_model,
                    d_inner=d_inner,
                    n_head=n_head,
                    d_k=d_model // n_head,
                    d_v=d_model // n_head,
                    normalize_before=normalize_before,
                    dropout=dropout,
                )
                self.encoding_to_virtual_kernel_nn = nn.Linear(d_model, self.num_params_per_kernel)
            else:
                if virtual_kernel_params is None:
                    self.virtual_kernel_params = nn.Parameter(
                        torch.log(torch.expm1(torch.tensor([[[[1.0, 1.0]]]]))),
                        requires_grad=True,
                    )
                else:
                    virtual_kernel_params_inf_mask = virtual_kernel_params > 90
                    virtual_kernel_params_transform = virtual_kernel_params.masked_fill(virtual_kernel_params_inf_mask, 0)
                    virtual_kernel_params_transform = torch.log(torch.expm1(virtual_kernel_params_transform))
                    virtual_kernel_params_transform = torch.where(virtual_kernel_params_inf_mask, virtual_kernel_params, virtual_kernel_params_transform)
                    self.virtual_kernel_params = nn.Parameter(
                        virtual_kernel_params_transform,
                        requires_grad=True,
                    )
            self.virtual_kernel = (
                kernel.Weibull
                if kernel_type == "Weibull"
                else kernel.Gompertz
                if kernel_type == "Gompertz"
                else kernel.WeibullShift
                if kernel_type == "WeibullShift"
                else kernel.Normal
                if kernel_type == "Normal"
                else kernel.Exponential
            )
            self.virtual_loglikelihood = -torch.tensor(float("Inf"))
            if virtual_background_rate is None:
                self.virtual_background_rate = nn.Parameter(
                    torch.log(torch.expm1(torch.tensor(0.1))),
                    requires_grad=True,
                )
            else:
                self.virtual_background_rate = nn.Parameter(
                    torch.log(torch.expm1(virtual_background_rate)),
                    requires_grad=True,
                )
            self.virtual_prop_background_rate = virtual_prop_background_rate
        if not top:
            if processes_type == "general":
                self.attn_encoder_realPP = custom_attention.Encoder(
                    n_layers=attn_n_layers,
                    d_model=d_model,
                    d_inner=d_inner,
                    n_head=n_head,
                    d_k=d_model // n_head,
                    d_v=d_model // n_head,
                    normalize_before=normalize_before,
                    dropout=dropout,
                )
                self.encoding_to_kernel_nn = nn.Linear(d_model, self.num_params_per_kernel)
            else:
                if kernel_params is None:
                    self.kernel_params = nn.Parameter(
                        torch.log(torch.expm1(torch.tensor([[[[1.0, 1.0]]]]))),
                        requires_grad=kernel_params_opt,
                    )
                else:
                    kernel_params_inf_mask = kernel_params > 90
                    kernel_params_transform = kernel_params.masked_fill(kernel_params_inf_mask, 0)
                    kernel_params_transform = torch.log(torch.expm1(kernel_params_transform))
                    kernel_params_transform = torch.where(kernel_params_inf_mask, kernel_params, kernel_params_transform)
                    self.kernel_params = nn.Parameter(
                        kernel_params_transform,
                        requires_grad=kernel_params_opt,
                    )
            self.kernel = (
                kernel.Weibull
                if kernel_type == "Weibull"
                else kernel.Gompertz
                if kernel_type == "Gompertz"
                else kernel.WeibullShift
                if kernel_type == "WeibullShift"
                else kernel.Normal
                if kernel_type == "Normal"
                else kernel.Exponential
            )
            if background_rate is None:
                self.background_rate = nn.Parameter(
                    torch.log(torch.expm1(torch.tensor(1e-10))),
                    requires_grad=background_rate_opt,
                )
            else:
                self.background_rate = nn.Parameter(
                    torch.log(torch.expm1(background_rate)),
                    requires_grad=background_rate_opt,
                )
        self.real_loglikelihood = -torch.tensor(float("Inf"))

        self.children_ids_dict = children_ids_dict
        self.children_ids_string = children_ids_string
        self.parents_ids_dict = parents_ids_dict
        self.parents_ids_string = parents_ids_string

        if bottom:
            self.real_events_times = evidence_events_times.unsqueeze(0) if not predict else None
            self.real_events_mask = evidence_events_mask.unsqueeze(0) if not predict else None
        else:
            self.real_events_times = None
            self.real_events_mask = None
        self.dev_real_events_times = dev_evidence_events_times.unsqueeze(0) if dev_evidence_events_times is not None else None
        self.dev_real_events_mask = dev_evidence_events_mask.unsqueeze(0) if dev_evidence_events_mask is not None else None
        self.test_real_events_times = test_evidence_events_times.unsqueeze(0) if test_evidence_events_times is not None else None
        self.test_real_events_mask = test_evidence_events_mask.unsqueeze(0) if test_evidence_events_mask is not None else None

        self.virtual_events_times = None
        self.virtual_events_mask = None
        self.dev_virtual_events_times = None
        self.dev_virtual_events_mask = None
        self.test_virtual_events_times = None
        self.test_virtual_events_mask = None

    def set_events_mask_none(self, dev, test):
        self.real_events_times = None
        self.real_events_mask = None
        self.virtual_events_times = None
        self.virtual_events_mask = None
        self.virtual_loglikelihood = None
        self.real_loglikelihood = None
        if dev:
            self.dev_real_events_times = None
            self.dev_real_events_mask = None
            self.dev_virtual_events_times = None
            self.dev_virtual_events_mask = None
            self.dev_virtual_loglikelihood = None
            self.dev_real_loglikelihood = None
        if test:
            self.test_real_events_times = None
            self.test_real_events_mask = None
            self.test_virtual_events_times = None
            self.test_virtual_events_mask = None
            self.test_virtual_loglikelihood = None
            self.test_real_loglikelihood = None

    def calc_real_events_encoding(
        self,
        real_events_embeddings,
        non_pad_mask,
        slf_attn_mask,
        kernel_virtualness,
    ):
        attn_output = real_events_embeddings
        if kernel_virtualness:
            for (
                attn_layer
            ) in self.attn_encoder_virtualPP.attention_layers_stack:
                attn_output, _ = attn_layer(
                    q=attn_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                )
        else:
            for attn_layer in self.attn_encoder_realPP.attention_layers_stack:
                attn_output, _ = attn_layer(
                    q=attn_output,
                    non_pad_mask=non_pad_mask,
                    slf_attn_mask=slf_attn_mask,
                )

        return attn_output

    def calc_params(
        self, neighbor_real_events_embeddings, neighbor_mask, virtual,
    ):

        slf_attn_mask = neighbor_mask > 0

        real_events_encodings = self.calc_real_events_encoding(
            real_events_embeddings=neighbor_real_events_embeddings,
            slf_attn_mask=slf_attn_mask,
            non_pad_mask=slf_attn_mask,
            kernel_virtualness=virtual,
        )
        if not virtual:
            params = self.encoding_to_kernel_nn(real_events_encodings)
        else:
            params = self.encoding_to_virtual_kernel_nn(real_events_encodings)
        return params  # [batch_size, seq_len, kernel_params]

    def virtual_ll(
        self,
        virtual_events_times,
        virtual_events_mask,
        children_real_events_times,
        params,
        children_real_events_mask,
        batch_end_time,
        prop_virtual_background=False,
        children_stats_ids=None,
    ):
        if prop_virtual_background:
            virtual_background_rate = (
                F.softplus(self.virtual_background_rate)
                + self.virtual_prop_background_rate
            )
        else:
            virtual_background_rate = F.softplus(self.virtual_background_rate)
        numel_children = torch.numel(children_real_events_times)
        if numel_children != 0:
            (
                diff_time_ll_events,  # [sample_size, batch_size, virtual_seq_len, n_seq_len]
                params_ll_events,
            ) = self.prepare_params_and_diff_time(
                real_virtual_events_times=virtual_events_times,
                neighbor_events_times=children_real_events_times,
                params=params,
                neighbor_real_events_mask=children_real_events_mask,
                neighbor="children",
                processes_type=self.virtual_processes_type,
                neighbor_stats_ids=children_stats_ids,
            )
        if self.virtual_processes_type == "general":
            # diff_time_ll_events  # [sample_size, batch_size, virtual_seq_len]
            if numel_children != 0:
                ll_events = torch.log(
                    self.virtual_kernel.forward(
                        diff_time_ll_events, params_ll_events
                    )
                    + virtual_background_rate
                )
                times_for_int = children_real_events_times - F.pad(
                    children_real_events_times[..., :-1],
                    pad=(1, 0),
                    mode="constant",
                    value=0.0,
                )
                ll_time = self.virtual_kernel.integral(times_for_int * children_real_events_mask, params)
            else:
                ll_events = virtual_background_rate.expand_as(
                    virtual_events_times
                )
                ll_time = 0
        else:
            if numel_children != 0:
                ll_events = torch.log(
                    torch.sum(
                        self.virtual_kernel.forward(
                            diff_time_ll_events, params_ll_events
                        ),
                        dim=-1,
                    )
                    + virtual_background_rate
                )
                ll_time = self.virtual_kernel.integral(
                    children_real_events_times * children_real_events_mask, params
                )
            else:
                ll_events = virtual_background_rate.expand_as(
                    virtual_events_times
                )
                ll_time = torch.tensor([0.])
        return (
            (ll_events * virtual_events_mask).sum(dim=-1)
            - (ll_time).sum(dim=-1)
            - batch_end_time.squeeze(-1) * virtual_background_rate
        )

    def sample_virtual_events(
        self,
        params,
        children_real_events_times,  # [1, batch_size, seq_len]
        children_mask,
        batch_end_time,
        prop_virtual_background=False,
        device=None,
    ):
        if torch.numel(children_real_events_times) != 0:
            slf_attn_mask = children_mask
            if self.virtual_processes_type == "general":
                times_for_int = children_real_events_times - F.pad(
                    children_real_events_times[..., :-1],
                    pad=(1, 0),
                    mode="constant",
                    value=0.0,
                )
            else:
                times_for_int = children_real_events_times
            integrals = self.virtual_kernel.integral(times_for_int, params)
            poisson_random_num = torch.poisson(integrals)
            if poisson_random_num.shape[-1] == 0:
                max_poisson_random_num = 0
            else:
                max_poisson_random_num = torch.max(poisson_random_num).int()

            batch_size = children_real_events_times.shape[-2]
            seq_len = children_real_events_times.shape[-1]
            uniform_samples = (
                torch.rand(1, batch_size, seq_len, max_poisson_random_num, device=device)
                * integrals[..., None]
            )
            events_times_samples = children_real_events_times[
                ..., None
            ] - self.virtual_kernel.integral_inv(
                uniform_samples, params[:, :, :, None, :]
            )
            non_pad_mask = torch.arange(max_poisson_random_num, device=device)[
                None, None, None, :
            ].expand(1, batch_size, seq_len, max_poisson_random_num)
            non_pad_mask = non_pad_mask < poisson_random_num[..., None]
            non_pad_mask = torch.logical_and(
                children_mask.unsqueeze(-1), non_pad_mask
            )
            events_times_samples = torch.reshape(
                events_times_samples, (1, batch_size, -1)
            )
            non_pad_mask = torch.reshape(non_pad_mask, (1, batch_size, -1))
        else:
            batch_size = children_real_events_times.shape[-2]
            events_times_samples = torch.tensor([[[]]], device=device).expand(1, batch_size, -1)
            non_pad_mask = torch.tensor([[[]]], device=device).expand(1, batch_size, -1).bool()

        # background_rate samples
        if prop_virtual_background:
            virtual_background_rate = (
                F.softplus(self.virtual_background_rate)
                + self.virtual_prop_background_rate
            )
        else:
            virtual_background_rate = F.softplus(self.virtual_background_rate)
        b_integral = virtual_background_rate * batch_end_time
        b_poisson_random_num = torch.poisson(b_integral)
        b_max_poisson_random_num = torch.max(b_poisson_random_num).int()
        if b_max_poisson_random_num > 0:
            b_events_times_samples = (
                torch.rand(1, batch_size, b_max_poisson_random_num, device=device)
                * batch_end_time
            )
            b_non_pad_mask = torch.arange(b_max_poisson_random_num, device=device)[
                None, None, :
            ].expand(1, batch_size, -1)
            b_non_pad_mask = b_non_pad_mask < b_poisson_random_num
            events_times_samples = torch.cat(
                (events_times_samples, b_events_times_samples), dim=-1
            )
            non_pad_mask = torch.cat((non_pad_mask, b_non_pad_mask), dim=-1)

        non_pad_mask, indices = torch.sort(non_pad_mask.int(), dim=-1, descending=True)
        non_pad_mask = non_pad_mask.bool()
        events_times_samples = torch.gather(events_times_samples, dim=-1, index=indices)
        min_zero_num = torch.min(torch.sum(~non_pad_mask, dim=-1))
        if min_zero_num > 0:
            events_times_samples = events_times_samples[..., :-min_zero_num]
            non_pad_mask = non_pad_mask[..., :-min_zero_num]
        events_times_samples = torch.clamp(events_times_samples, min=0)
        return events_times_samples, non_pad_mask

    def sample_real_events(
        self,
        params,
        parents_real_events_times,
        parents_mask,
        batch_end_time,
    ):
        slf_attn_mask = parents_mask
        if self.processes_type == "general":
            times_for_int = (
                F.pad(
                    parents_real_events_times[..., 1:],
                    pad=(0, 1),
                    mode="constant",
                    value=batch_end_time,
                )
                - parents_real_events_times
            )
        else:
            times_for_int = batch_end_time - parents_real_events_times
        integrals = self.kernel.integral(times_for_int * parents_mask, params)
        poisson_random_num = torch.poisson(integrals)
        if poisson_random_num.shape[-1] == 0:
            max_poisson_random_num = 0
        else:
            max_poisson_random_num = torch.max(poisson_random_num).int()

        batch_size = parents_real_events_times.shape[-2]
        seq_len = parents_real_events_times.shape[-1]
        uniform_samples = (
            torch.rand(1, batch_size, seq_len, max_poisson_random_num)
            * integrals[..., None]
        )
        events_times_samples = parents_real_events_times[
            ..., None
        ] + self.kernel.integral_inv(uniform_samples, params[:, :, :, None, :])
        non_pad_mask = torch.arange(max_poisson_random_num)[
            None, None, None, :
        ].expand(1, batch_size, seq_len, max_poisson_random_num)
        non_pad_mask = non_pad_mask < poisson_random_num[..., None]
        non_pad_mask = torch.logical_and(
            parents_mask.unsqueeze(-1), non_pad_mask
        )
        events_times_samples = torch.reshape(
            events_times_samples, (1, batch_size, -1)
        )
        non_pad_mask = torch.reshape(non_pad_mask, (1, batch_size, -1))

        b_integral = F.softplus(self.background_rate) * batch_end_time
        b_poisson_random_num = torch.poisson(b_integral)
        b_max_poisson_random_num = torch.max(b_poisson_random_num).int()
        if b_max_poisson_random_num > 0:
            b_events_times_samples = (
                torch.rand(1, batch_size, b_max_poisson_random_num)
                * batch_end_time
            )
            b_non_pad_mask = torch.arange(b_max_poisson_random_num)[
                None, :
            ].expand(1, batch_size, -1)
            b_non_pad_mask = b_non_pad_mask < b_poisson_random_num
            events_times_samples = torch.cat(
                (events_times_samples, b_events_times_samples), dim=-1
            )
            non_pad_mask = torch.cat((non_pad_mask, b_non_pad_mask), dim=-1)

        non_pad_mask, indices = torch.sort(non_pad_mask, dim=-1, descending=True)
        events_times_samples = torch.gather(events_times_samples, dim=-1, index=indices)
        min_zero_num = torch.min(torch.sum(~non_pad_mask, dim=-1))
        if min_zero_num > 0:
            events_times_samples = events_times_samples[..., :-min_zero_num]
            non_pad_mask = non_pad_mask[..., :-min_zero_num]
        events_times_samples[~non_pad_mask] = 1e20
        return events_times_samples, non_pad_mask

    def real_ll(
        self,
        real_events_times,  # [n_samples, batch_size, seq_len]
        real_events_mask,
        parents_real_events_times,
        params,
        parents_real_events_mask,
        batch_end_time,
        batch_ids=None,
        data_type=None,
        parents_stats_ids=None,
    ):
        if self.top:
            if batch_ids is None:
                if data_type == "train":
                    base_rate = self.base_rate
                elif data_type == "dev":
                    base_rate = self.dev_base_rate
                elif data_type == "test":
                    base_rate = self.test_base_rate
                ll_events = torch.log(base_rate).expand_as(
                    real_events_times
                ).clone()
                ll_events[~real_events_mask] = 0
                ll_time = base_rate * batch_end_time
            else:
                if data_type == "train":
                    base_rate = self.base_rate[:, batch_ids, ...]
                elif data_type == "dev":
                    base_rate = self.dev_base_rate[:, batch_ids, ...]
                elif data_type == "test":
                    base_rate = self.test_base_rate[:, batch_ids, ...]
                ll_events = torch.log(
                    base_rate
                ).expand_as(real_events_times).clone()
                ll_events[~real_events_mask] = 0
                ll_time = base_rate * batch_end_time
            return (ll_events * real_events_mask).sum(dim=-1) - ll_time.squeeze(-1)

        (
            diff_time_ll_events,
            params_ll_events,
        ) = self.prepare_params_and_diff_time(
            real_virtual_events_times=real_events_times,
            neighbor_events_times=parents_real_events_times,
            params=params,
            neighbor_real_events_mask=parents_real_events_mask,
            neighbor="parents",
            processes_type=self.processes_type,
            neighbor_stats_ids=parents_stats_ids,
        )
        if self.processes_type == "general":
            ll_events = torch.log(
                self.kernel.forward(
                    diff_time_ll_events, params_ll_events[:, :, :, None, :]
                )
                + F.softplus(self.background_rate)
            )
            times_for_int = (
                F.pad(
                    parents_real_events_times[..., :-1],
                    pad=(0, 1),
                    mode="constant",
                    value=batch_end_time,
                )
                - parents_real_events_times
            )
            ll_time = self.kernel.integral(times_for_int, params)
        else:
            ll_events = torch.log(
                torch.sum(
                    self.kernel.forward(
                        diff_time_ll_events, params_ll_events
                    ),
                    dim=-1,
                )
                + F.softplus(self.background_rate)
            )
            ll_time = self.kernel.integral(
                (batch_end_time - parents_real_events_times)
                * parents_real_events_mask,
                params,
            )
        return (
            (ll_events * real_events_mask).sum(dim=-1)
            - ll_time.sum(dim=-1)
            - (F.softplus(self.background_rate) * batch_end_time.squeeze(-1))
        )

    def prepare_params_and_diff_time(
        self,
        real_virtual_events_times,
        neighbor_events_times,
        params,
        neighbor_real_events_mask,
        neighbor,
        processes_type,
        neighbor_stats_ids,
    ):
        if neighbor == "parents":
            diff_time = (
                real_virtual_events_times[:, :, :, None]
                - neighbor_events_times[:, :, None, :]
            )
        else:
            diff_time = (
                neighbor_events_times[:, :, None, :]
                - real_virtual_events_times[:, :, :, None]
            )

        diff_time[diff_time < 0] = 1e20
        if torch.numel(neighbor_real_events_mask) != 0:
            diff_time[
                (~neighbor_real_events_mask[:, :, None, :]).expand_as(
                    diff_time
                )
            ] = 1e20
        if processes_type == "general":
            (
                diff_time_ll_events,
                neighbor_events_correspondings_ids,
            ) = torch.min(
                diff_time, dim=-1
            )  # [n_samples, batch_size, this_seq_len]

            params_ll_events = torch.gather(
                params,
                dim=-2,
                index=neighbor_events_correspondings_ids[..., None].expand(
                    -1, -1, -1, params.shape[-1]
                ),
            )
        else:
            diff_time_ll_events = diff_time
            if neighbor == "parents":
                params_ll_events = self.kernel_params.unsqueeze(-3).expand(
                    real_virtual_events_times.shape[0],
                    real_virtual_events_times.shape[1],
                    real_virtual_events_times.shape[2],
                    -1,
                    -1,
                )  # [n_sample, n_batch, r_v_seq_len, n_kernel, n_params]
                if params_ll_events.shape[-2] > 1:
                    params_ll_events = params_ll_events.gather(
                        dim=-2,
                        index=neighbor_stats_ids[:, :, None, :, None].expand(-1, -1, params_ll_events.shape[-3], -1, params_ll_events.shape[-1]),
                    )  # [n_sample, n_batch, r_v_seq_len, n_seq_len, n_params]
            else:
                params_ll_events = self.virtual_kernel_params.unsqueeze(-3).expand(
                    real_virtual_events_times.shape[0],
                    real_virtual_events_times.shape[1],
                    real_virtual_events_times.shape[2],
                    -1,
                    -1,
                )  # [n_sample, n_batch, r_v_seq_len, n_kernel, n_params]
                if params_ll_events.shape[-2] > 1:
                    params_ll_events = params_ll_events.gather(
                        dim=-2,
                        index=neighbor_stats_ids[:, :, None, :, None].expand(-1, -1, params_ll_events.shape[-3], -1, params_ll_events.shape[-1])
                    )
        return diff_time_ll_events, params_ll_events
