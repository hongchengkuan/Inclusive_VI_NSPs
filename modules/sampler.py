from tabnanny import check
import torch
import preprocess_dataset
from utils import get_children_events, get_parents_events, assign_events_to_batch, get_parents_params_mask, update_real_loglikelihood
import copy
import time
from collections import namedtuple
import math

PAD = preprocess_dataset.PAD

F = torch.nn.functional


class PriorSampler:
    def __init__(self, PPs, events_embedding) -> None:
        self.PPs = PPs
        self.events_embedding = events_embedding

    def sample(self, batch_size, plot, return_samples=False, n_samples=1, predict=False, device=None):
        if return_samples:
            pure_virtual_samples_events_times_dict = {}
            pure_virtual_samples_events_mask_dict = {}
            for pp in self.PPs.PPs_lst:
                if not pp.bottom:
                    pure_virtual_samples_events_times_dict[pp.id] = []
                    pure_virtual_samples_events_mask_dict[pp.id] = []
        else:
            pure_virtual_samples_events_times_dict = None
            pure_virtual_samples_events_mask_dict = None
        PPs_dev = self.PPs.dev
        PPs_test = self.PPs.test
        for i in range(n_samples):
            for pp in self.PPs.PPs_lst:
                if not pp.bottom:
                    pp.set_events_mask_none(dev=PPs_dev, test=PPs_test)
                    if not predict:
                        pp.real_loglikelihood = torch.tensor([-float('inf')]).expand(1, self.PPs.PPs_lst[0].real_events_times.shape[-2]).clone()
                        pp.virtual_loglikelihood = torch.tensor([-float('inf')]).expand(1, self.PPs.PPs_lst[0].real_events_times.shape[-2]).clone()
                    if PPs_dev:
                        pp.dev_real_loglikelihood = torch.tensor([-float('inf')]).expand(1, self.PPs.PPs_lst[0].dev_real_events_times.shape[-2]).clone()
                        pp.dev_virtual_loglikelihood = torch.tensor([-float('inf')]).expand(1, self.PPs.PPs_lst[0].dev_real_events_times.shape[-2]).clone()
                    if PPs_test:
                        pp.test_real_loglikelihood = torch.tensor([-float('inf')]).expand(1, self.PPs.PPs_lst[0].test_real_events_times.shape[-2]).clone()
                        pp.test_virtual_loglikelihood = torch.tensor([-float('inf')]).expand(1, self.PPs.PPs_lst[0].test_real_events_times.shape[-2]).clone()
                else:
                    if not predict:
                        pp.real_loglikelihood = torch.tensor([-float('inf')]).expand(1, pp.real_events_times.shape[-2]).clone()
                    if PPs_dev:
                        pp.dev_real_loglikelihood = torch.tensor([-float('inf')]).expand(1, pp.dev_real_events_times.shape[-2]).clone()
                    if PPs_test:
                        pp.test_real_loglikelihood = torch.tensor([-float('inf')]).expand(1, pp.test_real_events_times.shape[-2]).clone()
            for evidence_id in self.PPs.evidence_ids_set:
                if not predict:
                    self.sample_parent(
                        evidence_id,
                        batch_size,
                        pure_virtual_samples_events_times_dict,
                        pure_virtual_samples_events_mask_dict,
                        plot,
                        "train",
                        predict,
                        device,
                    )
                if PPs_dev:
                    with torch.random.fork_rng(enabled=True, devices=[]):
                        self.sample_parent(
                            evidence_id,
                            batch_size,
                            pure_virtual_samples_events_times_dict,
                            pure_virtual_samples_events_mask_dict,
                            plot,
                            "dev",
                            predict,
                            device,
                        )
                if PPs_test or predict:
                    with torch.random.fork_rng(enabled=True if not predict else False, devices=[]):
                        self.sample_parent(
                            evidence_id,
                            batch_size,
                            pure_virtual_samples_events_times_dict,
                            pure_virtual_samples_events_mask_dict,
                            plot,
                            "test",
                            predict,
                            device,
                        )

        if return_samples:
            return (
                pure_virtual_samples_events_times_dict,
                pure_virtual_samples_events_mask_dict,
            )

    @staticmethod
    def prior_sampler_get_children_events(childrenPPs, data_type, batch_ids, device=None):
        if data_type == "train":
            children_real_events_times = torch.cat(
                [pp.real_events_times[:, batch_ids, :] for pp in childrenPPs], dim=-1
            )  # [1, batch_size, seq_len]
            children_real_events_ids = torch.cat(
                [
                    torch.tensor([pp.id], device=device).expand_as(pp.real_events_times[:, batch_ids, :])
                    for pp in childrenPPs
                ],
                dim=-1,
            )  # [1, batch_size, seq_len]
            children_real_events_mask = torch.cat(
                [pp.real_events_mask[:, batch_ids, :] for pp in childrenPPs], dim=-1
            )  # [1, batch_size, seq_len]
        elif data_type == "dev":
            children_real_events_times = torch.cat(
                [pp.dev_real_events_times[:, batch_ids, :] for pp in childrenPPs], dim=-1
            )  # [1, batch_size, seq_len]
            children_real_events_ids = torch.cat(
                [
                    torch.tensor([pp.id], device=device).expand_as(pp.dev_real_events_times[:, batch_ids, :])
                    for pp in childrenPPs
                ],
                dim=-1,
            )  # [1, batch_size, seq_len]
            children_real_events_mask = torch.cat(
                [pp.dev_real_events_mask[:, batch_ids, :] for pp in childrenPPs], dim=-1
            )  # [1, batch_size, seq_len]
        elif data_type == "test":
            children_real_events_times = torch.cat(
                [pp.test_real_events_times[:, batch_ids, :] for pp in childrenPPs], dim=-1
            )  # [1, batch_size, seq_len]
            children_real_events_ids = torch.cat(
                [
                    torch.tensor([pp.id], device=device).expand_as(pp.test_real_events_times[:, batch_ids, :])
                    for pp in childrenPPs
                ],
                dim=-1,
            )  # [1, batch_size, seq_len]
            children_real_events_mask = torch.cat(
                [pp.test_real_events_mask[:, batch_ids, :] for pp in childrenPPs], dim=-1
            )  # [1, batch_size, seq_len]

        return (
            children_real_events_times.to(device),
            children_real_events_mask.to(device),
            children_real_events_ids,
        )

    def sample_parent(
        self,
        to_sample_id,
        batch_size,
        pure_virtual_samples_events_times_dict,
        pure_virtual_samples_events_mask_dict,
        plot,
        data_type,
        predict,
        device,
    ):
        PPs_lst = self.PPs.PPs_lst
        thisPP = PPs_lst[to_sample_id]
        if thisPP.parents_ids_dict:
            for p_id in thisPP.parents_ids_dict:
                parentPP = PPs_lst[p_id]

                if data_type == "train":
                    if parentPP.real_events_times is not None:
                        self.sample_parent(p_id, batch_size, pure_virtual_samples_events_times_dict, pure_virtual_samples_events_mask_dict, plot, data_type, predict, device)
                        continue
                elif data_type == "dev":
                    if parentPP.dev_real_events_times is not None:
                        self.sample_parent(p_id, batch_size, pure_virtual_samples_events_times_dict, pure_virtual_samples_events_mask_dict, plot, data_type, predict, device)
                        continue
                elif data_type == "test":
                    if parentPP.test_real_events_times is not None:
                        self.sample_parent(p_id, batch_size, pure_virtual_samples_events_times_dict, pure_virtual_samples_events_mask_dict, plot, data_type, predict, device)
                        continue
                children_ids_dict = parentPP.children_ids_dict
                childrenPPs = [PPs_lst[c_id] for c_id in children_ids_dict]

                childrenPPsNotDone = False
                for childPP in childrenPPs:
                    if data_type == "train":
                        if childPP.real_events_times is None:
                            childrenPPsNotDone = True
                            break
                    elif data_type == "dev":
                        if childPP.dev_real_events_times is None:
                            childrenPPsNotDone = True
                            break
                    elif data_type == "test":
                        if childPP.test_real_events_times is None:
                            childrenPPsNotDone = True
                            break

                if not childrenPPsNotDone:

                    if data_type == "train":
                        whole_data_length = PPs_lst[0].real_events_times.shape[-2]
                        num_iters = math.ceil(whole_data_length / batch_size)
                    elif data_type == "dev":
                        whole_data_length = PPs_lst[0].dev_real_events_times.shape[-2]
                        num_iters = math.ceil(whole_data_length / batch_size)
                    elif data_type == "test":
                        whole_data_length = PPs_lst[0].test_real_events_times.shape[-2]
                        num_iters = math.ceil(whole_data_length / batch_size)
                    samples = []
                    for i in range(num_iters):
                        batch_start = i * batch_size
                        batch_end = (i + 1) * batch_size
                        if batch_end > whole_data_length:
                            batch_end = whole_data_length
                        batch_ids=list(range(batch_start, batch_end))

                        (
                            children_real_events_times,
                            children_real_events_mask,
                            children_real_events_ids,
                        ) = self.prior_sampler_get_children_events(
                            childrenPPs=childrenPPs,
                            data_type=data_type,
                            batch_ids=batch_ids,
                            device=device,
                        )
                        children_real_events_times[~children_real_events_mask] = 1e20
                        children_real_events_times, indices = torch.sort(
                            children_real_events_times, dim=-1
                        )
                        children_real_events_mask = torch.gather(
                            children_real_events_mask, dim=-1, index=indices
                        )
                        children_real_events_ids = torch.gather(
                            children_real_events_ids, dim=-1, index=indices
                        )
                        if thisPP.synthetic_end:
                            children_real_events_times = F.pad(
                                children_real_events_times,
                                pad=(0, 1),
                                mode="constant",
                                value=self.PPs.end_time if data_type == "train" else self.PPs.dev_end_time if data_type == "dev" else self.PPs.test_end_time,
                            )
                            children_real_events_ids = F.pad(
                                children_real_events_ids,
                                pad=(0, 1),
                                mode="constant",
                                value=parentPP.id,
                            )
                            children_real_events_mask = F.pad(
                                children_real_events_mask,
                                pad=(0, 1),
                                mode="constant",
                                value=True,
                            )
                        numel_children = torch.numel(children_real_events_times)
                        if numel_children != 0:
                            children_real_events_embeddings = self.events_embedding(
                                x=children_real_events_times,
                                PPId=children_real_events_ids,
                            )  # [batch_size, seq_len, embedding]
                            if parentPP.virtual_processes_type == "general":
                                params = parentPP.calc_params(
                                    neighbor_real_events_embeddings=children_real_events_embeddings,
                                    neighbor_mask=children_real_events_mask.unsqueeze(-1),
                                    virtual=True,
                                )
                                children_real_events_stats_ids = None
                            else:
                                children_shape = children_real_events_times.shape
                                params = parentPP.virtual_kernel_params.expand(children_shape[0], children_shape[1], -1, -1) # [n_samples, n_batches, n_kernel, n_params]
                                children_real_events_stats_ids = children_real_events_ids.clone()
                                for key, val in parentPP.children_ids_dict.items():
                                    children_real_events_stats_ids = children_real_events_stats_ids.masked_fill(children_real_events_stats_ids == key, val)
                                if params.shape[-2] > 1:
                                    params = params.gather(dim=-2, index=children_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, params.shape[-1]))
                                    # [n_samples, n_batches, n_seq_len, n_params]
                        samples.append(
                            parentPP.sample_virtual_events(
                                params if numel_children != 0 else None,
                                children_real_events_times,
                                children_real_events_mask,
                                batch_end_time=self.PPs.end_time[:, batch_ids, ...] if data_type == "train" else self.PPs.dev_end_time[:, batch_ids, ...] if data_type == "dev" else self.PPs.test_end_time[:, batch_ids, ...],
                                prop_virtual_background=True
                                if not plot
                                else False,
                                device=device,
                            )
                        )
                        if pure_virtual_samples_events_times_dict is None:
                            if not predict:
                                if data_type == "train":
                                    parentPP.virtual_loglikelihood[:, batch_start:batch_end, ...] = parentPP.virtual_ll(
                                        samples[-1][0],
                                        samples[-1][1],
                                        children_real_events_times,
                                        params if numel_children != 0 else None,
                                        children_real_events_mask,
                                        batch_end_time=self.PPs.end_time[:, batch_ids, ...],
                                        prop_virtual_background=True
                                        if not plot
                                        else False,
                                        children_stats_ids=children_real_events_stats_ids,
                                    )
                                elif data_type == "dev":
                                    parentPP.dev_virtual_loglikelihood[:, batch_ids, ...] = parentPP.virtual_ll(
                                        samples[-1][0],
                                        samples[-1][1],
                                        children_real_events_times,
                                        params,
                                        children_real_events_mask,
                                        batch_end_time=self.PPs.dev_end_time[:, batch_ids, ...],
                                        prop_virtual_background=True
                                        if not plot
                                        else False,
                                        children_stats_ids=children_real_events_stats_ids,
                                    )
                                elif data_type == "test":
                                    parentPP.test_virtual_loglikelihood[:, batch_ids, ...] = parentPP.virtual_ll(
                                        samples[-1][0],
                                        samples[-1][1],
                                        children_real_events_times,
                                        params,
                                        children_real_events_mask,
                                        batch_end_time=self.PPs.test_end_time[:, batch_ids, ...],
                                        prop_virtual_background=True
                                        if not plot
                                        else False,
                                        children_stats_ids=children_real_events_stats_ids,
                                    )
                        else:
                            if data_type == "train":
                                pure_virtual_samples_events_times_dict[p_id].append(
                                    parentPP.virtual_events_times.cpu()
                                )
                                pure_virtual_samples_events_mask_dict[p_id].append(
                                    parentPP.virtual_events_mask.cpu()
                                )
                            elif data_type == "dev":
                                pure_virtual_samples_events_times_dict[p_id].append(
                                    parentPP.dev_virtual_events_times.cpu()
                                )
                                pure_virtual_samples_events_mask_dict[p_id].append(
                                    parentPP.dev_virtual_events_mask.cpu()
                                )
                            elif data_type == "test":
                                pure_virtual_samples_events_times_dict[p_id].append(
                                    parentPP.test_virtual_events_times.cpu()
                                )
                                pure_virtual_samples_events_mask_dict[p_id].append(
                                    parentPP.test_virtual_events_mask.cpu()
                                )
                    real_events_times, real_events_mask = list(zip(*samples))
                    if len(real_events_times) > 1:
                        max_len = max(
                            sample.shape[-1] for sample in real_events_times
                        )
                        real_events_times = torch.cat(
                            [
                                torch.cat(
                                    (
                                        sample,
                                        torch.ones(
                                            1,
                                            sample.shape[-2],
                                            max_len - sample.shape[-1],
                                            device=device,
                                        )
                                        * PAD,
                                    ),
                                    dim=-1,
                                )
                                for sample in real_events_times
                            ],
                            dim=-2,
                        )
                        real_events_mask = torch.cat(
                            [
                                torch.cat(
                                    (
                                        sample,
                                        torch.zeros(
                                            1,
                                            sample.shape[-2],
                                            max_len - sample.shape[-1],
                                            device=device,
                                        ).bool(),
                                    ),
                                    dim=-1,
                                )
                                for sample in real_events_mask
                            ],
                            dim=-2,
                        )
                        if data_type == "train":
                            parentPP.real_events_times = real_events_times
                            parentPP.real_events_mask = real_events_mask
                        elif data_type == "dev":
                            parentPP.dev_real_events_times = real_events_times
                            parentPP.dev_real_events_mask = real_events_mask
                        elif data_type == "test":
                            parentPP.test_real_events_times = real_events_times
                            parentPP.test_real_events_mask = real_events_mask
                    else:
                        if data_type == "train":
                            parentPP.real_events_times = real_events_times[0]
                            parentPP.real_events_mask = real_events_mask[0]
                        elif data_type == "dev":
                            parentPP.dev_real_events_times = real_events_times[
                                0
                            ]
                            parentPP.dev_real_events_mask = real_events_mask[0]
                        elif data_type == "test":
                            parentPP.test_real_events_times = real_events_times[
                                0
                            ]
                            parentPP.test_real_events_mask = real_events_mask[
                                0
                            ]
                    if data_type == "train":
                        parentPP.virtual_events_times = (
                            parentPP.real_events_times.clone().detach()
                        )
                        parentPP.virtual_events_mask = (
                            parentPP.real_events_mask.clone().detach()
                        )
                    elif data_type == "dev":
                        parentPP.dev_virtual_events_times = (
                            parentPP.dev_real_events_times.clone().detach()
                        )
                        parentPP.dev_virtual_events_mask = (
                            parentPP.dev_real_events_mask.clone().detach()
                        )
                    elif data_type == "test":
                        parentPP.test_virtual_events_times = (
                            parentPP.test_real_events_times.clone().detach()
                        )
                        parentPP.test_virtual_events_mask = (
                            parentPP.test_real_events_mask.clone().detach()
                        )

                    self.sample_parent(
                        p_id,
                        batch_size,
                        pure_virtual_samples_events_times_dict,
                        pure_virtual_samples_events_mask_dict,
                        plot,
                        data_type,
                        predict,
                        device,
                    )

    def sample_top_down(self, batch_size):
        for top_id in self.PPs.top_ids_set:
            thisPP = self.PPs.PPs_lst[top_id]

            b_integral = thisPP.base_rate[:, 0:batch_size, :] * self.PPs.end_time[:, 0:batch_size, :]
            b_poisson_random_num = torch.poisson(b_integral)
            b_max_poisson_random_num = torch.max(b_poisson_random_num).int()
            thisPP.real_events_times = (
                torch.rand(1, batch_size, b_max_poisson_random_num)
                * self.PPs.end_time[:, 0:batch_size, :]
            )
            b_non_pad_mask = torch.arange(b_max_poisson_random_num)[
                None, :
            ].expand(1, batch_size, -1)
            thisPP.real_events_mask = b_non_pad_mask < b_poisson_random_num

            self.sample_child(top_id, batch_size)

    def sample_child(self, to_sample_id, batch_size):
        PPs_lst = self.PPs.PPs_lst
        thisPP = PPs_lst[to_sample_id]
        if thisPP.children_ids_dict:
            for p_id in thisPP.children_ids_dict:
                childPP = PPs_lst[p_id]

                parents_ids_dict = childPP.parents_ids_dict
                parentsPPs = [PPs_lst[p_id] for p_id in parents_ids_dict]

                parentsPPsNotDone = False
                for parentPP in parentsPPs:
                    if parentPP.real_events_times is None:
                        parentsPPsNotDone = True
                        break

                if not parentsPPsNotDone:
                    parents_real_events_times = torch.cat(
                        [pp.real_events_times for pp in parentsPPs], dim=-1
                    )
                    parents_real_events_ids = torch.cat(
                        [
                            torch.ones_like(pp.real_events_times) * pp.id
                            for pp in parentsPPs
                        ],
                        dim=-1,
                    )
                    parents_real_events_mask = torch.cat(
                        [pp.real_events_mask for pp in parentsPPs], dim=-1
                    )
                    parents_real_events_embeddings = self.events_embedding(
                        x=parents_real_events_times,
                        PPId=parents_real_events_ids,
                    )

                    num_iters = (
                        math.ceil(parents_real_events_times.shape[-2] / batch_size)
                    )
                    parents_real_events_times[~parents_real_events_mask] = 1e20
                    parents_real_events_times, indices = torch.sort(
                        parents_real_events_times, dim=-1
                    )
                    parents_real_events_mask = torch.gather(
                        parents_real_events_mask, dim=-1, index=indices
                    )
                    parents_real_events_embeddings = torch.gather(
                        parents_real_events_embeddings,
                        dim=-2,
                        index=indices[..., None].expand_as(
                            parents_real_events_embeddings
                        ),
                    )
                    if childPP.processes_type == "general":
                        params = childPP.calc_params(
                            neighbor_real_events_embeddings=parents_real_events_embeddings,
                            neighbor_mask=parents_real_events_mask.unsqueeze(-1),
                            virtual=False,
                        )
                    else:
                        parents_shape = parents_real_events_times.shape
                        parents_real_events_stats_ids = parents_real_events_ids.clone()
                        for key, val in parentPP.children_ids_dict.items():
                            parents_real_events_stats_ids = parents_real_events_stats_ids.masked_fill(parents_real_events_stats_ids == key, val)
                        params = childPP.kernel_params.expand(parents_shape[0], parents_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
                        if params.shape[-2] > 1:
                            params = params.gather(dim=-2, index=parents_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, params.shape[-1]))
                    samples = [
                        childPP.sample_real_events(
                            params[
                                :, i * batch_size : (i + 1) * batch_size, ...
                            ],
                            parents_real_events_times[
                                :, i * batch_size : (i + 1) * batch_size, ...
                            ],
                            parents_real_events_mask[
                                :, i * batch_size : (i + 1) * batch_size, ...
                            ],
                            batch_end_time=self.PPs.end_time[:, i * batch_size : (i + 1) * batch_size, ...]
                        )
                        for i in range(num_iters)
                    ]
                    real_events_times, real_events_mask = list(zip(*samples))
                    if len(real_events_times) > 1:
                        max_len = max(
                            sample.shape[-1] for sample in real_events_times
                        )
                        real_events_times = torch.cat(
                            [
                                torch.cat(
                                    (
                                        sample,
                                        torch.ones(
                                            1,
                                            sample.shape[-2],
                                            max_len - sample.shape[1],
                                        )
                                        * PAD,
                                    ),
                                    dim=-1,
                                )
                                for sample in real_events_times
                            ],
                            dim=-2,
                        )
                        real_events_mask = torch.cat(
                            [
                                torch.cat(
                                    (
                                        sample,
                                        torch.zeros(
                                            1,
                                            sample.shape[-2],
                                            max_len - sample.shape[1],
                                        ).bool(),
                                    ),
                                    dim=-1,
                                )
                                for sample in real_events_mask
                            ],
                            dim=-2,
                        )
                        childPP.real_events_times = real_events_times
                        childPP.real_events_mask = real_events_mask
                    else:
                        childPP.real_events_times = real_events_times[0]
                        childPP.real_events_mask = real_events_mask[0]
                    self.sample_child(p_id, batch_size)


class PosteriorSampler:
    def __init__(
        self,
        PPs,
        events_embedding,
        resample_only,
        record_virtual_samples,
        check_real_ll,
        device,
        predict=False,
    ) -> None:
        self.PPs = PPs
        self.events_embedding = events_embedding
        self.resample_only = resample_only
        self.record_virtual_samples = record_virtual_samples
        self.check_real_ll = check_real_ll
        if check_real_ll:
            self.prior_sampler_check_real_ll = PriorSampler(
                self.PPs, self.events_embedding
            )
        self.device = device
        self.predict = predict

    def initialize_ll(self, initialize_batch_size, plot, device):
        with torch.no_grad():
            prior_sampler = PriorSampler(self.PPs, self.events_embedding)
            prior_sampler.sample(initialize_batch_size, plot, predict=self.predict, device=device)

            PPs_lst = self.PPs.PPs_lst
            Data_type_named_tuple = namedtuple('Data_type_named_tuple', ['train', 'dev', 'test'])
            if not self.predict:
                data_type_named_tuple = Data_type_named_tuple(True, self.PPs.dev, self.PPs.test)
            else:
                data_type_named_tuple = Data_type_named_tuple(False, False, True)
            for field, value in zip(data_type_named_tuple._fields, data_type_named_tuple):
                if not value:
                    continue
                batch_real_events_times = {}
                batch_real_events_mask = {}
                    
                for pp in self.PPs.PPs_lst:
                    if field == "train":
                        batch_real_events_times[pp.id] = pp.real_events_times
                        batch_real_events_mask[pp.id] = pp.real_events_mask
                    elif field == "dev":
                        batch_real_events_times[pp.id] = pp.dev_real_events_times
                        batch_real_events_mask[pp.id] = pp.dev_real_events_mask
                    elif field == "test":
                        batch_real_events_times[pp.id] = pp.test_real_events_times
                        batch_real_events_mask[pp.id] = pp.test_real_events_mask
                for thisPP in PPs_lst:
                    parents_ids_dict = thisPP.parents_ids_dict
                    if field == "train":
                        num_iters = (
                            math.ceil(thisPP.real_events_times.shape[-2] / initialize_batch_size)
                        )
                    elif field == "dev":
                        num_iters = (
                            math.ceil(thisPP.dev_real_events_times.shape[-2] / initialize_batch_size)
                        )
                    else:
                        num_iters = (
                            math.ceil(thisPP.test_real_events_times.shape[-2] / initialize_batch_size)
                        )
                    if parents_ids_dict is not None:
                        (
                            parents_real_events_times,
                            parents_real_events_mask,
                            parents_real_events_ids,
                            _,
                        ) = get_parents_events(
                            thisPP, batch_real_events_times, batch_real_events_mask, device=device
                        )

                    else:
                        parents_real_events_stats_ids = None
                    for i in range(num_iters):
                        batch_start = i * initialize_batch_size
                        batch_end = (i + 1) * initialize_batch_size
                        if batch_end > parents_real_events_times.shape[-2]:
                            batch_end = parents_real_events_times.shape[-2]
                        if parents_ids_dict:
                            if thisPP.processes_type == "general":
                                parents_real_events_embeddings = self.PPs.events_embedding(
                                    parents_real_events_times[:, batch_start:batch_end, ...], parents_real_events_ids[:, batch_start:batch_end, ...]
                                )
                                parents_real_events_params = thisPP.calc_params(
                                    neighbor_real_events_embeddings=parents_real_events_embeddings,
                                    neighbor_mask=parents_real_events_mask[:, batch_start:batch_end, ...],
                                    virtual=False,
                                )
                                parents_real_events_stats_ids = None
                            else:
                                parents_shape = parents_real_events_times[:, batch_start:batch_end, ...].shape
                                parents_real_events_stats_ids = parents_real_events_ids[:, batch_start:batch_end, ...].clone()
                                for key, val in thisPP.parents_ids_dict.items():
                                    parents_real_events_stats_ids = parents_real_events_stats_ids.masked_fill(parents_real_events_stats_ids == key, val)
                                parents_real_events_params = thisPP.kernel_params.expand(parents_shape[0], parents_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
                                if parents_real_events_params.shape[-2] > 1:
                                    parents_real_events_params = parents_real_events_params.gather(dim=-2, index=parents_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, parents_real_events_params.shape[-1]))
                        else:
                            parents_real_events_stats_ids = None
                        if field == "train":
                            thisPP.real_loglikelihood[:, batch_start:batch_end, ...] = thisPP.real_ll(
                                thisPP.real_events_times[:, batch_start:batch_end, ...],
                                thisPP.real_events_mask[:, batch_start:batch_end, ...],
                                parents_real_events_times[:, batch_start:batch_end, ...]
                                if parents_ids_dict is not None
                                else None,
                                parents_real_events_params
                                if parents_ids_dict is not None
                                else None,
                                parents_real_events_mask[:, batch_start:batch_end, ...]
                                if parents_ids_dict is not None
                                else None,
                                batch_end_time=self.PPs.end_time[:, batch_start:batch_end, ...],
                                batch_ids=list(range(batch_start, batch_end)),
                                data_type="train",
                                parents_stats_ids=parents_real_events_stats_ids,
                            )
                        elif field == "dev":
                            thisPP.dev_real_loglikelihood[:, batch_start:batch_end, ...] = thisPP.real_ll(
                                thisPP.dev_real_events_times[:, batch_start:batch_end, ...],
                                thisPP.dev_real_events_mask[:, batch_start:batch_end, ...],
                                parents_real_events_times[:, batch_start:batch_end, ...]
                                if parents_ids_dict is not None
                                else None,
                                parents_real_events_params
                                if parents_ids_dict is not None
                                else None,
                                parents_real_events_mask[:, batch_start:batch_end, ...]
                                if parents_ids_dict is not None
                                else None,
                                batch_end_time=self.PPs.dev_end_time[:, batch_start:batch_end, ...],
                                batch_ids=list(range(batch_start, batch_end)),
                                data_type="dev",
                                parents_stats_ids=parents_real_events_stats_ids,
                            )
                        elif field == "test":
                            thisPP.test_real_loglikelihood[:, batch_start:batch_end, ...] = thisPP.real_ll(
                                thisPP.test_real_events_times[:, batch_start:batch_end, ...],
                                thisPP.test_real_events_mask[:, batch_start:batch_end, ...],
                                parents_real_events_times[:, batch_start:batch_end, ...]
                                if parents_ids_dict is not None
                                else None,
                                parents_real_events_params
                                if parents_ids_dict is not None
                                else None,
                                parents_real_events_mask[:, batch_start:batch_end, ...]
                                if parents_ids_dict is not None
                                else None,
                                batch_end_time=self.PPs.test_end_time[:, batch_start:batch_end, ...],
                                batch_ids=list(range(batch_start, batch_end)),
                                data_type="test",
                                parents_stats_ids=parents_real_events_stats_ids,
                            )


    def get_children_params_mask(
        self, thisPP, batch_real_events_times, batch_real_events_mask, batch_virtual_loglikelihood, return_virtual_samples, batch_end_time
    ):
        thisPP_id = thisPP.id
        virtual_processes_type = thisPP.virtual_processes_type
        (
            children_real_events_times,  # [1, batch_size, seq_len]
            children_real_events_mask,
            children_real_events_ids,
            _,
        ) = get_children_events(thisPP, batch_real_events_times, batch_real_events_mask, device=self.device)
        # children_real_events_times sorted in an ascending order
        if thisPP.synthetic_end:
            children_real_events_times = F.pad(
                children_real_events_times,
                pad=(0, 1),
                mode="constant",
                value=thisPP.end,
            )
            children_real_events_mask = F.pad(
                children_real_events_mask,
                pad=(0, 1),
                mode="constant",
                value=True,
            )
            children_real_events_ids = F.pad(
                children_real_events_ids,
                pad=(0, 1),
                mode="constant",
                value=thisPP_id,
            )
        numel_children = torch.numel(children_real_events_times)
        if numel_children != 0:
            if virtual_processes_type == "general":
                children_real_events_embeddings = self.PPs.events_embedding(
                    children_real_events_times, children_real_events_ids
                )
                params = thisPP.calc_params(
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
                params = thisPP.virtual_kernel_params.expand(children_shape[0], children_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
                if params.shape[-2] > 1:
                    params = params.gather(dim=-2, index=children_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, params.shape[-1]))
        else:
            params = None
            children_real_events_stats_ids = None
            
        (
            sampled_virtual_events,
            sampled_virtual_events_mask,  # boolean
        ) = thisPP.sample_virtual_events(
            params,
            children_real_events_times,
            children_real_events_mask,
            batch_end_time=batch_end_time,
            prop_virtual_background=True,
            device=self.device,
        )

        batch_virtual_loglikelihood[thisPP_id] = thisPP.virtual_ll(
            sampled_virtual_events,
            sampled_virtual_events_mask,
            children_real_events_times,
            params,
            children_real_events_mask,
            batch_end_time=batch_end_time,
            prop_virtual_background=True,
            children_stats_ids=children_real_events_stats_ids,
        )

        real_virtual_events_times = torch.cat(
            (batch_real_events_times[thisPP_id], sampled_virtual_events),
            -1,
        )
        real_virtual_events_mask = torch.cat(
            (
                batch_real_events_mask[thisPP_id],
                -1 * sampled_virtual_events_mask,
            ),
            -1,
        )
        # real 1; virtual -1; pad -2
        _real_virtual_events_mask = real_virtual_events_mask.clone()
        _real_virtual_events_mask[_real_virtual_events_mask == 0] = -2
        _, indices = torch.sort(
            _real_virtual_events_mask, descending=True
        )  # real -> virtual -> pad
        real_virtual_events_mask = torch.gather(
            real_virtual_events_mask, -1, indices
        )
        real_virtual_events_times = torch.gather(
            real_virtual_events_times, -1, indices
        )

        # virtual params
        if numel_children != 0:
            (
                children_diff_time_ll_events,
                params_ll_events,
            ) = thisPP.prepare_params_and_diff_time(
                real_virtual_events_times=real_virtual_events_times,
                neighbor_events_times=children_real_events_times,
                params=params,
                neighbor_real_events_mask=children_real_events_mask,
                neighbor="children",
                processes_type=virtual_processes_type,
                neighbor_stats_ids=children_real_events_stats_ids,
            )
        else:
            children_diff_time_ll_events = None
            params_ll_events = None
        if return_virtual_samples:
            return (
                real_virtual_events_times,
                real_virtual_events_mask,
                params,
                params_ll_events,
                children_diff_time_ll_events,
                children_real_events_times,
                children_real_events_mask,
            )
        else:
            return (
                params,
                params_ll_events,
                children_diff_time_ll_events,
                children_real_events_times,
                children_real_events_mask,
            )


    def sample(self, batch_ids, burn_in_steps, sample_size, sample_intervals, data_type="train", batch_end_time=None, first_after_opt=True):
        with torch.no_grad():
            samples_events_times_dict = {}
            samples_events_mask_dict = {}
            record_virtual_samples = self.record_virtual_samples
            if record_virtual_samples:
                virtual_samples_events_times_dict = {}
                virtual_samples_events_mask_dict = {}
            for pp in self.PPs.PPs_lst:
                if not pp.bottom or self.check_real_ll:
                    samples_events_times_dict[pp.id] = []
                    samples_events_mask_dict[pp.id] = []
                    if record_virtual_samples:
                        virtual_samples_events_times_dict[pp.id] = []
                        virtual_samples_events_mask_dict[pp.id] = []

            batch_real_events_times = {}
            batch_real_events_mask = {}
            batch_virtual_events_times = {}
            batch_virtual_events_mask = {}
            batch_real_loglikelihood = {}
            batch_virtual_loglikelihood = {}
            batch_base_rate = {}
            assign_events_to_batch(
                batch_real_events_times,
                batch_real_events_mask,
                batch_virtual_events_times,
                batch_virtual_events_mask,
                batch_real_loglikelihood,
                batch_virtual_loglikelihood,
                batch_base_rate,
                self.PPs.PPs_lst,
                batch_ids,
                self.device,
                data_type,
            )
            if first_after_opt:
                before_opt_sum = 0
                for pp in self.PPs.PPs_lst:
                    before_opt_sum += torch.sum(batch_real_loglikelihood[pp.id])
                print("before opt sum = ", before_opt_sum)
                update_real_loglikelihood(
                    self.PPs,
                    batch_ids,
                    batch_real_events_times,
                    batch_real_events_mask,
                    batch_real_loglikelihood,
                    batch_end_time,
                    data_type,
                    self.device,
                )
                after_opt_sum = 0
                for pp in self.PPs.PPs_lst:
                    after_opt_sum += torch.sum(batch_real_loglikelihood[pp.id])
                print("after opt sum = ", after_opt_sum)
            for b in range(burn_in_steps):
                self.sample_one_scan(
                    batch_ids, 
                    batch_real_events_times, 
                    batch_real_events_mask, 
                    batch_virtual_events_times, 
                    batch_virtual_events_mask, 
                    batch_real_loglikelihood, 
                    batch_virtual_loglikelihood, 
                    batch_base_rate, 
                    self.resample_only, 
                    b,
                    True
                    if (b == burn_in_steps - 1)
                    else False,
                    data_type,
                    batch_end_time,
                )
            if burn_in_steps > 0:
                return None, None

            time1 = time.time()
            for ss in range(sample_size):
                if self.check_real_ll:
                    self.prior_sampler_check_real_ll.sample_top_down(
                        len(batch_ids)
                    )
                    assign_events_to_batch(
                        batch_real_events_times,
                        batch_real_events_mask,
                        batch_virtual_events_times,
                        batch_virtual_events_mask,
                        batch_real_loglikelihood,
                        batch_virtual_loglikelihood,
                        batch_base_rate,
                        self.PPs.PPs_lst,
                        batch_ids,
                        None,
                        data_type,
                    )
                else:
                    for si in range(sample_intervals):
                        self.sample_one_scan(
                            batch_ids,
                            batch_real_events_times,
                            batch_real_events_mask,
                            batch_virtual_events_times,
                            batch_virtual_events_mask,
                            batch_real_loglikelihood,
                            batch_virtual_loglikelihood,
                            batch_base_rate,
                            self.resample_only,
                            ss,
                            True
                            if (ss == sample_size - 1)
                            and (si == sample_intervals - 1)
                            else False,
                            data_type,
                            batch_end_time,
                        )
                for pp in self.PPs.PPs_lst:
                    if not pp.bottom or self.check_real_ll:
                        samples_events_times_dict[pp.id].append(
                            batch_real_events_times[pp.id]
                        )
                        samples_events_mask_dict[pp.id].append(
                            batch_real_events_mask[pp.id]
                        )
                        if record_virtual_samples:
                            virtual_samples_events_times_dict[pp.id].append(
                                batch_virtual_events_times[pp.id]
                            )
                            virtual_samples_events_mask_dict[pp.id].append(
                                batch_virtual_events_mask[pp.id]
                            )
            # print("sampling takes", time.time() - time1)

            batch_size = len(batch_ids)
            for pp in self.PPs.PPs_lst:
                if not pp.bottom or self.check_real_ll:
                    max_len = max(
                        sample.shape[-1]
                        for sample in samples_events_times_dict[pp.id]
                    )
                    samples_events_times_dict[pp.id] = torch.cat(
                        [
                            torch.cat(
                                (
                                    sample,
                                    torch.ones(
                                        1, batch_size, max_len - sample.shape[-1], device=self.device
                                    )
                                    * PAD,
                                ),
                                dim=-1,
                            )
                            for sample in samples_events_times_dict[pp.id]
                        ], dim=0
                    )
                    samples_events_mask_dict[pp.id] = torch.cat(
                        [
                            torch.cat(
                                (
                                    sample,
                                    torch.zeros(
                                        1, batch_size, max_len - sample.shape[-1], device=self.device
                                    ).bool(),
                                ),
                                dim=-1,
                            )
                            for sample in samples_events_mask_dict[pp.id]
                        ], dim=0
                    )
                else:
                    # The observation needs to be expanded to have the same size the the first dimension
                    if data_type =="train":
                        samples_events_times_dict[pp.id] = (
                            pp.real_events_times[:, batch_ids, ...]
                            .expand(sample_size, -1, -1)
                            .to(self.device)
                        )
                        samples_events_mask_dict[pp.id] = (
                            pp.real_events_mask[:, batch_ids, ...]
                            .expand(sample_size, -1, -1)
                            .to(self.device)
                        )
                    elif data_type == "dev":
                        samples_events_times_dict[pp.id] = (
                            pp.dev_real_events_times[:, batch_ids, ...]
                            .expand(sample_size, -1, -1)
                            .to(self.device)
                        )
                        samples_events_mask_dict[pp.id] = (
                            pp.dev_real_events_mask[:, batch_ids, ...]
                            .expand(sample_size, -1, -1)
                            .to(self.device)
                        )
                    elif data_type == "test":
                        samples_events_times_dict[pp.id] = (
                            pp.test_real_events_times[:, batch_ids, ...]
                            .expand(sample_size, -1, -1)
                            .to(self.device)
                        )
                        samples_events_mask_dict[pp.id] = (
                            pp.test_real_events_mask[:, batch_ids, ...]
                            .expand(sample_size, -1, -1)
                            .to(self.device)
                        )
                if record_virtual_samples and not pp.bottom:
                    virtual_max_len = max(
                        virtual_sample.shape[-1]
                        for virtual_sample in virtual_samples_events_times_dict[
                            pp.id
                        ]
                    )
                    virtual_samples_events_times_dict[pp.id] = torch.cat(
                        [
                            torch.cat(
                                (
                                    virtual_sample,
                                    torch.ones(
                                        1,
                                        batch_size,
                                        virtual_max_len
                                        - virtual_sample.shape[-1], device=self.device
                                    )
                                    * PAD,
                                ),
                                dim=-1,
                            )
                            for virtual_sample in virtual_samples_events_times_dict[
                                pp.id
                            ]
                        ], dim=0
                    )
                    virtual_samples_events_mask_dict[pp.id] = torch.cat(
                        [
                            torch.cat(
                                (
                                    virtual_sample,
                                    torch.zeros(
                                        1,
                                        batch_size,
                                        virtual_max_len
                                        - virtual_sample.shape[-1], device=self.device
                                    ).bool(),
                                ),
                                dim=-1,
                            )
                            for virtual_sample in virtual_samples_events_mask_dict[
                                pp.id
                            ]
                        ], dim=0
                    )

            if record_virtual_samples:
                return (
                    samples_events_times_dict,
                    samples_events_mask_dict,
                    virtual_samples_events_times_dict,
                    virtual_samples_events_mask_dict,
                )
            else:
                return samples_events_times_dict, samples_events_mask_dict
    
    def get_p_children(
        self,
        parents_ids_dict,
        thisPPid,
        PPs_lst,
        batch_real_events_times,
        batch_real_events_mask,
        real_virtual_events_times,
        real_virtual_events_mask,
    ):
        if parents_ids_dict is None:
            return None, None, None, None, None, None
        p_children_real_events_times_dict = {}
        p_children_real_events_mask_dict = {}
        p_children_real_events_embeddings_dict = {}
        p_thisPP_shift_dict = {}
        p_children_real_events_sort_indices = {}
        p_children_stats_ids_dict = {}
        for p_id in parents_ids_dict:
            parentPP = PPs_lst[p_id]
            children_ids_string = parentPP.children_ids_string
            if (
                children_ids_string
                in p_children_real_events_times_dict
            ):
                continue

            (
                p_children_real_events_times,
                p_children_real_events_mask_dict[
                    children_ids_string
                ],
                p_children_real_ids,
                p_thisPP_shift_dict[children_ids_string],
                p_children_real_events_sort_indices[
                    children_ids_string
                ],
            ) = get_children_events(
                parentPP,
                batch_real_events_times,
                batch_real_events_mask,
                thisPPid,
                real_virtual_events_times,
                real_virtual_events_mask,
                device=self.device,
            )
            if parentPP.synthetic_end:
                p_children_real_events_times = F.pad(
                    p_children_real_events_times,
                    pad=(0, 1),
                    mode="constant",
                    value=self.PPs.end_time,
                )
                p_children_real_ids = F.pad(
                    p_children_real_ids,
                    pad=(0, 1),
                    mode="constant",
                    value=parentPP.id,
                )
            p_children_real_events_times_dict[
                children_ids_string
            ] = p_children_real_events_times
            p_children_real_events_embeddings_dict[children_ids_string] = self.PPs.events_embedding(
                p_children_real_events_times,
                p_children_real_ids,
            )
            p_children_stats_ids = p_children_real_ids.clone()
            for key, val in parentPP.children_ids_dict.items():
                p_children_stats_ids = p_children_stats_ids.masked_fill(p_children_stats_ids == key, val)
            p_children_stats_ids_dict[children_ids_string] = p_children_stats_ids
        return (
            p_children_real_events_times_dict,
            p_children_real_events_mask_dict,
            p_children_real_events_embeddings_dict,
            p_thisPP_shift_dict,
            p_children_real_events_sort_indices,
            p_children_stats_ids_dict,
        )

    def get_c_parents(
        self, 
        children_ids_dict,
        thisPPid,
        PPs_lst,
        batch_real_events_times,
        batch_real_events_mask,
        real_virtual_events_times,
        real_virtual_events_mask,
    ):
        c_parents_real_events_times_dict = {}
        c_parents_real_events_mask_dict = {}
        c_parents_real_events_embeddings_dict = {} 
        c_thisPP_shift_dict = {}
        c_parents_real_events_sort_indices = {}
        c_parents_stats_ids_dict = {}
        for c_id in children_ids_dict:
            childPP = PPs_lst[c_id]
            parents_ids_string = childPP.parents_ids_string
            if parents_ids_string in c_parents_real_events_times_dict:
                continue

            (
                c_parents_real_events_times,
                c_parents_real_events_mask_dict[parents_ids_string],
                c_parents_real_ids,
                c_thisPP_shift_dict[parents_ids_string],
                c_parents_real_events_sort_indices[parents_ids_string],
            ) = get_parents_events(
                childPP,
                batch_real_events_times,
                batch_real_events_mask,
                thisPPid,
                real_virtual_events_times,
                real_virtual_events_mask,
                device=self.device,
            )
            c_parents_real_events_embeddings_dict[parents_ids_string] = self.PPs.events_embedding(
                c_parents_real_events_times, c_parents_real_ids
            )
            c_parents_real_events_times_dict[parents_ids_string] = c_parents_real_events_times
            c_parents_stats_ids = c_parents_real_ids.clone()
            for key, val in childPP.parents_ids_dict.items():
                c_parents_stats_ids = c_parents_stats_ids.masked_fill(c_parents_stats_ids == key, val)
            c_parents_stats_ids_dict[parents_ids_string] = c_parents_stats_ids
        return (
            c_parents_real_events_times_dict,
            c_parents_real_events_mask_dict,
            c_parents_real_events_embeddings_dict,
            c_thisPP_shift_dict,
            c_parents_real_events_sort_indices,
            c_parents_stats_ids_dict,
        )

    def sample_one_scan(
        self, 
        batch_ids, 
        batch_real_events_times, 
        batch_real_events_mask, 
        batch_virtual_events_times, 
        batch_virtual_events_mask, 
        batch_real_loglikelihood, 
        batch_virtual_loglikelihood, 
        batch_base_rate, 
        resample_only, 
        ss,
        last_scan,
        data_type,
        batch_end_time,
    ):
        if len(self.PPs.hidden_pps_ids_lst) > 1:
            PPIds_randperm = torch.randperm(self.PPs.num_of_hidden_PPs)
            PPIds_randperm = [
                self.PPs.hidden_pps_ids_lst[r] for r in PPIds_randperm
            ]
        else:
            PPIds_randperm = self.PPs.hidden_pps_ids_lst
        PPs_lst = self.PPs.PPs_lst
        for PPId in PPIds_randperm:
            thisPP = PPs_lst[PPId]
            (
                real_virtual_events_times,
                real_virtual_events_mask,
                virtual_kernel_params,
                virtual_kernel_params_ll_events,
                children_diff_time_ll_events,
                children_real_events_times,
                children_real_events_mask,
            ) = self.get_children_params_mask(
                thisPP, batch_real_events_times, batch_real_events_mask, batch_virtual_loglikelihood, return_virtual_samples=True, batch_end_time=batch_end_time,
            )
            if not resample_only:
                if children_diff_time_ll_events is None:
                    virtual_ll_events = torch.log(
                        F.softplus(thisPP.virtual_background_rate) 
                        + thisPP.virtual_prop_background_rate
                    ).expand_as(real_virtual_events_times)
                else:
                    if thisPP.virtual_processes_type == "general":
                        virtual_ll_events = torch.log(
                            thisPP.virtual_kernel.forward(
                                children_diff_time_ll_events,
                                virtual_kernel_params_ll_events,
                            )
                            + F.softplus(thisPP.virtual_background_rate)
                            + thisPP.virtual_prop_background_rate
                        )
                    else:
                        virtual_ll_events = torch.log(
                            torch.sum(
                                thisPP.virtual_kernel.forward(
                                    children_diff_time_ll_events,
                                    virtual_kernel_params_ll_events,
                                ),
                                dim=-1,
                            )
                            + F.softplus(thisPP.virtual_background_rate)
                            + thisPP.virtual_prop_background_rate
                        )
                if thisPP.parents_ids_dict:
                    (
                        kernel_params,
                        kernel_params_ll_events,
                        parents_diff_time_ll_events,
                        parents_real_events_times,
                        parents_real_events_mask,
                    ) = get_parents_params_mask(
                        self.PPs, thisPP, batch_real_events_times, batch_real_events_mask, real_virtual_events_times, self.device
                    )

                    if thisPP.processes_type == "general":
                        real_ll_events = torch.log(
                            thisPP.kernel.forward(
                                parents_diff_time_ll_events,
                                kernel_params_ll_events.unsqueeze(-2),
                            )
                            + F.softplus(thisPP.background_rate)
                        )
                    else:
                        real_ll_events = torch.log(
                            torch.sum(
                                thisPP.kernel.forward(
                                    parents_diff_time_ll_events,
                                    kernel_params_ll_events,
                                ),
                                dim=-1,
                            )
                            + F.softplus(thisPP.background_rate)
                        )
                else:
                    real_ll_events = torch.log(
                        batch_base_rate[PPId]
                    ).expand_as(virtual_ll_events)
                diff_real_virtual_ll_events = (
                    real_ll_events - virtual_ll_events
                )

                parents_ids_dict = thisPP.parents_ids_dict
                (
                    p_children_real_events_times_dict,
                    p_children_real_events_mask_dict,
                    p_children_real_events_embeddings_dict,
                    p_thisPP_shift_dict,
                    p_children_real_events_sort_indices,
                    p_children_stats_ids_dict,
                ) = self.get_p_children(
                    parents_ids_dict,
                    thisPP.id,
                    PPs_lst,
                    batch_real_events_times,
                    batch_real_events_mask,
                    real_virtual_events_times,
                    real_virtual_events_mask,
                )

                children_ids_dict = thisPP.children_ids_dict
                thisPPid = thisPP.id
                (
                    c_parents_real_events_times_dict,
                    c_parents_real_events_mask_dict,
                    c_parents_real_events_embeddings_dict,
                    c_thisPP_shift_dict,
                    c_parents_real_events_sort_indices,
                    c_parents_stats_ids_dict,
                ) = self.get_c_parents(
                    children_ids_dict,
                    thisPPid,
                    PPs_lst,
                    batch_real_events_times,
                    batch_real_events_mask,
                    real_virtual_events_times,
                    real_virtual_events_mask,
                )

                for i in range(2):
                    for j in range(3):
                        (
                            real_virtual_events_mask_prop,
                            c_parents_real_events_mask_proposal_dict,
                            p_children_real_events_mask_proposal_dict,
                        ) = self.move(
                            thisPP,
                            PPs_lst,
                            real_virtual_events_times,
                            real_virtual_events_mask,
                            diff_real_virtual_ll_events,
                            batch_real_events_times,
                            batch_real_events_mask,
                            batch_virtual_events_times,
                            batch_virtual_events_mask,
                            batch_real_loglikelihood,
                            batch_virtual_loglikelihood,
                            p_children_real_events_times_dict,
                            p_children_real_events_mask_dict,
                            p_children_real_events_embeddings_dict,
                            p_thisPP_shift_dict,
                            p_children_real_events_sort_indices,
                            p_children_stats_ids_dict,
                            c_parents_real_events_times_dict,
                            c_parents_real_events_mask_dict,
                            c_parents_real_events_embeddings_dict,
                            c_thisPP_shift_dict,
                            c_parents_real_events_sort_indices,
                            c_parents_stats_ids_dict,
                            type="flip",
                            data_type=data_type,
                            batch_end_time=batch_end_time,
                        )
                        if real_virtual_events_mask_prop is not None:
                            real_virtual_events_mask = (
                                real_virtual_events_mask_prop
                            )
                            c_parents_real_events_mask_dict = (
                                c_parents_real_events_mask_proposal_dict
                            )
                            p_children_real_events_mask_dict = (
                                p_children_real_events_mask_proposal_dict
                            )
                    (
                        real_virtual_events_mask_prop,
                        c_parents_real_events_mask_proposal_dict,
                        p_children_real_events_mask_proposal_dict,
                    ) = self.move(
                        thisPP,
                        PPs_lst,
                        real_virtual_events_times,
                        real_virtual_events_mask,
                        diff_real_virtual_ll_events,
                        batch_real_events_times,
                        batch_real_events_mask,
                        batch_virtual_events_times,
                        batch_virtual_events_mask,
                        batch_real_loglikelihood,
                        batch_virtual_loglikelihood,
                        p_children_real_events_times_dict,
                        p_children_real_events_mask_dict,
                        p_children_real_events_embeddings_dict,
                        p_thisPP_shift_dict,
                        p_children_real_events_sort_indices,
                        p_children_stats_ids_dict,
                        c_parents_real_events_times_dict,
                        c_parents_real_events_mask_dict,
                        c_parents_real_events_embeddings_dict,
                        c_thisPP_shift_dict,
                        c_parents_real_events_sort_indices,
                        c_parents_stats_ids_dict,
                        type="swap",
                        data_type=data_type,
                        batch_end_time=batch_end_time,
                    )
                    if real_virtual_events_mask_prop is not None:
                        real_virtual_events_mask = (
                            real_virtual_events_mask_prop
                        )
                        c_parents_real_events_mask_dict = (
                            c_parents_real_events_mask_proposal_dict
                        )
                        p_children_real_events_mask_dict = (
                            p_children_real_events_mask_proposal_dict
                        )

            (
                real_mask_sorted,
                real_mask_sorted_indices,
            ) = real_virtual_events_mask.sort(dim=-1, descending=True)
            real_events_times = real_virtual_events_times.gather(
                dim=-1, index=real_mask_sorted_indices
            )
            num_virtual_to_del = torch.min(
                torch.sum(real_mask_sorted <= 0, dim=-1)
            )

            virtual_mask_sorted = real_mask_sorted.flip((-1))
            virtual_events_times = real_events_times.flip((-1))
            num_real_to_del = torch.min(
                torch.sum(virtual_mask_sorted >= 0, dim=-1)
            )

            # comment out this part of code if need to check
            if num_virtual_to_del != 0: 
                batch_real_events_times[PPId] =  real_events_times[..., :-num_virtual_to_del]
                batch_real_events_mask[PPId] = real_mask_sorted[..., :-num_virtual_to_del] == 1
            else:
                batch_real_events_times[PPId] =  real_events_times
                batch_real_events_mask[PPId] = real_mask_sorted == 1

            if num_real_to_del != 0:
                batch_virtual_events_times[PPId] = virtual_events_times[..., :-num_real_to_del]
                batch_virtual_events_mask[PPId] = virtual_mask_sorted[..., :-num_real_to_del] == -1
            else:
                batch_virtual_events_times[PPId] = virtual_events_times
                batch_virtual_events_mask[PPId] = virtual_mask_sorted == -1
            
            if not resample_only:
                batch_real_loglikelihood[PPId] = self.calc_thisPP_real_loglikelihood(
                    thisPP,
                    real_ll_events,
                    real_virtual_events_mask,
                    None if not thisPP.parents_ids_dict else parents_real_events_times,
                    None if not thisPP.parents_ids_dict else parents_real_events_mask, 
                    None if not thisPP.parents_ids_dict else kernel_params, 
                    batch_base_rate,
                    batch_end_time,
                )
                batch_virtual_loglikelihood[PPId] = self.calc_thisPP_virtual_loglikelihood(
                    thisPP,
                    virtual_ll_events,
                    real_virtual_events_mask,
                    children_real_events_times,
                    children_real_events_mask,
                    virtual_kernel_params,
                    batch_end_time,
                )

            # comment out this part of code
            if not last_scan or resample_only:
                continue

            for pp in self.PPs.PPs_lst:
                if data_type == "train":
                    pp.real_loglikelihood[
                        :, batch_ids, ...
                    ] = batch_real_loglikelihood[pp.id].cpu()
                    if pp.id in batch_virtual_loglikelihood:
                        pp.virtual_loglikelihood[
                            :, batch_ids, ...
                        ] = batch_virtual_loglikelihood[pp.id].cpu()
                elif data_type == "dev":
                    pp.dev_real_loglikelihood[
                        :, batch_ids, ...
                    ] = batch_real_loglikelihood[pp.id].cpu()
                    if pp.id in batch_virtual_loglikelihood:
                        pp.dev_virtual_loglikelihood[
                            :, batch_ids, ...
                        ] = batch_virtual_loglikelihood[pp.id].cpu()
                elif data_type == "test":
                    pp.test_real_loglikelihood[
                        :, batch_ids, ...
                    ] = batch_real_loglikelihood[pp.id].cpu()
                    if pp.id in batch_virtual_loglikelihood:
                        pp.test_virtual_loglikelihood[
                            :, batch_ids, ...
                        ] = batch_virtual_loglikelihood[pp.id].cpu()
            self.assign_batch_sample(
                thisPP,
                real_virtual_events_mask, 
                batch_ids, 
                None if not thisPP.parents_ids_dict else parents_real_events_times,
                None if not thisPP.parents_ids_dict else parents_real_events_mask,
                None if not thisPP.parents_ids_dict else kernel_params,
                None if resample_only else real_ll_events,
                children_real_events_times,
                children_real_events_mask,
                virtual_kernel_params,
                None if resample_only else virtual_ll_events,
                resample_only,
                real_events_times,
                real_mask_sorted,
                num_virtual_to_del,
                virtual_events_times,
                virtual_mask_sorted,
                num_real_to_del,
                data_type,
            )


    def calc_thisPP_virtual_loglikelihood(
        self,
        thisPP,
        virtual_ll_events,
        real_virtual_events_mask,
        children_real_events_times,
        children_real_events_mask,
        virtual_kernel_params,
        batch_end_time,
    ):
        numel_children = torch.numel(children_real_events_times)
        if thisPP.virtual_processes_type == "general":
            # diff_time_ll_events  # [sample_size, batch_size, virtual_seq_len]
            if numel_children != 0:
                times_for_int = children_real_events_times - F.pad(
                    children_real_events_times[..., :-1],
                    pad=(1, 0),
                    mode="constant",
                    value=0.0,
                )
                ll_time = thisPP.virtual_kernel.integral(
                    times_for_int, virtual_kernel_params
                )
            else:
                ll_time = 0
        else:
            if numel_children != 0:
                ll_time = thisPP.virtual_kernel.integral(
                    children_real_events_times, virtual_kernel_params
                )
            else:
                ll_time = 0
        virtual_loglikelihood = (
            (virtual_ll_events * (real_virtual_events_mask == -1)).sum(dim=-1)
            - (ll_time * children_real_events_mask).sum(dim=-1)
            - batch_end_time.squeeze(-1)
            * (
                F.softplus(thisPP.virtual_background_rate)
                + thisPP.virtual_prop_background_rate
            )
        )
        return virtual_loglikelihood

    def calc_thisPP_real_loglikelihood(
        self, 
        thisPP, 
        real_ll_events, 
        real_virtual_events_mask, 
        parents_real_events_times, 
        parents_real_events_mask, 
        kernel_params, 
        batch_base_rate,
        batch_end_time,
    ):
        if thisPP.parents_ids_dict:
            if thisPP.processes_type == "general":
                times_for_int = (
                    F.pad(
                        parents_real_events_times[..., :-1],
                        pad=(0, 1),
                        mode="constant",
                        value=batch_end_time,
                    )
                    - parents_real_events_times
                )
                ll_time = thisPP.kernel.integral(
                    times_for_int, kernel_params
                )
            else:
                ll_time = thisPP.kernel.integral(
                    (batch_end_time - parents_real_events_times)
                    * parents_real_events_mask,
                    kernel_params,
                )
            real_loglikelihood = (
                (real_ll_events * (real_virtual_events_mask == 1)).sum(
                    dim=-1
                )
                - ll_time.sum(dim=-1)
                - (F.softplus(thisPP.background_rate) * batch_end_time.squeeze(-1))
            )
        else:
            base_rate = batch_base_rate[thisPP.id]
            ll_events = torch.log(base_rate).expand_as(real_virtual_events_mask.unsqueeze(0)).clone()
            ll_events[real_virtual_events_mask.unsqueeze(0) != 1] = 0
            ll_time = base_rate * batch_end_time
            real_loglikelihood = (ll_events * (real_virtual_events_mask == 1)).sum(dim=-1) - ll_time.squeeze(-1)
        return real_loglikelihood

    def assign_batch_sample(
        self,
        thisPP, 
        real_virtual_events_mask, 
        batch_ids, 
        parents_real_events_times,
        parents_real_events_mask,
        kernel_params,
        real_ll_events,
        children_real_events_times,
        children_real_events_mask,
        virtual_kernel_params,
        virtual_ll_events,
        resample_only,
        real_events_times,
        real_mask_sorted,
        num_virtual_to_del,
        virtual_events_times,
        virtual_mask_sorted,
        num_real_to_del,
        data_type,
    ):
        if data_type == "train":
            thisPP_real_seq_len = thisPP.real_events_times.shape[-1]
            thisPP_virtual_seq_len = thisPP.virtual_events_times.shape[-1]
            thisPP_whole_batch_size = thisPP.real_events_times.shape[1]
        elif data_type == "dev":
            thisPP_real_seq_len = thisPP.dev_real_events_times.shape[-1]
            thisPP_virtual_seq_len = thisPP.dev_virtual_events_times.shape[-1]
            thisPP_whole_batch_size = thisPP.dev_real_events_times.shape[1]
        elif data_type == "test":
            thisPP_real_seq_len = thisPP.test_real_events_times.shape[-1]
            thisPP_virtual_seq_len = thisPP.test_virtual_events_times.shape[-1]
            thisPP_whole_batch_size = thisPP.test_real_events_times.shape[1]
        if thisPP_whole_batch_size > 1:
            if thisPP_real_seq_len >= (
                real_events_times.shape[-1] - num_virtual_to_del
            ):
                min_seq_len = thisPP_real_seq_len
                diff_shape = min_seq_len - real_events_times.shape[-1]
                if diff_shape > 0:
                    real_events_times = torch.cat(
                        (
                            real_events_times,
                            torch.ones(1, len(batch_ids), diff_shape, device=self.device) * PAD,
                        ),
                        dim=-1,
                    )
                    real_events_mask = torch.cat(
                        (
                            real_mask_sorted == 1,
                            torch.zeros(1, len(batch_ids), diff_shape, device=self.device).bool(),
                        ),
                        dim=-1,
                    )
                else:
                    if diff_shape != 0:
                        real_events_times = real_events_times[..., :diff_shape]
                        real_events_mask = real_mask_sorted[..., :diff_shape] == 1
                    else:
                        real_events_mask = real_mask_sorted == 1
            else:
                min_seq_len = (
                    real_events_times.shape[-1] - num_virtual_to_del
                )
                diff_shape = min_seq_len - thisPP_real_seq_len

                if data_type == "train":
                    thisPP.real_events_times = torch.cat(
                        (
                            thisPP.real_events_times,
                            torch.ones(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            )
                            * PAD,
                        ),
                        dim=-1,
                    )
                    thisPP.real_events_mask = torch.cat(
                        (
                            thisPP.real_events_mask,
                            torch.zeros(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            ).bool(),
                        ),
                        dim=-1,
                    )
                elif data_type == "dev":
                    thisPP.dev_real_events_times = torch.cat(
                        (
                            thisPP.dev_real_events_times,
                            torch.ones(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            )
                            * PAD,
                        ),
                        dim=-1,
                    )
                    thisPP.dev_real_events_mask = torch.cat(
                        (
                            thisPP.dev_real_events_mask,
                            torch.zeros(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            ).bool(),
                        ),
                        dim=-1,
                    )
                elif data_type == "test":
                    thisPP.test_real_events_times = torch.cat(
                        (
                            thisPP.test_real_events_times,
                            torch.ones(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None if not self.predict else self.device,
                            )
                            * PAD,
                        ),
                        dim=-1,
                    )
                    thisPP.test_real_events_mask = torch.cat(
                        (
                            thisPP.test_real_events_mask,
                            torch.zeros(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None if not self.predict else self.device,
                            ).bool(),
                        ),
                        dim=-1,
                    )
                if num_virtual_to_del != 0:
                    real_events_times = real_events_times[..., :-num_virtual_to_del]
                    real_events_mask = real_mask_sorted[..., :-num_virtual_to_del] == 1
                else:
                    real_events_mask = real_mask_sorted == 1
            if data_type == "train":
                thisPP.real_events_times[
                    :, batch_ids, ...
                ] = real_events_times.cpu()
                thisPP.real_events_mask[:, batch_ids, ...] = real_events_mask.cpu()
            elif data_type == "dev":
                thisPP.dev_real_events_times[
                    :, batch_ids, ...
                ] = real_events_times.cpu()
                thisPP.dev_real_events_mask[:, batch_ids, ...] = real_events_mask.cpu()
            elif data_type == "test":
                if not self.predict:
                    thisPP.test_real_events_times[
                        :, batch_ids, ...
                    ] = real_events_times.cpu()
                    thisPP.test_real_events_mask[:, batch_ids, ...] = real_events_mask.cpu()
                else:
                    thisPP.test_real_events_times[
                        :, batch_ids, ...
                    ] = real_events_times
                    thisPP.test_real_events_mask[:, batch_ids, ...] = real_events_mask
        else:
            if num_virtual_to_del != 0:
                real_events_times = real_events_times[
                    ..., :-num_virtual_to_del
                ]
                real_events_mask = real_mask_sorted[..., :-num_virtual_to_del] == 1
            else:
                real_events_mask = real_mask_sorted == 1
            if data_type == "train":
                thisPP.real_events_times = real_events_times.cpu()
                thisPP.real_events_mask = real_events_mask.cpu()
            elif data_type == "dev":
                thisPP.dev_real_events_times = real_events_times.cpu()
                thisPP.dev_real_events_mask = real_events_mask.cpu()
            elif data_type == "test":
                if not self.predict:
                    thisPP.test_real_events_times = real_events_times.cpu()
                    thisPP.test_real_events_mask = real_events_mask.cpu()
                else:
                    thisPP.test_real_events_times = real_events_times
                    thisPP.test_real_events_mask = real_events_mask

        if thisPP_whole_batch_size > 1:
            if thisPP_virtual_seq_len >= (
                virtual_events_times.shape[-1] - num_real_to_del
            ):
                min_seq_len = thisPP_virtual_seq_len
                diff_shape = min_seq_len - virtual_events_times.shape[-1]
                if diff_shape > 0:
                    virtual_events_times = torch.cat(
                        (
                            virtual_events_times,
                            torch.ones(1, len(batch_ids), diff_shape, device=self.device) * PAD,
                        ),
                        dim=-1,
                    )
                    virtual_events_mask = torch.cat(
                        (
                            virtual_mask_sorted == -1,
                            torch.zeros(1, len(batch_ids), diff_shape, device=self.device).bool(),
                        ),
                        dim=-1,
                    )
                else:
                    if diff_shape != 0:
                        virtual_events_times = virtual_events_times[..., :diff_shape]
                        virtual_events_mask = virtual_mask_sorted[..., :diff_shape] == -1
                    else:
                        virtual_events_mask = virtual_mask_sorted == -1
            else:
                min_seq_len = (
                    virtual_events_times.shape[-1] - num_real_to_del
                )
                diff_shape = (
                    min_seq_len - thisPP_virtual_seq_len
                )

                if data_type == "train":
                    thisPP.virtual_events_times = torch.cat(
                        (
                            thisPP.virtual_events_times,
                            torch.ones(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            )
                            * PAD,
                        ),
                        dim=-1,
                    )
                    thisPP.virtual_events_mask = torch.cat(
                        (
                            thisPP.virtual_events_mask,
                            torch.zeros(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            ).bool(),
                        ),
                        dim=-1,
                    )
                elif data_type == "dev":
                    thisPP.dev_virtual_events_times = torch.cat(
                        (
                            thisPP.dev_virtual_events_times,
                            torch.ones(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            )
                            * PAD,
                        ),
                        dim=-1,
                    )
                    thisPP.dev_virtual_events_mask = torch.cat(
                        (
                            thisPP.dev_virtual_events_mask,
                            torch.zeros(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None,
                            ).bool(),
                        ),
                        dim=-1,
                    )
                elif data_type == "test":
                    thisPP.test_virtual_events_times = torch.cat(
                        (
                            thisPP.test_virtual_events_times,
                            torch.ones(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None if not self.predict else self.device,
                            )
                            * PAD,
                        ),
                        dim=-1,
                    )
                    thisPP.test_virtual_events_mask = torch.cat(
                        (
                            thisPP.test_virtual_events_mask,
                            torch.zeros(
                                1,
                                thisPP_whole_batch_size,
                                diff_shape,
                                device=None if not self.predict else self.device,
                            ).bool(),
                        ),
                        dim=-1,
                    )
                if num_real_to_del != 0:
                    virtual_events_times = virtual_events_times[..., :-num_real_to_del]
                    virtual_events_mask = virtual_mask_sorted[..., :-num_real_to_del] == -1
                else:
                    virtual_events_mask = virtual_mask_sorted == -1
            if data_type == "train":
                thisPP.virtual_events_times[
                    :, batch_ids, ...
                ] = virtual_events_times.cpu()
                thisPP.virtual_events_mask[
                    :, batch_ids, ...
                ] = virtual_events_mask.cpu()
            elif data_type == "dev":
                thisPP.dev_virtual_events_times[
                    :, batch_ids, ...
                ] = virtual_events_times.cpu()
                thisPP.dev_virtual_events_mask[
                    :, batch_ids, ...
                ] = virtual_events_mask.cpu()
            elif data_type == "test":
                if not self.predict:
                    thisPP.test_virtual_events_times[
                        :, batch_ids, ...
                    ] = virtual_events_times.cpu()
                    thisPP.test_virtual_events_mask[
                        :, batch_ids, ...
                    ] = virtual_events_mask.cpu()
                else:
                    thisPP.test_virtual_events_times[
                        :, batch_ids, ...
                    ] = virtual_events_times
                    thisPP.test_virtual_events_mask[
                        :, batch_ids, ...
                    ] = virtual_events_mask
        else:
            if num_real_to_del != 0:
                virtual_events_times = virtual_events_times[
                    ..., :-num_real_to_del
                ]
                virtual_events_mask = virtual_mask_sorted[
                    ..., :-num_real_to_del
                ] == -1
            else:
                virtual_events_mask = virtual_mask_sorted == -1
            if data_type == "train":
                thisPP.virtual_events_times = virtual_events_times.cpu()
                thisPP.virtual_events_mask = virtual_events_mask.cpu()
            elif data_type == "dev":
                thisPP.dev_virtual_events_times = virtual_events_times.cpu()
                thisPP.dev_virtual_events_mask = virtual_events_mask.cpu()
            elif data_type == "test":
                if not self.predict:
                    thisPP.test_virtual_events_times = virtual_events_times.cpu()
                    thisPP.test_virtual_events_mask = virtual_events_mask.cpu()
                else:
                    thisPP.test_virtual_events_times = virtual_events_times
                    thisPP.test_virtual_events_mask = virtual_events_mask


    def move(
        self,
        thisPP,
        PPs_lst,
        real_virtual_events_times,
        real_virtual_events_mask,
        diff_real_virtual_ll_events,
        batch_real_events_times,
        batch_real_events_mask,
        batch_virtual_events_times,
        batch_virtual_events_mask,
        batch_real_loglikelihood,
        batch_virtual_loglikelihood,
        p_children_real_events_times_dict,
        p_children_real_events_mask_dict,
        p_children_real_events_embeddings_dict,
        p_thisPP_shift_dict,
        p_children_real_events_sort_indices,
        p_children_stats_ids_dict,
        c_parents_real_events_times_dict,
        c_parents_real_events_mask_dict,
        c_parents_real_events_embeddings_dict,
        c_thisPP_shift_dict,
        c_parents_real_events_sort_indices,
        c_parents_stats_ids_dict,
        type,
        data_type,
        batch_end_time,
    ):
        if type == "flip":
            _real_virtual_events_mask = torch.logical_or(
                real_virtual_events_mask == -1, real_virtual_events_mask == 1
            )
            num_to_flip = torch.sum(
                _real_virtual_events_mask, -1, keepdim=True
            )
            movable_batch_ids_bool_mask = num_to_flip != 0  # [batch_size, 1]
            if torch.sum(movable_batch_ids_bool_mask) == 0:
                return (
                    real_virtual_events_mask,
                    c_parents_real_events_mask_dict,
                    p_children_real_events_mask_dict,
                )
            flip_id = torch.floor(
                torch.rand(num_to_flip.shape, device=self.device) * num_to_flip
            ).long()
        else:
            num_of_real = torch.sum(
                real_virtual_events_mask == 1, -1, keepdim=True
            )
            num_of_virtual = torch.sum(
                real_virtual_events_mask == -1, -1, keepdim=True
            )
            movable_batch_ids_bool_mask = torch.logical_and(
                num_of_real != 0, num_of_virtual != 0
            )
            if torch.sum(movable_batch_ids_bool_mask) == 0:
                return (
                    real_virtual_events_mask,
                    c_parents_real_events_mask_dict,
                    p_children_real_events_mask_dict,
                )

            mat_id = torch.arange(real_virtual_events_mask.shape[-1], device=self.device)[
                None, :
            ].expand_as(real_virtual_events_mask)
            real_events_cum_sum = torch.cumsum(
                real_virtual_events_mask > 0, dim=-1
            )
            real_shift_id = mat_id - real_events_cum_sum + 1
            real_shift_id[real_virtual_events_mask <= 0] = 1e10
            real_shift_id, _ = torch.sort(real_shift_id, dim=-1)

            virtual_events_cum_sum = torch.cumsum(
                real_virtual_events_mask < 0, dim=-1
            )
            virtual_shift_id = mat_id - virtual_events_cum_sum + 1
            virtual_shift_id[real_virtual_events_mask >= 0] = 1e10
            virtual_shift_id, _ = torch.sort(virtual_shift_id, dim=-1)

            flip_real_id = torch.floor(
                torch.rand(num_of_real.shape, device=self.device) * num_of_real
            ).long()
            flip_real_id += torch.gather(
                real_shift_id, dim=-1, index=flip_real_id
            )
            flip_virtual_id = torch.floor(
                torch.rand(num_of_virtual.shape, device=self.device) * num_of_virtual
            ).long()
            flip_virtual_id += torch.gather(
                virtual_shift_id, dim=-1, index=flip_virtual_id
            )
            flip_id = torch.cat((flip_real_id, flip_virtual_id), dim=-1).long()
        flip_id = flip_id * movable_batch_ids_bool_mask
        real_virtual_events_mask_proposal = real_virtual_events_mask.float().scatter(
            -1, flip_id, -1, reduce="multiply"
        )
        real_virtual_events_mask_proposal = real_virtual_events_mask_proposal.long()


        diff_proposal_origin_mask = (
            real_virtual_events_mask_proposal - real_virtual_events_mask
        )

        log_likelihood_ratio_first = torch.gather(
            (
                diff_proposal_origin_mask
                * diff_real_virtual_ll_events
            )
            / 2,
            index=flip_id,
            dim=-1,
        )
        log_likelihood_ratio_first = torch.sum(
            log_likelihood_ratio_first, dim=-1
        )
        parents_ids_dict = thisPP.parents_ids_dict
        if parents_ids_dict:
            parentsPPs = [PPs_lst[p_id] for p_id in parents_ids_dict]
            original_parents_ll = torch.stack(
                [batch_virtual_loglikelihood[p_id] for p_id in parents_ids_dict]
            )
            original_parents_ll = torch.sum(original_parents_ll, dim=0)
            proposal_parents_ll = []
            p_children_real_events_mask_proposal_dict = {}
            for pp in parentsPPs:
                children_ids_string = pp.children_ids_string
                p_children_real_events_times_calc_ll = p_children_real_events_times_dict[
                    children_ids_string
                ]
                p_children_real_events_mask = p_children_real_events_mask_dict[
                    children_ids_string
                ]

                p_children_real_events_embeddings = p_children_real_events_embeddings_dict[children_ids_string]
                p_thisPP_shift = p_thisPP_shift_dict[children_ids_string]

                if flip_id.shape[-1] == 1:
                    p_children_real_events_mask_proposal = (
                        p_children_real_events_mask
                        * torch.where(
                            p_children_real_events_sort_indices[
                                children_ids_string
                            ]
                            == (flip_id + p_thisPP_shift),
                            -1,
                            1,
                        )
                    )
                else:
                    this_p_children_real_events_sort_indices = p_children_real_events_sort_indices[
                        children_ids_string
                    ]
                    p_children_real_events_mask_proposal = (
                        p_children_real_events_mask
                        * torch.where(
                            torch.logical_or(
                                this_p_children_real_events_sort_indices
                                == (flip_id[..., [0]] + p_thisPP_shift),
                                this_p_children_real_events_sort_indices
                                == (flip_id[..., [1]] + p_thisPP_shift),
                            ),
                            -1,
                            1,
                        )
                    )
                p_children_real_events_mask_proposal_dict[
                    children_ids_string
                ] = p_children_real_events_mask_proposal

                if thisPP.synthetic_end:
                    p_children_real_events_mask_proposal_calc_ll = F.pad(
                        p_children_real_events_mask_proposal,
                        pad=(0, 1),
                        mode="constant",
                        value=True,
                    )
                else:
                    p_children_real_events_mask_proposal_calc_ll = (
                        p_children_real_events_mask_proposal
                    )
                if pp.virtual_processes_type == "general":
                    p_children_params = pp.calc_params(
                        neighbor_real_events_embeddings=p_children_real_events_embeddings,
                        neighbor_mask=p_children_real_events_mask_proposal.unsqueeze(-1),
                        virtual=True,
                    )
                    p_children_real_events_times_calc_ll = p_children_real_events_times_calc_ll.masked_fill(p_children_real_events_mask_proposal_calc_ll != 1, 1e20)
                    p_children_real_events_times_calc_ll, indices = torch.sort(
                        p_children_real_events_times_calc_ll
                    )
                    p_children_real_events_mask_proposal_calc_ll = torch.gather(
                        p_children_real_events_mask_proposal_calc_ll, dim=-1, index=indices,
                    )
                    p_children_params = torch.gather(p_children_params, dim=-2, index=indices.unsqueeze(-1).expand_as(p_children_params))
                    children_real_events_stats_ids = None
                else:
                    children_shape = p_children_real_events_times_calc_ll.shape
                    children_real_events_stats_ids = p_children_stats_ids_dict[children_ids_string]
                    p_children_params = pp.virtual_kernel_params.expand(children_shape[0], children_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
                    if p_children_params.shape[-2] > 1:
                        p_children_params = p_children_params.gather(dim=-2, index=children_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, p_children_params.shape[-1]))

                proposal_parents_ll.append(
                    pp.virtual_ll(
                        batch_virtual_events_times[pp.id],
                        batch_virtual_events_mask[pp.id],
                        p_children_real_events_times_calc_ll,
                        p_children_params,
                        p_children_real_events_mask_proposal_calc_ll > 0,
                        batch_end_time=batch_end_time,
                        prop_virtual_background=True,
                        children_stats_ids=children_real_events_stats_ids,
                    )
                )
            proposal_parents_ll = torch.stack(proposal_parents_ll)
            proposal_parents_ll_sum = torch.sum(proposal_parents_ll, dim=0)
        else:
            original_parents_ll = 0
            proposal_parents_ll_sum = 0

        children_ids_dict = thisPP.children_ids_dict
        childrenPPs = [PPs_lst[c_id] for c_id in children_ids_dict]
        original_children_ll = torch.stack(
            [batch_real_loglikelihood[c_id] for c_id in children_ids_dict]
        )
        original_children_ll = torch.sum(original_children_ll, dim=0)
        proposal_children_ll = []
        c_parents_real_events_mask_proposal_dict = {}
        for pp in childrenPPs:
            parents_ids_string = pp.parents_ids_string
            c_parents_real_events_times = c_parents_real_events_times_dict[
                parents_ids_string
            ]
            c_parents_real_events_mask = c_parents_real_events_mask_dict[
                parents_ids_string
            ]
            c_parents_real_events_embeddings = c_parents_real_events_embeddings_dict[parents_ids_string]
            c_thisPP_shift = c_thisPP_shift_dict[parents_ids_string]
            if flip_id.shape[-1] == 1:
                c_parents_real_events_mask_proposal = (
                    c_parents_real_events_mask
                    * torch.where(
                        c_parents_real_events_sort_indices[parents_ids_string]
                        == (flip_id + c_thisPP_shift),
                        -1,
                        1,
                    )
                )
            else:
                this_c_parents_real_events_sort_indices = c_parents_real_events_sort_indices[
                    parents_ids_string
                ]
                c_parents_real_events_mask_proposal = (
                    c_parents_real_events_mask
                    * torch.where(
                        torch.logical_or(
                            this_c_parents_real_events_sort_indices
                            == (flip_id[..., [0]] + c_thisPP_shift),
                            this_c_parents_real_events_sort_indices
                            == (flip_id[..., [1]] + c_thisPP_shift),
                        ),
                        -1,
                        1,
                    )
                )
            c_parents_real_events_mask_proposal_dict[
                parents_ids_string
            ] = c_parents_real_events_mask_proposal
            if pp.processes_type == "general":
                c_parents_params = pp.calc_params(
                    neighbor_real_events_embeddings=c_parents_real_events_embeddings,
                    neighbor_mask=c_parents_real_events_mask_proposal.unsqueeze(-1),
                    virtual=False,
                )
                c_parents_real_events_times = c_parents_real_events_times.masked_fill(c_parents_real_events_mask_proposal != 1, 1e20)
                c_parents_real_events_times, indices = torch.sort(
                    c_parents_real_events_times
                )
                c_parents_real_events_mask_proposal = torch.gather(
                    c_parents_real_events_mask_proposal, dim=-1, index=indices,
                )
                c_parents_params = torch.gather(c_parents_params, dim=-2, index=indices.unsqueeze(-1).expand_as(c_parents_params))
                parents_real_events_stats_ids = None
            else:
                parents_shape = c_parents_real_events_times.shape
                parents_real_events_stats_ids = c_parents_stats_ids_dict[parents_ids_string]
                c_parents_params = pp.kernel_params.expand(parents_shape[0], parents_shape[1], -1, -1)
                if c_parents_params.shape[-2] > 1:
                    c_parents_params = c_parents_params.gather(dim=-2, index=parents_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, c_parents_params.shape[-1]))

            proposal_children_ll.append(
                pp.real_ll(
                    batch_real_events_times[pp.id],
                    batch_real_events_mask[pp.id],
                    c_parents_real_events_times,
                    c_parents_params,
                    c_parents_real_events_mask_proposal > 0,
                    batch_end_time=batch_end_time,
                    data_type=data_type,
                    parents_stats_ids=parents_real_events_stats_ids,
                )
            )
        proposal_children_ll = torch.stack(
            proposal_children_ll
        )  # [n_pp, n_sample, batch_size]
        proposal_children_ll_sum = torch.sum(proposal_children_ll, dim=0)

        ll_diff = (
            (
                proposal_children_ll_sum
                - original_children_ll
                + proposal_parents_ll_sum
                - original_parents_ll
                + log_likelihood_ratio_first
            )
            .unsqueeze(-1)
        )  # [n_sample, batch_size] -> [n_sample, batch_size, 1]

        p = torch.log(torch.rand(ll_diff.shape, device=self.device))
        accept_bool_mask = torch.logical_and(
            p < ll_diff, movable_batch_ids_bool_mask
        )
        accept_bool_mask_expand = accept_bool_mask
        accept_bool_mask = accept_bool_mask.squeeze(-1)
        real_virtual_events_mask_accept = torch.where(
            accept_bool_mask_expand.expand_as(real_virtual_events_mask),
            real_virtual_events_mask_proposal,
            real_virtual_events_mask,
        )

        p_children_real_events_mask_dict_accept = {}
        if parents_ids_dict:
            for count, pp in enumerate(parentsPPs):
                children_ids_string = pp.children_ids_string
                p_children_real_events_mask_dict_accept[
                    children_ids_string
                ] = torch.where(
                    accept_bool_mask_expand.expand_as(p_children_real_events_mask_dict[children_ids_string]),
                    p_children_real_events_mask_proposal_dict[
                        children_ids_string
                    ],
                    p_children_real_events_mask_dict[children_ids_string],
                )
                batch_virtual_loglikelihood[pp.id] = torch.where(
                    accept_bool_mask,
                    proposal_parents_ll[count],
                    batch_virtual_loglikelihood[pp.id],
                )

        c_parents_real_events_mask_dict_accept = {}
        for count, pp in enumerate(childrenPPs):
            parents_ids_string = pp.parents_ids_string
            c_parents_real_events_mask_dict_accept[
                parents_ids_string
            ] = torch.where(
                accept_bool_mask_expand.expand_as(c_parents_real_events_mask_dict[parents_ids_string]),
                c_parents_real_events_mask_proposal_dict[parents_ids_string],
                c_parents_real_events_mask_dict[parents_ids_string],
            )

            batch_real_loglikelihood[pp.id] = torch.where(
                accept_bool_mask,
                proposal_children_ll[count, ...],
                batch_real_loglikelihood[pp.id],
            )
            # print(pp.real_loglikelihood)
        return (
            real_virtual_events_mask_accept,
            c_parents_real_events_mask_dict_accept,
            p_children_real_events_mask_dict_accept,
        )

