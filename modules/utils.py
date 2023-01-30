import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import pickle
from modules.preprocess_dataset import get_dataloader

class EventsEmbedding(nn.Module):
    def __init__(self, d_model, numPPs) -> None:
        super().__init__()
        position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / d_model)
                for i in range(d_model)
            ],  # * -(math.log(10000.0) / d_model)
        )  # [d_model // 2, ]
        self.register_buffer('position_vec', position_vec)
        self.types_embedding = nn.Embedding(numPPs, d_model)  # [1, d_model]

    def forward(self, x, PPId):
        temporal_enc = x.unsqueeze(-1) / self.position_vec
        temporal_enc[..., 0::2] = torch.sin(temporal_enc[..., 0::2])
        temporal_enc[..., 1::2] = torch.cos(temporal_enc[..., 1::2])
        return temporal_enc + self.types_embedding(
            PPId.int()
        )  # [batch_size, seq_len, d_model]


def searchsorted(bin_locations, inputs):
    search_bin_locations = bin_locations.clone()
    search_bin_locations[..., -1] *= 1 + torch.finfo().eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def plot_mcmc(
    samples_events_times_dict,
    samples_events_mask_dict,
    PPs,
    plot_num_points,
    log_folder,
    fig=None,
    axs=None,
):
    n_hidden = len(samples_events_times_dict) - 1
    if fig is None:
        fig, axs = plt.subplots(n_hidden)
    time_mesh = torch.linspace(0, 1, plot_num_points) * PPs.end_time
    PPs_lst = PPs.PPs_lst
    for p_id in samples_events_times_dict:
        if not PPs_lst[p_id].bottom:
            samples = samples_events_times_dict[p_id]
            mask = samples_events_mask_dict[p_id]
            if n_hidden == 1:
                axs.set_title(str(p_id), fontsize=20)
                axs.set_xlabel("time", fontsize=15)
                axs.set_ylabel("intensity", fontsize=15)
            else:
                axs[n_hidden - p_id].set_title(str(p_id), fontsize=20)
                axs[n_hidden - p_id].set_xlabel("time", fontsize=15)
                axs[n_hidden - p_id].set_ylabel("intensity", fontsize=15)

            hist, bin_edges = torch.histogram(
                samples[mask],
                bins=800,
                range=(0, torch.max(time_mesh).detach().numpy()),
            )
            mcmc_x = (bin_edges[1:] + bin_edges[:-1]) / 2
            mcmc_y = hist / (bin_edges[1:] - bin_edges[:-1]) / samples.shape[0]

            if n_hidden == 1:
                axs.plot(mcmc_x[:], mcmc_y[:], alpha=0.2)
            else:
                axs[n_hidden - p_id].plot(mcmc_x[:], mcmc_y[:], alpha=0.2)
    fig.tight_layout()
    fig.savefig(log_folder + "hidden_test.pdf")


def get_children_events(
    thisPP,
    batch_real_events_times,
    batch_real_events_mask,
    pos_childId=None,
    real_virtual_events_times=None,
    real_virtual_events_mask=None,
    device=None,
):
    # get the events from the children
    children_ids_dict = thisPP.children_ids_dict

    if real_virtual_events_times is None:
        children_real_events_times = torch.cat(
            [batch_real_events_times[c_id] for c_id in children_ids_dict], dim=-1
        )
        children_real_events_mask = torch.cat(
            [batch_real_events_mask[c_id] for c_id in children_ids_dict], dim=-1
        )
        children_real_events_ids = torch.cat(
            [
                torch.tensor([c_id], device=device).expand_as(batch_real_events_times[c_id])
                for c_id in children_ids_dict
            ],
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
    else:
        children_real_events_times = torch.cat(
            [
                batch_real_events_times[c_id]
                if c_id != pos_childId
                else real_virtual_events_times
                for c_id in children_ids_dict
            ],
            dim=-1,
        )
        children_real_events_mask = torch.cat(
            [
                batch_real_events_mask[c_id]
                if c_id != pos_childId
                else real_virtual_events_mask
                for c_id in children_ids_dict
            ],
            dim=-1,
        )
        children_real_events_ids = torch.cat(
            [
                torch.tensor([c_id], device=device).expand_as(batch_real_events_times[c_id])
                if c_id != pos_childId
                else torch.tensor([c_id], device=device).expand_as(real_virtual_events_times)
                for c_id in children_ids_dict
            ],
            dim=-1,
        )
        indices = torch.arange(children_real_events_times.shape[-1], device=device)[None, None, ...]

    if pos_childId is not None:
        pre_pos_elements = torch.sum(
            torch.tensor(
                [
                    batch_real_events_times[c_id].shape[-1] if c_id < pos_childId else 0
                    for c_id in children_ids_dict
                ], device=device
            )
        )
    if pos_childId is not None:
        return (
            children_real_events_times,
            children_real_events_mask,
            children_real_events_ids,
            pre_pos_elements,
            indices,
        )
    else:
        return (
            children_real_events_times,
            children_real_events_mask,
            children_real_events_ids,
            indices,
        )

def get_parents_events(
    thisPP,
    batch_real_events_times,
    batch_real_events_mask,
    pos_parentId=None,
    real_virtual_events_times=None,
    real_virtual_events_mask=None,
    device=None,
):
    # get the events from the parents
    parents_ids_dict = thisPP.parents_ids_dict

    if pos_parentId is None:
        parents_real_events_times = torch.cat(
            [batch_real_events_times[p_id] for p_id in parents_ids_dict], dim=-1
        )
        parents_real_events_mask = torch.cat(
            [batch_real_events_mask[p_id] for p_id in parents_ids_dict], dim=-1
        )
        parents_real_events_ids = torch.cat(
            [
                torch.tensor([p_id], device=device).expand_as(batch_real_events_times[p_id])
                for p_id in parents_ids_dict
            ],
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
    else:
        parents_real_events_times = torch.cat(
            [
                batch_real_events_times[p_id]
                if p_id != pos_parentId
                else real_virtual_events_times
                for p_id in parents_ids_dict
            ],
            dim=-1,
        )
        parents_real_events_mask = torch.cat(
            [
                batch_real_events_mask[p_id]
                if p_id != pos_parentId
                else real_virtual_events_mask
                for p_id in parents_ids_dict
            ],
            dim=-1,
        )
        parents_real_events_ids = torch.cat(
            [
                torch.tensor([p_id], device=device).expand_as(batch_real_events_times[p_id])
                if p_id != pos_parentId
                else torch.tensor([p_id], device=device).expand_as(
                    real_virtual_events_times
                )
                for p_id in parents_ids_dict
            ],
            dim=-1,
        )
        indices = torch.arange(parents_real_events_times.shape[-1], device=device)[None, None, ...]

    if pos_parentId is not None:
        pre_pos_elements = torch.sum(
            torch.tensor(
                [
                    batch_real_events_times[p_id].shape[-1]
                    if p_id < pos_parentId
                    else 0
                    for p_id in parents_ids_dict
                ], device=device
            )
        )

    if pos_parentId is not None:
        return (
            parents_real_events_times,
            parents_real_events_mask,
            parents_real_events_ids,
            pre_pos_elements,
            indices,
        )
    else:
        return (
            parents_real_events_times,
            parents_real_events_mask,
            parents_real_events_ids,
            indices,
        )

def get_parents_params_mask(
    PPs, thisPP, batch_real_events_times, batch_real_events_mask, real_virtual_events_times, device
):
    processes_type = thisPP.processes_type
    (
        parents_real_events_times,
        parents_real_events_mask,
        parents_real_events_ids,
        _,
    ) = get_parents_events(thisPP, batch_real_events_times, batch_real_events_mask, device=device)
    parents_real_events_embeddings = (
        PPs.events_embedding(
            parents_real_events_times, parents_real_events_ids
        )
        if processes_type == "general"
        else None
    )
    # kernel params
    # virtual params
    if processes_type == "general":
        params = thisPP.calc_params(
            neighbor_real_events_embeddings=parents_real_events_embeddings,
            neighbor_mask=parents_real_events_mask,
            virtual=False,
        )
        parents_real_events_stats_ids = None
    else:
        parents_shape = parents_real_events_times.shape
        parents_real_events_stats_ids = parents_real_events_ids.clone()
        for key, val in thisPP.parents_ids_dict.items():
            parents_real_events_stats_ids = parents_real_events_stats_ids.masked_fill(parents_real_events_stats_ids == key, val)
        params = thisPP.kernel_params.expand(parents_shape[0], parents_shape[1], -1, -1) # [n_samples, n_batchs, n_kernel, n_params]
        if params.shape[-2] > 1:
            params = params.gather(dim=-2, index=parents_real_events_stats_ids.unsqueeze(-1).expand(-1, -1, -1, params.shape[-1]))
    if real_virtual_events_times is None:
        return (
            params,
            parents_real_events_times,
            parents_real_events_mask,
        )
    else:
        (
            parents_diff_time_ll_events,
            params_ll_events,
        ) = thisPP.prepare_params_and_diff_time(
            real_virtual_events_times=real_virtual_events_times,
            neighbor_events_times=parents_real_events_times,
            params=params,
            neighbor_real_events_mask=parents_real_events_mask,
            neighbor="parents",
            processes_type=processes_type,
            neighbor_stats_ids=parents_real_events_stats_ids,
        )
        return (
            params,
            params_ll_events,
            parents_diff_time_ll_events,
            parents_real_events_times,
            parents_real_events_mask,
        )

def assign_events_to_batch(
    batch_real_events_times,
    batch_real_events_mask,
    batch_virtual_events_times,
    batch_virtual_events_mask,
    batch_real_loglikelihood,
    batch_virtual_loglikelihood,
    batch_base_rate,
    PPs_lst,
    batch_ids,
    device=None,
    data_type="train",
):
    if data_type == "train":
        for pp in PPs_lst:
            batch_real_events_times[pp.id] = pp.real_events_times[
                :, batch_ids, ...
            ].to(device)
            batch_real_events_mask[pp.id] = pp.real_events_mask[
                :, batch_ids, ...
            ].to(device)
            if pp.virtual_events_times is not None:
                batch_virtual_events_times[
                    pp.id
                ] = pp.virtual_events_times[:, batch_ids, ...].to(
                    device
                )
                batch_virtual_events_mask[pp.id] = pp.virtual_events_mask[
                    :, batch_ids, ...
                ].to(device)
                batch_virtual_loglikelihood[
                    pp.id
                ] = pp.virtual_loglikelihood[:, batch_ids, ...].to(
                    device
                )
            batch_real_loglikelihood[pp.id] = pp.real_loglikelihood[
                :, batch_ids, ...
            ].to(device)
            if pp.top:
                batch_base_rate[pp.id] = pp.base_rate[:, batch_ids, :]
    elif data_type == "dev":
        for pp in PPs_lst:
            batch_real_events_times[pp.id] = pp.dev_real_events_times[
                :, batch_ids, ...
            ].to(device)
            batch_real_events_mask[pp.id] = pp.dev_real_events_mask[
                :, batch_ids, ...
            ].to(device)
            if pp.dev_virtual_events_times is not None:
                batch_virtual_events_times[
                    pp.id
                ] = pp.dev_virtual_events_times[:, batch_ids, ...].to(
                    device
                )
                batch_virtual_events_mask[pp.id] = pp.dev_virtual_events_mask[
                    :, batch_ids, ...
                ].to(device)
                batch_virtual_loglikelihood[
                    pp.id
                ] = pp.dev_virtual_loglikelihood[:, batch_ids, ...].to(
                    device
                )
            batch_real_loglikelihood[pp.id] = pp.dev_real_loglikelihood[
                :, batch_ids, ...
            ].to(device)
            if pp.top:
                batch_base_rate[pp.id] = pp.dev_base_rate[:, batch_ids, :]
    elif data_type == "test":
        for pp in PPs_lst:
            batch_real_events_times[pp.id] = pp.test_real_events_times[
                :, batch_ids, ...
            ].to(device)
            batch_real_events_mask[pp.id] = pp.test_real_events_mask[
                :, batch_ids, ...
            ].to(device)
            if pp.test_virtual_events_times is not None:
                batch_virtual_events_times[
                    pp.id
                ] = pp.test_virtual_events_times[:, batch_ids, ...].to(
                    device
                )
                batch_virtual_events_mask[pp.id] = pp.test_virtual_events_mask[
                    :, batch_ids, ...
                ].to(device)
                batch_virtual_loglikelihood[
                    pp.id
                ] = pp.test_virtual_loglikelihood[:, batch_ids, ...].to(
                    device
                )
            batch_real_loglikelihood[pp.id] = pp.test_real_loglikelihood[
                :, batch_ids, ...
            ].to(device)
            if pp.top:
                batch_base_rate[pp.id] = pp.test_base_rate[:, batch_ids, :]

def update_real_loglikelihood(
    PPs,
    batch_ids,
    batch_real_events_times,
    batch_real_events_mask,
    batch_real_loglikelihood,
    batch_end_time,
    data_type,
    device
):
    PPs_lst = PPs.PPs_lst
    for thisPP in PPs_lst:
        parents_ids_dict = thisPP.parents_ids_dict
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
        if parents_ids_dict:
            if thisPP.processes_type == "general":
                parents_real_events_embeddings = PPs.events_embedding(
                    parents_real_events_times, parents_real_events_ids
                )
                parents_real_events_params = thisPP.calc_params(
                    neighbor_real_events_embeddings=parents_real_events_embeddings,
                    neighbor_mask=parents_real_events_mask,
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
        else:
            parents_real_events_stats_ids = None
        if data_type == "train":
            batch_real_loglikelihood[thisPP.id] = thisPP.real_ll(
                batch_real_events_times[thisPP.id],
                batch_real_events_mask[thisPP.id],
                parents_real_events_times
                if parents_ids_dict is not None
                else None,
                parents_real_events_params
                if parents_ids_dict is not None
                else None,
                parents_real_events_mask
                if parents_ids_dict is not None
                else None,
                batch_end_time=batch_end_time,
                batch_ids=batch_ids,
                data_type="train",
                parents_stats_ids=parents_real_events_stats_ids,
            )
            thisPP.real_loglikelihood[
                :, batch_ids, ...
            ] = batch_real_loglikelihood[thisPP.id].cpu()

        elif data_type == "dev":
            batch_real_loglikelihood[thisPP.id] = thisPP.real_ll(
                batch_real_events_times[thisPP.id],
                batch_real_events_mask[thisPP.id],
                parents_real_events_times
                if parents_ids_dict is not None
                else None,
                parents_real_events_params
                if parents_ids_dict is not None
                else None,
                parents_real_events_mask
                if parents_ids_dict is not None
                else None,
                batch_end_time=batch_end_time,
                batch_ids=batch_ids,
                data_type="dev",
                parents_stats_ids=parents_real_events_stats_ids,
            )
            thisPP.dev_real_loglikelihood[
                :, batch_ids, ...
            ] = batch_real_loglikelihood[thisPP.id].cpu()
        elif data_type == "test":
            batch_real_loglikelihood[thisPP.id] = thisPP.real_ll(
                batch_real_events_times[thisPP.id],
                batch_real_events_mask[thisPP.id],
                parents_real_events_times
                if parents_ids_dict is not None
                else None,
                parents_real_events_params
                if parents_ids_dict is not None
                else None,
                parents_real_events_mask
                if parents_ids_dict is not None
                else None,
                batch_end_time=batch_end_time,
                batch_ids=batch_ids,
                data_type="test",
                parents_stats_ids=parents_real_events_stats_ids,
            )
            thisPP.test_real_loglikelihood[
                :, batch_ids, ...
            ] = batch_real_loglikelihood[thisPP.id].cpu()

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, "rb") as f:
            data = pickle.load(f, encoding="latin-1")
            num_types = data["dim_process"]
            data = data[dict_name]
            return data, int(num_types)

    if opt.train_obs_size != 0:
        print("[Info] Loading train data...")
        train_data, num_types = load_data(opt.data + "train.pkl", "train")
    if opt.dev_obs_size != 0:
        print("[Info] Loading dev data...")
        dev_data, num_types = load_data(opt.data + "dev.pkl", "dev")
    if opt.test_obs_size != 0:
        print("[Info] Loading test data...")
        test_data, num_types = load_data(opt.data + "test.pkl", "test")

    if opt.train_obs_size != 0:
        trainloader = get_dataloader(
            train_data,
            opt.train_obs_size,
            opt.training_data_ratio,
            num_types,
            opt.time_scale,
            shuffle=False,
        )
    else:
        trainloader = None
    if opt.dev_obs_size != 0:
        devloader = get_dataloader(
            dev_data,
            opt.dev_obs_size,
            opt.dev_data_ratio,
            num_types,
            opt.time_scale,
            shuffle=False,
        )
    else:
        devloader = None
    if opt.test_obs_size != 0:
        testloader = get_dataloader(
            test_data, 
            opt.test_obs_size, 
            opt.test_data_ratio, 
            num_types, 
            opt.time_scale, 
            shuffle=False
        )
    else:
        testloader = None
    return trainloader, devloader, testloader, num_types

def process_events(PPs):
    '''remove the extra padding elements'''
    for pp in PPs.PPs_lst:
        if pp.real_events_times is not None and not pp.bottom:
            non_pad_mask = pp.real_events_mask
            min_zero_num = torch.min(torch.sum(~non_pad_mask, dim=-1))
            if min_zero_num > 0:
                non_pad_mask, indices = torch.sort(non_pad_mask, dim=-1, descending=True)
                events_times_samples = torch.gather(pp.real_events_times, dim=-1, index=indices)
                pp.real_events_times = events_times_samples[..., :-min_zero_num]
                pp.real_events_mask = non_pad_mask[..., :-min_zero_num]

        if pp.dev_real_events_times is not None and not pp.bottom:
            non_pad_mask = pp.dev_real_events_mask
            min_zero_num = torch.min(torch.sum(~non_pad_mask, dim=-1))
            if min_zero_num > 0:
                non_pad_mask, indices = torch.sort(non_pad_mask, dim=-1, descending=True)
                events_times_samples = torch.gather(pp.dev_real_events_times, dim=-1, index=indices)
                pp.dev_real_events_times = events_times_samples[..., :-min_zero_num]
                pp.dev_real_events_mask = non_pad_mask[..., :-min_zero_num]

        if pp.test_real_events_times is not None and not pp.bottom:
            non_pad_mask = pp.test_real_events_mask
            min_zero_num = torch.min(torch.sum(~non_pad_mask, dim=-1))
            if min_zero_num > 0:
                non_pad_mask, indices = torch.sort(non_pad_mask, dim=-1, descending=True)
                events_times_samples = torch.gather(pp.test_real_events_times, dim=-1, index=indices)
                pp.test_real_events_times = events_times_samples[..., :-min_zero_num]
                pp.test_real_events_mask = non_pad_mask[..., :-min_zero_num]

def get_variational_params(DPPs, nn_lr=1e-3):
    params = []
    for pp in DPPs.PPs_lst:
        if not pp.bottom:
            if pp.virtual_processes_type == "general":
                params.append({'params': pp.attn_encoder_virtualPP.parameters(), 'lr': nn_lr})
                params.append({'params': pp.encoding_to_virtual_kernel_nn.parameters(), 'lr': nn_lr})
            else:
                params.append({'params': pp.virtual_kernel_params})
            params.append({'params': pp.virtual_background_rate})
    params.append({'params': DPPs.events_embedding.parameters(), 'lr': nn_lr})
    return params

def get_model_params(DPPs, nn_lr=1e-3):
    params = []
    for pp in DPPs.PPs_lst:
        if not pp.top:
            if pp.processes_type == "general":
                params.append({'params': pp.attn_encoder_realPP.parameters(), 'lr': nn_lr})
                params.append({'params': pp.encoding_to_kernel_nn.parameters(), 'lr': nn_lr})
            else:
                params.append({'params': pp.kernel_params})
            params.append({'params': pp.background_rate})
    return params