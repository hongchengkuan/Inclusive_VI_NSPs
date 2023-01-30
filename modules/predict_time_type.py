import torch
import copy
import math
import time
import inference
from sampler import PriorSampler

def get_potential_next_events(evidences_to_predict, device):
    next_event_time_all_types = torch.cat(
        [
            evidences_to_predict[type][:, [0]]
            if torch.numel(evidences_to_predict[type]) != 0
            else torch.tensor([1e20], device=device).expand(evidences_to_predict[type].shape[0], 1)
            for type in range(len(evidences_to_predict))
        ],
        dim=-1,
    )  # [batch_size, num_types]
    next_event_times, next_event_types = torch.sort(
        next_event_time_all_types, dim=-1
    )
    next_event_times = next_event_times[:, [0]]  # [batch_size, 1]
    next_event_types = next_event_types[:, [0]]  # [batch_size, 1]

    for count in range(len(evidences_to_predict)):
        evidences_to_predict_remove_mask = next_event_types.expand_as(
            evidences_to_predict[count]
        )
        evidences_to_predict_remove_mask = (
            evidences_to_predict_remove_mask == count
        )
        evidences_to_predict_remove_mask[:, 1:] = False
        evidences_to_predict[count][evidences_to_predict_remove_mask] = 1e20
        evidences_to_predict[count] = evidences_to_predict[count].sort(dim=-1).values
        delete_num = torch.min(
            torch.sum(evidences_to_predict[count] == 1e20, dim=-1)
        )
        if delete_num != 0:
            evidences_to_predict[count] = evidences_to_predict[count][:, :-delete_num]

    return next_event_times, next_event_types


def update_known_evidences(
    known_evidences, next_event_times, next_event_types, prev_batch_end_time, device
):
    if prev_batch_end_time is None:
        end_time = torch.zeros(1, known_evidences[0].shape[0], 1, device=device)
    else:
        end_time = prev_batch_end_time
    end_time = torch.maximum(
        end_time,
        next_event_times.masked_fill(
            next_event_times == 1e20, -1
        ).unsqueeze(0),
    )
    # debug
    assert (end_time != 0).all()
    # if (next_event_times == 1e20).any():
    #     print()
    #
    for count in range(len(known_evidences)):
        to_add_event_times = next_event_times.masked_fill(
            next_event_types != count, 1e20
        )
        known_evidences[count] = torch.cat(
            (known_evidences[count], to_add_event_times), dim=-1
        )
        known_evidences[count] = known_evidences[count].sort(dim=-1).values
        delete_num = torch.min(
            torch.sum(known_evidences[count] == 1e20, dim=-1)
        )
        if delete_num != 0:
            known_evidences[count] = known_evidences[count][:, :-delete_num]

    return end_time


def predict_time_type(
    DPPs,
    pred_sample_size,
    test_evidences,
    batch_size,
    em_iters,
    virtual_only_to_predict,
    device,
    log_folder_name,
    top_base_rate,
):
    DPPs.eval()
    time_pred_square_err_sum = torch.zeros(pred_sample_size, device=device)
    type_pred_correct_num = torch.zeros(pred_sample_size, device=device)

    num_of_events = sum(
        [(test_evidences[i] < 1e20).sum() for i in range(len(test_evidences))]
    )
    num_of_events_to_predict = num_of_events - test_evidences[0].shape[0]

    tot_test_real_all_ll = 0
    tot_test_virtual_all_ll = 0
    tot_test_bottom_real_ll = 0
    true_pred_num = 0
    time0 = time.time()

    num_iters = math.ceil(test_evidences[0].shape[0] / batch_size)
    for n_iter in range(num_iters):
        last_iter = False
        if virtual_only_to_predict:
            last_iter = True
        batch_start = n_iter * batch_size
        batch_end = (n_iter + 1) * batch_size
        if batch_end > test_evidences[0].shape[0]:
            batch_end = test_evidences[0].shape[0]
        batch_ids = list(range(batch_start, batch_end))
        evidences_to_predict = copy.deepcopy(
            test_evidences
        )  # [num_types * tensor(batch_size, seq_len)]
        for type, data in enumerate(evidences_to_predict):
            evidences_to_predict[type] = torch.sort(
                data[batch_ids, :], dim=-1
            ).values.to(device)

        next_event_times, next_event_types = get_potential_next_events(
            evidences_to_predict, device
        )  # [batch_size, 1]

        known_evidences = [
            torch.tensor([], device=device).expand(len(batch_ids), -1)
            for _ in range(len(evidences_to_predict))
        ]
        batch_end_time = update_known_evidences(
            known_evidences, next_event_times, next_event_types, None, device
        )

        next_event_times, next_event_types = get_potential_next_events(
            evidences_to_predict, device
        )  # [batch_size, 1]

        for type, evidence in enumerate(known_evidences):
            DPPs.PPs_lst[type].test_real_events_times = evidence.unsqueeze(0)
            DPPs.PPs_lst[type].test_real_events_mask = evidence.unsqueeze(0) < 1e20
        DPPs.test_end_time = batch_end_time

        for top_pp_id in DPPs.top_ids_set:
            pp = DPPs.PPs_lst[top_pp_id]
            pp.test_base_rate = torch.ones(1, len(batch_ids), 1, device=device) * top_base_rate

        DPPs.to(device)
        if not virtual_only_to_predict:
            em_class = inference.EM(
                PPs=DPPs,
                events_embedding=None,
                initialize_batch_size=len(batch_ids),
                resample_only=False,
                record_virtual_samples=False,
                check_real_ll=False,
                device=device,
                predict=True,
            )

        remaining_count_to_predict = sum(
            [
                (evidences_to_predict[i] < 1e20).sum()
                for i in range(len(evidences_to_predict))
            ]
        ) + torch.sum(next_event_times < 1e20)
        em_iters_count = em_iters
        while remaining_count_to_predict >= 0:
            while True:
                if em_iters_count == 0 or virtual_only_to_predict:
                    pred_time_lst = [0 for _ in range(pred_sample_size)]
                    pred_type_lst = [0 for _ in range(pred_sample_size)]
                    if not virtual_only_to_predict:
                        em_iters_count = em_iters

                        tot_base_rate_dict = {}
                        for top_pp_id in em_class.PPs.top_ids_set:
                            tot_base_rate_dict[top_pp_id] = torch.zeros(
                                1, len(batch_ids), 1, device=device
                            )

                        for n_iter_sampling in range(pred_sample_size):
                            (
                                real_all_ll,
                                virtual_all_ll,
                                bottom_real_ll,
                                base_rate_dict,
                            ) = em_class.em_step(
                                batch_ids=list(range(len(batch_ids))),
                                burn_in_steps=0,
                                sample_size=1,
                                sample_intervals=1,
                                log_folder=None,
                                fig=None,
                                axs=None,
                                save_mcmc=False,
                                mcmc_load_path=None,
                                calc_elbo=False,
                                data_type="test",
                                first_after_opt=True
                                if n_iter_sampling == 0
                                else False,
                            )

                            if remaining_count_to_predict == 0:
                                tot_test_real_all_ll += (
                                    real_all_ll.item()
                                    * len(batch_ids)
                                    / pred_sample_size
                                )
                                tot_test_virtual_all_ll += (
                                    virtual_all_ll.item()
                                    * len(batch_ids)
                                    / pred_sample_size
                                )
                                tot_test_bottom_real_ll += (
                                    bottom_real_ll * len(batch_ids) / pred_sample_size
                                )
                                last_iter = True

                            first_event_dict = DPPs.sample_first_event(
                                batch_ids=list(range(len(batch_ids))), device=device
                            )
                            pred_time_lst[n_iter_sampling] = first_event_dict[
                                "time"
                            ]
                            pred_type_lst[n_iter_sampling] = first_event_dict[
                                "type"
                            ]

                            for top_pp_id in em_class.PPs.top_ids_set:
                                tot_base_rate_dict[top_pp_id] += (
                                    base_rate_dict[top_pp_id]
                                    / pred_sample_size
                                )

                        for top_pp_id in DPPs.top_ids_set:
                            pp = DPPs.PPs_lst[top_pp_id]
                            pp.test_base_rate[:, list(range(len(batch_ids))), :] = tot_base_rate_dict[
                                top_pp_id
                            ]

                    else:
                        prior_sampler = PriorSampler(
                            PPs=DPPs, events_embedding=DPPs.events_embedding
                        )

                        tot_base_rate_dict = {}
                        for top_pp_id in DPPs.top_ids_set:
                            tot_base_rate_dict[top_pp_id] = torch.zeros(
                                1, len(batch_ids), 1, device=device
                            )

                        for n_iter_sampling in range(pred_sample_size):
                            prior_sampler.sample(
                                batch_size=len(batch_ids),
                                plot=True,
                                return_samples=False,
                                predict=True,
                                device=device,
                            )
                            for top_pp_id in DPPs.top_ids_set:
                                tot_base_rate_dict[top_pp_id] += (
                                    torch.sum(
                                        DPPs.PPs_lst[
                                            top_pp_id
                                        ].test_real_events_mask,
                                        dim=-1,
                                        keepdim=True,
                                    )
                                    / DPPs.test_end_time
                                    / pred_sample_size
                                )

                        for top_pp_id in DPPs.top_ids_set:
                            pp = DPPs.PPs_lst[top_pp_id]
                            pp.test_base_rate[:, list(range(len(batch_ids))), :] = tot_base_rate_dict[
                                top_pp_id
                            ]

                        for n_iter_sampling in range(pred_sample_size):
                            prior_sampler.sample(
                                batch_size=len(batch_ids),
                                plot=True,
                                return_samples=False,
                                predict=True,
                                device=device,
                            )
                            first_event_dict = DPPs.sample_first_event(
                                batch_ids=list(range(len(batch_ids))), device=device
                            )
                            pred_time_lst[n_iter_sampling] = first_event_dict[
                                "time"
                            ]
                            pred_type_lst[n_iter_sampling] = first_event_dict[
                                "type"
                            ]



                    if remaining_count_to_predict != 0:
                        pred_time = torch.cumsum(
                            torch.cat(pred_time_lst, dim=-1), dim=-1
                        ) / (
                            torch.arange(pred_sample_size, device=device) + 1
                        )  # [batch_size, pred_sample_size]
                        pred_type_count = torch.zeros(
                            len(batch_ids), pred_sample_size, len(test_evidences), device=device
                        )  # [batch_size, pred_sample_size, num_types]
                        pred_type_lst_tensor = torch.cat(
                            pred_type_lst, dim=-1
                        )  # [batch_size, pred_sample_size]
                        for type in range(len(test_evidences)):
                            pred_type_count[..., [type]] = (
                                (pred_type_lst_tensor == type)
                                .cumsum(dim=-1)
                                .unsqueeze(-1).float()
                            )
                        pred_type = torch.max(
                            pred_type_count, dim=-1
                        ).indices  # [batch_size, pred_sample_size]

                        time_pred_square_err_sum += torch.sum(
                            (pred_time - next_event_times.masked_fill(next_event_times == 1e20, 100)) ** 2 * (next_event_times != 1e20), dim=0
                        )
                        type_pred_correct_num += torch.sum(
                            (pred_type == next_event_types) * (next_event_times != 1e20), dim=0
                        )

                        true_pred_num += torch.sum(next_event_times < 1e20)

                        (
                            next_event_times,
                            next_event_types,
                        ) = get_potential_next_events(evidences_to_predict, device)
                        batch_end_time = update_known_evidences(
                            known_evidences, next_event_times, next_event_types, prev_batch_end_time=batch_end_time, device=device
                        )

                        for type, evidence in enumerate(known_evidences):
                            DPPs.PPs_lst[type].test_real_events_times = evidence.unsqueeze(0)
                            DPPs.PPs_lst[type].test_real_events_mask = (
                                evidence < 1e20
                            ).unsqueeze(0)
                        DPPs.test_end_time = batch_end_time

                    remaining_count_to_predict = sum(
                        [
                            (evidences_to_predict[i] < 1e20).sum()
                            for i in range(len(evidences_to_predict))
                        ]
                    ) + torch.sum(next_event_times < 1e20)
                    # ----debug
                    # if remaining_count_to_predict == 3:
                    #     print()
                    # ----------------
                    print(
                        f"num of events to predict = {remaining_count_to_predict}",
                        flush=True,
                    )
                    print("whole time = ", time.time() - time0, flush=True)
                    break

                if not virtual_only_to_predict:
                    tot_real_all_ll = 0
                    tot_virtual_all_ll = 0
                    tot_bottom_real_ll = 0

                    tot_base_rate_dict = {}
                    for top_pp_id in em_class.PPs.top_ids_set:
                        tot_base_rate_dict[top_pp_id] = torch.zeros(
                            1, len(batch_ids), 1, device=device
                        )

                    for n_iter_sampling in range(pred_sample_size):
                        (
                            real_all_ll,
                            virtual_all_ll,
                            bottom_real_ll,
                            base_rate_dict,
                        ) = em_class.em_step(
                            batch_ids=list(range(len(batch_ids))),
                            burn_in_steps=0,
                            sample_size=1,
                            sample_intervals=1,
                            log_folder=None,
                            fig=None,
                            axs=None,
                            save_mcmc=False,
                            mcmc_load_path=None,
                            calc_elbo=False,
                            data_type="test",
                            first_after_opt=True
                            if n_iter_sampling == 0
                            else False,
                        )

                        tot_real_all_ll += (
                            real_all_ll.item() * len(batch_ids) / pred_sample_size
                        )
                        tot_virtual_all_ll += (
                            virtual_all_ll.item()
                            * len(batch_ids)
                            / pred_sample_size
                        )
                        tot_bottom_real_ll += (
                            bottom_real_ll * len(batch_ids) / pred_sample_size
                        )

                        for top_pp_id in em_class.PPs.top_ids_set:
                            tot_base_rate_dict[top_pp_id] += (
                                base_rate_dict[top_pp_id] / pred_sample_size
                            )

                    for top_pp_id in DPPs.top_ids_set:
                        pp = DPPs.PPs_lst[top_pp_id]
                        pp.test_base_rate[:, list(range(len(batch_ids))), :] = tot_base_rate_dict[
                            top_pp_id
                        ]
                    em_iters_count -= 1

                    # -------------------------------------------------------------------------------------
                    print(f"bottom ll = {tot_bottom_real_ll}", flush=True)
                    print(
                        f"num of events to predict = {remaining_count_to_predict}",
                        flush=True,
                    )
                    # -------------------------------------------------------------------------------------
            if remaining_count_to_predict == 0 and last_iter:
                break
    ex_prediction_name = log_folder_name + "prediction.log"
    with open(ex_prediction_name, "w") as f:
        print(
            "type pred correct num = ",
            type_pred_correct_num.tolist(),
            file=f,
            flush=True,
        )
        print(
            "time pred square err sum = ",
            time_pred_square_err_sum.tolist(),
            file=f,
            flush=True,
        )
        print(
            "tot evidence ll = ", tot_test_bottom_real_ll, file=f, flush=True
        )
        print("true pred num = ", true_pred_num, file=f, flush=True)
        print("num of events = ", num_of_events, file=f, flush=True)
        print(
            "num of events to predict = ",
            num_of_events_to_predict,
            file=f,
            flush=True,
        )
        print("whole time = ", time.time() - time0, file=f, flush=True)

