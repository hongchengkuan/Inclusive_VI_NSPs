from collections import namedtuple
from tqdm import tqdm
import time
import torch
import os
import modules.pps as pps
import modules.inference as inference
from modules.utils import prepare_dataloader
from modules.utils import process_events
from modules.utils import get_variational_params
from modules.utils import get_model_params
import math

torch.manual_seed(0)


def train_epoch(
    model, em_class, var_opt, model_opt, opt, fig, axs, calc_elbo, epoch
):

    cum_batch_size = 0
    tot_real_all_ll = 0
    tot_virtual_all_ll = 0
    tot_elbo = 0
    tot_bottom_real_ll = 0

    var_opt.zero_grad()
    model_opt.zero_grad()
    whole_obs_ids = torch.randperm(model.train_obs_size)
    num_iters = math.ceil(len(whole_obs_ids) / opt.batch_size)
    batch_ids_lst = [
        whole_obs_ids[i * opt.batch_size : (i + 1) * opt.batch_size].tolist()
        for i in range(num_iters)
    ]
    for batch_ids in tqdm(
        batch_ids_lst, mininterval=2, desc="  - (Training)  ", leave=False
    ):
        batch_size = len(batch_ids)
        num_iters_sampling = opt.tot_sample_size // opt.sample_size

        tot_base_rate_dict = {}
        for top_pp_id in em_class.PPs.top_ids_set:
            tot_base_rate_dict[top_pp_id] = torch.zeros(1, len(batch_ids), 1, device=opt.device)

        for n_iter_sampling in range(num_iters_sampling):
            if em_class.posterior_sampler.resample_only:
                for _ in range(1000):
                    (
                        real_all_ll,
                        virtual_all_ll,
                        virtual_all_ll_wrt_virtual,
                    ) = em_class.em_step(
                        batch_ids,
                        opt.burn_in_steps,
                        opt.sample_size,
                        opt.sample_intervals,
                        opt.log,
                        fig,
                        axs,
                        opt.save_mcmc,
                    )
                    loss = virtual_all_ll_wrt_virtual / 1000
                    loss.backward()
                for pp in model.PPs_lst:
                    if pp.virtual_processes_type == "general":
                        if not pp.bottom:
                            attn_grad = torch.max(
                                torch.stack(
                                    [
                                        torch.max(torch.abs(param.grad))
                                        for param in list(
                                            pp.attn_encoder_virtualPP.parameters()
                                        )
                                    ]
                                )
                            )
                            encoding_to_virtual = torch.max(
                                torch.stack(
                                    [
                                        torch.max(torch.abs(param.grad))
                                        for param in list(
                                            pp.encoding_to_virtual_kernel_nn.parameters()
                                        )
                                    ]
                                )
                            )
                            print(
                                "id = ",
                                pp.id,
                                " attn grad = ",
                                attn_grad,
                                "encoding to virtual = ",
                                encoding_to_virtual,
                                " virtual background rate = ",
                                pp.virtual_background_rate.grad,
                            )
                    else:
                        if not pp.bottom:
                            print(
                                "id = ",
                                pp.id,
                                " grad = ",
                                pp.virtual_kernel_params.grad,
                                " background rate = ",
                                pp.virtual_background_rate.grad,
                            )
            elif em_class.posterior_sampler.check_real_ll:
                real_all_ll, virtual_all_ll, _ = em_class.em_step(
                    batch_ids,
                    opt.burn_in_steps,
                    opt.sample_size,
                    opt.sample_intervals,
                    opt.log,
                    fig,
                    axs,
                    opt.save_mcmc,
                )
                real_all_ll.backward()
                for pp in model.PPs_lst:
                    if not pp.top:
                        print(
                            "id = ",
                            pp.id,
                            " grad = ",
                            pp.kernel_params.grad,
                            " background rate = ",
                            pp.background_rate,
                            " background rate grad = ",
                            pp.background_rate.grad,
                            flush=True,
                        )
                    else:
                        print(
                            "id = ",
                            pp.id,
                            " base rate grad = ",
                            pp.base_rate.grad,
                        )

            else:
                if not calc_elbo:
                    real_all_ll, virtual_all_ll, bottom_real_ll, base_rate_dict = em_class.em_step(
                        batch_ids,
                        opt.burn_in_steps,
                        opt.sample_size,
                        opt.sample_intervals,
                        opt.log,
                        fig,
                        axs,
                        opt.save_mcmc,
                        opt.mcmc_load_path,
                        calc_elbo,
                        first_after_opt=True if n_iter_sampling == 0 else False
                    )

                    for top_pp_id in em_class.PPs.top_ids_set:
                        tot_base_rate_dict[top_pp_id] += (base_rate_dict[top_pp_id] / num_iters_sampling)
                else:
                    real_all_ll, virtual_all_ll, elbo = em_class.em_step(
                        batch_ids,
                        opt.burn_in_steps,
                        opt.sample_size,
                        opt.sample_intervals,
                        opt.log,
                        fig,
                        axs,
                        opt.save_mcmc,
                        opt.mcmc_load_path,
                        calc_elbo,
                    )
                    tot_elbo += elbo


            loss = -(real_all_ll + virtual_all_ll) / (opt.update_batch_size / batch_size) / num_iters_sampling

            """backward"""
            loss.backward()

            tot_real_all_ll += real_all_ll.item() * batch_size / num_iters_sampling
            tot_virtual_all_ll += virtual_all_ll.item() * batch_size / num_iters_sampling
            tot_bottom_real_ll += bottom_real_ll * batch_size / num_iters_sampling

        for top_pp_id in em_class.PPs.top_ids_set:
            pp = em_class.PPs.PPs_lst[top_pp_id]
            pp.base_rate[:, batch_ids, :] = tot_base_rate_dict[top_pp_id]

        # print real grad
        for pp in model.PPs_lst:
            if not pp.top:
                print(
                    "id = ",
                    pp.id,
                    " grad = ",
                    pp.kernel_params.grad,
                    " background rate = ",
                    pp.background_rate.grad,
                )

        # # print virtual grad
        # for pp in model.PPs_lst:
        #     if not pp.bottom:
        #         print(
        #             "id = ",
        #             pp.id,
        #             " virtual grad = ",
        #             pp.virtual_kernel_params.grad,
        #             " virtual background rate = ",
        #             pp.virtual_background_rate.grad,
        #         )

        # print virtual grad
        for pp in model.PPs_lst:
            if not pp.bottom:
                print(
                    "id = ",
                    pp.id,
                    " virtual grad = ",
                    torch.sum(torch.stack(
                        [
                            torch.sum(torch.abs(p.grad))
                            for p in pp.attn_encoder_virtualPP.parameters()
                        ]
                    )),
                    " virtual background rate = ",
                    pp.virtual_background_rate.grad,
                )

        cum_batch_size += len(batch_ids)

        if (
            cum_batch_size % opt.update_batch_size == 0
            or cum_batch_size == model.train_obs_size
        ):
            with open(opt.log + "grad_track", "a") as f:
                for pp in model.PPs_lst:
                    if not pp.top:
                        print(
                            "epoch = ",
                            epoch,
                            "id = ",
                            pp.id,
                            " grad = ",
                            pp.kernel_params.grad,
                            " background rate = ",
                            pp.background_rate.grad,
                            file=f,
                        )
                    if not pp.bottom:
                        print(
                            "epoch = ",
                            epoch,
                            "id = ",
                            pp.id,
                            " virtual grad = ",
                            torch.sum(torch.stack(
                                [
                                    torch.sum(torch.abs(p.grad))
                                    for p in pp.attn_encoder_virtualPP.parameters()
                                ]
                            )),
                            " virtual background rate = ",
                            pp.virtual_background_rate.grad,
                            file=f,
                        )

            var_opt.step()
            var_opt.zero_grad()
            if not opt.model_opt_flag:
                if abs((opt.ll_cum_prev - tot_real_all_ll) / opt.ll_cum_prev) < 1e-2:
                    model_opt.step()
                    model_opt.zero_grad()
                    opt.model_opt_flag = True
            else:
                model_opt.step()
                model_opt.zero_grad()
            for pp in model.PPs_lst:
                if not pp.top:
                    with open(opt.log + "kernel_params", "a") as f:
                        print(
                            "epoch = ",
                            epoch,
                            "cum batch size = ",
                            cum_batch_size,
                            "id = ",
                            pp.id,
                            " kernel params = ",
                            pp.kernel_params, 
                            " background rate = ",
                            pp.background_rate, file=f
                        )
                else:
                    with open(opt.log + "base_rate", "a") as f:
                        print(
                            "epoch = ",
                            epoch,
                            "id = ",
                            pp.id,
                            " mean base rate = ",
                            torch.mean(pp.base_rate), file=f
                        )
                if not pp.bottom:
                    with open(opt.log + "kernel_params", "a") as f:
                        print(
                            "epoch = ",
                            epoch,
                            "id = ",
                            pp.id,
                            " virtual background rate = ",
                            pp.virtual_background_rate,
                            file=f,
                        )
            if cum_batch_size % opt.eval_batch_size == 0 and epoch % opt.epoch_record_gap == 0:
                checkpoint_name = opt.log + "checkpoint/" + str(epoch) + "_" + str(cum_batch_size)
                rng_state = torch.get_rng_state()
                torch.save(
                    {
                        "epoch": epoch,
                        "cum_batch_size": cum_batch_size,
                        "model_state_dict": model.state_dict(),
                        "var_opt": var_opt.state_dict(),
                        "model_opt": model_opt.state_dict(),
                        "rng_state": rng_state,
                    },
                    checkpoint_name,
                )
                if opt.dev_obs_size != 0:
                    with torch.random.fork_rng(enabled=True, devices=[]):
                        with torch.no_grad():
                            eval_epoch(
                                model,
                                em_class,
                                opt,
                                fig,
                                axs,
                                calc_elbo=False,
                                data_type="dev",
                                epoch=epoch,
                                cum_batch_size=cum_batch_size,
                            )
                            with open(opt.log + "dev_base_rate", "a") as f:
                                print(
                                    "epoch = ",
                                    epoch,
                                    "id = ",
                                    pp.id,
                                    " mean base rate = ",
                                    torch.mean(pp.dev_base_rate), file=f
                                )
                if opt.test_obs_size != 0:
                    with torch.random.fork_rng(enabled=True, devices=[]):
                        with torch.no_grad():
                            eval_epoch(
                                model,
                                em_class,
                                opt,
                                fig,
                                axs,
                                calc_elbo=False,
                                data_type="test",
                                epoch=epoch,
                                cum_batch_size=cum_batch_size,
                            )
                            with open(opt.log + "test_base_rate", "a") as f:
                                print(
                                    "epoch = ",
                                    epoch,
                                    "id = ",
                                    pp.id,
                                    " mean base rate = ",
                                    torch.mean(pp.test_base_rate), file=f
                                )

    opt.ll_cum_prev = tot_real_all_ll
    if calc_elbo:
        return tot_real_all_ll, tot_virtual_all_ll, tot_elbo
    else:
        return tot_real_all_ll, tot_virtual_all_ll, tot_bottom_real_ll

def eval_epoch(
    model, em_class, opt, fig, axs, calc_elbo, data_type, epoch, cum_batch_size
):
    start = time.time()

    tot_real_all_ll = 0
    tot_virtual_all_ll = 0
    tot_elbo = 0
    tot_bottom_real_ll = 0

    if data_type == "dev":
        whole_obs_ids = torch.randperm(model.dev_obs_size)
    elif data_type == "test":
        whole_obs_ids = torch.randperm(model.test_obs_size)
    num_iters = math.ceil(len(whole_obs_ids) / opt.batch_size)
    batch_ids_lst = [
        whole_obs_ids[i * opt.batch_size : (i + 1) * opt.batch_size].tolist()
        for i in range(num_iters)
    ]
    for batch_ids in tqdm(
        batch_ids_lst, mininterval=2, desc="  - (Evaluating)  ", leave=False
    ):
        batch_size = len(batch_ids)
        num_iters_sampling = opt.tot_sample_size // opt.sample_size

        tot_base_rate_dict = {}
        for top_pp_id in em_class.PPs.top_ids_set:
            tot_base_rate_dict[top_pp_id] = torch.zeros(1, len(batch_ids), 1, device=opt.device)

        for n_iters_sampling in range(num_iters_sampling):
            if not calc_elbo:
                real_all_ll, virtual_all_ll, bottom_real_ll, base_rate_dict = em_class.em_step(
                    batch_ids,
                    opt.burn_in_steps,
                    opt.sample_size,
                    opt.sample_intervals,
                    opt.log,
                    fig,
                    axs,
                    opt.save_mcmc,
                    opt.mcmc_load_path,
                    calc_elbo,
                    data_type=data_type,
                )


                for top_pp_id in em_class.PPs.top_ids_set:
                    tot_base_rate_dict[top_pp_id] += (base_rate_dict[top_pp_id] / num_iters_sampling)
            else:
                real_all_ll, virtual_all_ll, elbo = em_class.em_step(
                    batch_ids,
                    opt.burn_in_steps,
                    opt.sample_size,
                    opt.sample_intervals,
                    opt.log,
                    fig,
                    axs,
                    opt.save_mcmc,
                    opt.mcmc_load_path,
                    calc_elbo,
                    data_type=data_type,
                )
                tot_elbo += elbo


            tot_real_all_ll += real_all_ll.item() * batch_size / num_iters_sampling
            tot_virtual_all_ll += virtual_all_ll.item() * batch_size / num_iters_sampling
            tot_bottom_real_ll += bottom_real_ll * batch_size / num_iters_sampling

        for top_pp_id in em_class.PPs.top_ids_set:
            pp = em_class.PPs.PPs_lst[top_pp_id]
            if data_type == "dev":
                pp.dev_base_rate[:, batch_ids, :] = tot_base_rate_dict[top_pp_id]
            elif data_type == "test":
                pp.test_base_rate[:, batch_ids, :] = tot_base_rate_dict[top_pp_id]
    if data_type == "dev":
        print(
            "  - (Validation)     real_all_ll: {real_all_ll: 8.5f}, virtual_all_ll: {virtual_all_ll: 8.5f}  time: {elapse}".format(
                real_all_ll=tot_real_all_ll / opt.num_dev_events + math.log(opt.time_scale),
                virtual_all_ll=tot_virtual_all_ll / opt.num_dev_events + math.log(opt.time_scale),
                elapse=(time.time() - start) / 60,
            )
        )
        validation_log = opt.log + "validation.log"
        with open(validation_log, "a") as f:
            f.write(
                "{epoch}, {cum_batch_size}, {real_all_ll: 8.5f}, {virtual_all_ll: 8.5f}, {bottom_real_ll: 8.5f}\n".format(
                    epoch=epoch,
                    cum_batch_size=cum_batch_size,
                    real_all_ll=tot_real_all_ll / opt.num_dev_events + math.log(opt.time_scale),
                    virtual_all_ll=tot_virtual_all_ll / opt.num_dev_events + math.log(opt.time_scale),
                    bottom_real_ll=tot_bottom_real_ll / opt.num_dev_events + math.log(opt.time_scale),
                )
            )
    elif data_type == "test":
        print(
            "  - (Test)     real_all_ll: {real_all_ll: 8.5f}, virtual_all_ll: {virtual_all_ll: 8.5f}  time: {elapse}".format(
                real_all_ll=tot_real_all_ll / opt.num_test_events + math.log(opt.time_scale),
                virtual_all_ll=tot_virtual_all_ll / opt.num_test_events + math.log(opt.time_scale),
                elapse=(time.time() - start) / 60,
            )
        )
        test_log = opt.log + "test.log"
        with open(test_log, "a") as f:
            f.write(
                "{epoch}, {cum_batch_size}, {real_all_ll: 8.5f}, {virtual_all_ll: 8.5f}, {bottom_real_ll: 8.5f}\n".format(
                    epoch=epoch,
                    cum_batch_size=cum_batch_size,
                    real_all_ll=tot_real_all_ll / opt.num_test_events + math.log(opt.time_scale),
                    virtual_all_ll=tot_virtual_all_ll / opt.num_test_events + math.log(opt.time_scale),
                    bottom_real_ll=tot_bottom_real_ll / opt.num_test_events + math.log(opt.time_scale),
                )
            )

def train(
    model,
    var_opt,
    model_opt,
    opt,
    em_class,
    fig=None,
    axs=None,
):
    with open(opt.log + "train.log", "a") as f:
        f.write('epoch, real_all_ll, virtual_all_ll, bottom_real_all_ll, elapse\n')
    with open(opt.log + "validation.log", "a") as f:
        f.write('epoch, cum_batch_size, real_all_ll, virtual_all_ll, bottom_real_ll\n')
    with open(opt.log + "test.log", "a") as f:
        f.write('epoch, cum_batch_size, real_all_ll, virtual_all_ll, bottom_real_ll\n')
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1

        process_events(model)
        start = time.time()
        real_all_ll, virtual_all_ll, bottom_real_all_ll = train_epoch(
            model,
            em_class,
            var_opt,
            model_opt,
            opt,
            fig,
            axs,
            calc_elbo=False,
            epoch=epoch_i,
        )
        if epoch_i % 1 == 0:
            print("[ Epoch", epoch, "]")
            print(
                "  - (Training)     real_all_ll: {real_all_ll: 8.5f}, virtual_all_ll: {virtual_all_ll: 8.5f}  time: {elapse}".format(
                    real_all_ll=real_all_ll,
                    virtual_all_ll=virtual_all_ll,
                    elapse=(time.time() - start) / 60,
                )
            )

        if epoch_i % opt.epoch_record_gap == 0:
            train_log = opt.log + "train.log"
            with open(train_log, "a") as f:
                f.write(
                    "{epoch}, {real_all_ll: 8.5f}, {virtual_all_ll: 8.5f}, {bottom_real_all_ll:8.5f}, {elapse}\n".format(
                        epoch=epoch,
                        real_all_ll=real_all_ll / opt.num_train_events + math.log(opt.time_scale),
                        virtual_all_ll=virtual_all_ll / opt.num_train_events + math.log(opt.time_scale),
                        bottom_real_all_ll=bottom_real_all_ll / opt.num_train_events + math.log(opt.time_scale),
                        elapse=(time.time() - opt.initial_start) / 60,
                    )
                )
            # # print real parameters
            # for pp in model.PPs_lst:
            #     if not pp.top:
            #         print(
            #             "id = ",
            #             pp.id,
            #             " kernel params = ",
            #             pp.kernel_params,
            #             " background rate = ",
            #             pp.background_rate,
            #         )
            #     else:
            #         print(
            #             "id = ",
            #             pp.id,
            #             " base rate = ",
            #             pp.base_rate,
            #         )

            # # print virtual parameters
            # for pp in model.PPs_lst:
            #     if not pp.bottom:
            #         print(
            #             "id = ",
            #             pp.id,
            #             " virtual kernel params = ",
            #             pp.virtual_kernel_params,
            #             " virtual background rate = ",
            #             pp.virtual_background_rate,
            #         )


if __name__ == "__main__":
    opt = namedtuple(
        "option",
        [
            "epoch",
            "log",
            "data",
            "batch_size",
            "update_batch_size",
            "device",
            "time_scale",
        ],
    )

    opt.epoch = 10000
    opt.log = (
        "../experiments/test/"
        + str(time.time())
        + "/"
    )
    os.makedirs(opt.log + "checkpoint")
    opt.data = "/your/path/to/synthetic_data/2_hidden/"
    opt.train_obs_size = 1000
    opt.dev_obs_size = 100
    opt.test_obs_size = 100
    opt.batch_size = 500
    opt.training_data_ratio = 1
    opt.dev_data_ratio = 1
    opt.test_data_ratio = 1
    opt.update_batch_size = 500
    opt.model_opt_flag = True
    opt.eval_batch_size = 500
    opt.time_scale = 1
    opt.burn_in_steps = 0
    opt.tot_sample_size = 80
    opt.sample_size = 1
    opt.sample_intervals = 1
    opt.save_mcmc = False
    opt.mcmc_load_path = None
    opt.calc_elbo = False
    opt.epoch_record_gap = 1
    opt.device = "cuda"
    opt.ll_cum_prev = 1e10
    opt.initial_start = time.time()

    d_model = 32
    trainloader, devloader, testloader, num_types = prepare_dataloader(opt)


    evidences = next(iter(trainloader))
    with torch.random.fork_rng(enabled=True, devices=[]):
        dev_evidences = next(iter(devloader)) if devloader else None
        test_evidences = next(iter(testloader)) if testloader else None

    opt.num_train_events = sum([torch.sum(evidences[0][i] < trainloader.dataset.pad) for i in range(num_types)])
    opt.num_dev_events = sum([torch.sum(dev_evidences[0][i] < devloader.dataset.pad) for i in range(num_types)]) if opt.dev_obs_size != 0 else 0
    opt.num_test_events = sum([torch.sum(test_evidences[0][i] < testloader.dataset.pad) for i in range(num_types)]) if opt.test_obs_size != 0 else 0

    end_time = torch.tensor([20.])[None, :, None].expand(-1, 1000, 1)
    dev_end_time = torch.tensor([20.])[None, :, None].expand(-1, 100, -1)
    test_end_time = torch.tensor([20.])[None, :, None].expand(-1, 100, -1)
    DPPs = pps.PPs(
        train_obs_size=opt.train_obs_size, 
        dev_obs_size=opt.dev_obs_size, 
        test_obs_size=opt.test_obs_size, 
        processes_type="nsp", 
        virtual_processes_type="general", 
        end_time=end_time, 
        dev_end_time=dev_end_time, 
        test_end_time=test_end_time,
    )
    d_model = 32
    d_inner = 64
    DPPs.add_PP(
        id=0,
        end_time=None,
        top_base_rate=None,
        kernel_type="Exponential",
        parents_ids_dict={2:0},
        parents_ids_string="2",
        children_ids_dict=None,
        children_ids_string=None,
        attn_n_layers=1,
        d_model=d_model,
        d_inner=d_inner,
        n_head=4,
        top=False,
        bottom=True,
        kernel_params=torch.tensor([3.0, 5.0])[None, None, None, :],
        evidence_events_times=evidences[0][0],
        evidence_events_mask=evidences[0][0] < trainloader.dataset.pad,
        dev_evidence_events_times=dev_evidences[0][0] if opt.dev_obs_size != 0 else None,
        dev_evidence_events_mask=dev_evidences[0][0] < devloader.dataset.pad if opt.dev_obs_size != 0 else None,
        test_evidence_events_times=test_evidences[0][0] if opt.test_obs_size != 0 else None,
        test_evidence_events_mask=test_evidences[0][0] < testloader.dataset.pad if opt.test_obs_size != 0 else None,
        dropout=0.1,
        kernel_params_opt=True,
        base_rate_opt=True,
        background_rate_opt=False,
    )
    DPPs.add_PP(
        id=1,
        end_time=None,
        top_base_rate=None,
        kernel_type="Exponential",
        parents_ids_dict={3:0},
        parents_ids_string="3",
        children_ids_dict=None,
        children_ids_string=None,
        attn_n_layers=1,
        d_model=d_model,
        d_inner=d_inner,
        n_head=4,
        top=False,
        bottom=True,
        kernel_params=torch.tensor([3.0, 6.0])[None, None, None, :],
        evidence_events_times=evidences[0][1],
        evidence_events_mask=evidences[0][1] < trainloader.dataset.pad,
        dev_evidence_events_times=dev_evidences[0][1] if opt.dev_obs_size != 0 else None,
        dev_evidence_events_mask=dev_evidences[0][1] < devloader.dataset.pad if opt.dev_obs_size != 0 else None,
        test_evidence_events_times=test_evidences[0][1] if opt.test_obs_size != 0 else None,
        test_evidence_events_mask=test_evidences[0][1] < testloader.dataset.pad if opt.test_obs_size != 0 else None,
        dropout=0.1,
        kernel_params_opt=True,
        base_rate_opt=True,
        background_rate_opt=False,
    )
    DPPs.add_PP(
        id=2,
        end_time=None,
        top_base_rate=None,
        kernel_type="Exponential",
        parents_ids_dict={4:0},
        parents_ids_string="4",
        children_ids_dict={0:0},
        children_ids_string="0",
        attn_n_layers=1,
        d_model=d_model,
        d_inner=d_inner,
        n_head=4,
        top=False,
        bottom=False,
        kernel_params=torch.tensor([3.0, 4.0])[None, None, None, :],
        virtual_kernel_params=torch.tensor([1 / 3.0, 5.0])[None, None, None, :],
        virtual_prop_background_rate=0.1,
        dropout=0.0,
        kernel_params_opt=True,
        base_rate_opt=True,
        background_rate_opt=False,
    )
    DPPs.add_PP(
        id=3,
        end_time=None,
        top_base_rate=0.15,
        kernel_type="Exponential",
        parents_ids_dict={4:0},
        parents_ids_string="4",
        children_ids_dict={1:0},
        children_ids_string="1",
        attn_n_layers=1,
        d_model=d_model,
        d_inner=d_inner,
        n_head=4,
        top=False,
        bottom=False,
        kernel_params=torch.tensor([2.0, 2.0])[None, None, None, :],
        virtual_kernel_params=torch.tensor([1 / 3.0, 6.0])[None, None, None, :],
        virtual_prop_background_rate=0.1,
        dropout=0.0,
        kernel_params_opt=True,
        base_rate_opt=True,
        background_rate_opt=False,
    )
    DPPs.add_PP(
        id=4,
        end_time=None,
        top_base_rate=0.15,
        kernel_type="Exponential",
        parents_ids_dict=None,
        parents_ids_string=None,
        children_ids_dict={2:0, 3:1},
        children_ids_string="2,3",
        attn_n_layers=1,
        d_model=d_model,
        d_inner=d_inner,
        n_head=4,
        top=True,
        bottom=False,
        virtual_kernel_params=torch.tensor([[1 / 3.0, 4.0], [1 / 2.0, 2.0]])[None, None, ...],
        virtual_prop_background_rate=0.1,
        dropout=0.0,
        kernel_params_opt=True,
        base_rate_opt=True,
    )
    DPPs.register_all_PPs(d_model)
    DPPs.to(torch.float32)

    torch.set_default_dtype(torch.float32)
    resample_only = False
    record_virtual_samples = False
    check_real_ll = False
    em_class = inference.EM(
        PPs=DPPs,
        events_embedding=None,
        initialize_batch_size=opt.batch_size,
        resample_only=resample_only,
        record_virtual_samples=record_virtual_samples,
        check_real_ll=check_real_ll,
        device=opt.device,
    )

    var_opt = torch.optim.Adam(get_variational_params(DPPs, nn_lr=1e-2), 1e-2, eps=1e-06)
    model_opt = torch.optim.Adam(get_model_params(DPPs, nn_lr=1e-2), 1e-2, eps=1e-06)


    train(DPPs, var_opt, model_opt, opt, em_class)
