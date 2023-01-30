import modules.pps as pps
import torch
from modules.utils import prepare_dataloader
from collections import namedtuple
from modules.predict_time_type import predict_time_type
import time
import os

torch.set_default_dtype(torch.float32)
test_obs_size = 53
torch.manual_seed(0)

DPPs = pps.PPs(
    train_obs_size=0,
    dev_obs_size=0,
    test_obs_size=test_obs_size,
    processes_type="nsp",
    virtual_processes_type="nsp",
    end_time=None,
    dev_end_time=None,
    test_end_time=torch.zeros(1, test_obs_size, 1),
)
d_model = 32
d_inner = 64
DPPs.add_PP(
    id=0,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
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
    kernel_params=torch.tensor([4.4027, 0.4381, 77.5566])[None, None, None, :],
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=1,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
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
    kernel_params=torch.tensor([2.9016, 0.3394, 99.5761])[None, None, None, :],
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=2,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
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
    kernel_params=torch.tensor([0.9112, 0.5076, 0.1045])[None, None, None, :],
    virtual_kernel_params=torch.tensor([0.2888, 0.4952, 1.0565])[None, None, None, :],
    virtual_prop_background_rate=0.05,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=3,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
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
    kernel_params=torch.tensor([0.9827, 0.4073, 0.0381])[None, None, None, :],
    virtual_kernel_params=torch.tensor([0.4165, 0.4086, 0.8875])[None, None, None, :],
    virtual_prop_background_rate=0.05,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=4,
    end_time=None,
    top_base_rate=0.1,
    kernel_type="Weibull",
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
    virtual_kernel_params=torch.tensor([[4.4732e-01, 4.9931e-01, 3.1218e-02], [2.8689e+01, 4.4368e-03, 6.1280e-29]])[None, None, ...],
    virtual_prop_background_rate=0.05,
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    virtual_background_rate=torch.tensor([0.005]),
    background_rate_opt=False,
    predict=True,
)
DPPs.register_all_PPs(d_model)
DPPs.to(torch.float32)

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

opt.log = (
    "../experiments/earthquake/2_hidden/posterior/20/"
    + str(time.time())
    + "/"
)
os.makedirs(opt.log + "checkpoint")
opt.data = "/your/path/to/data_earthquake/"
opt.train_obs_size = 0
opt.dev_obs_size = 0
opt.test_obs_size = 53
opt.training_data_ratio = 1
opt.dev_data_ratio = 1
opt.test_data_ratio = 1
opt.device = "cuda"
opt.time_scale = 1e-3
opt.batch_size = 53

_, _, testloader, num_types = prepare_dataloader(opt)

test_evidences = next(iter(testloader)) if testloader else None

#-------------------------------------------------------------------
check_point = torch.load("../pretrained_models/earthquake/2_hidden_earthquake_nsp")
pretrained_dict = check_point["model_state_dict"]
pretrained_dict["test_end_time"] = DPPs.test_end_time
pretrained_dict["PPs_lst.4.base_rate"] = DPPs.PPs_lst[4].base_rate
pretrained_dict["PPs_lst.4.dev_base_rate"] = DPPs.PPs_lst[4].dev_base_rate
pretrained_dict["PPs_lst.4.test_base_rate"] = DPPs.PPs_lst[4].test_base_rate
del pretrained_dict["end_time"]
del pretrained_dict["dev_end_time"]
DPPs.load_state_dict(pretrained_dict)
#-------------------------------------------------------------------

with torch.no_grad():
    predict_time_type(
        DPPs,
        pred_sample_size=20,
        test_evidences=test_evidences[0],
        batch_size=opt.batch_size,
        em_iters=1,
        virtual_only_to_predict=False,
        device=opt.device,
        log_folder_name=opt.log,
        top_base_rate=0.1,
    )

