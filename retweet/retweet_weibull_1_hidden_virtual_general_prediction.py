import modules.pps as pps
import torch
from modules.utils import prepare_dataloader
from collections import namedtuple
from modules.predict_time_type import predict_time_type
import time
import os

torch.set_default_dtype(torch.float32)
test_obs_size = 2000
torch.manual_seed(0)

DPPs = pps.PPs(
    train_obs_size=0,
    dev_obs_size=0,
    test_obs_size=test_obs_size,
    processes_type="nsp",
    virtual_processes_type="general",
    end_time=None,
    dev_end_time=None,
    test_end_time=torch.zeros(1, test_obs_size, 1),
)


d_model = 64
d_inner = 128
DPPs.add_PP(
    id=0,
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
    kernel_params=torch.tensor([20.0857, 0.3622, 3.7983])[None, None, None, :],
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
    kernel_params=torch.tensor([17.0176, 0.2776, 2.1416])[None, None, None, :],
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
    kernel_params=torch.tensor([2.0877, 0.2170, 2.0956])[None, None, None, :],
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=3,
    end_time=None,
    top_base_rate=0.1,
    kernel_type="Weibull",
    parents_ids_dict=None,
    parents_ids_string=None,
    children_ids_dict={0:0, 1:1, 2:2},
    children_ids_string="0,1,2",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=True,
    bottom=False,
    virtual_kernel_params=torch.tensor([[18.452, 3.33e-04, 4.60e-02], [0.26, 0.17, 2.30e-06], [0.36, 0.21, 2.77e-05]])[None, None, ...],
    virtual_prop_background_rate=0.05,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    virtual_background_rate=torch.tensor([1.09e-09]),
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
    "../experiments/retweet_prediction/1_hidden/general/"
    + str(time.time())
    + "/"
)
os.makedirs(opt.log + "checkpoint")
opt.data = "/your/path/to/data_retweet/"
opt.train_obs_size = 0
opt.dev_obs_size = 0
opt.test_obs_size = 2000
opt.training_data_ratio = 1
opt.dev_data_ratio = 1
opt.test_data_ratio = 1
opt.device = "cuda"
opt.time_scale = 1e-3
opt.batch_size = 500

_, _, testloader, num_types = prepare_dataloader(opt)

test_evidences = next(iter(testloader)) if testloader else None

#-------------------------------------------------------------------
check_point = torch.load("../pretrained_models/retweet/1_hidden_retweet_general")
pretrained_dict = check_point["model_state_dict"]
pretrained_dict["test_end_time"] = DPPs.test_end_time
pretrained_dict["PPs_lst.3.base_rate"] = DPPs.PPs_lst[3].base_rate
pretrained_dict["PPs_lst.3.dev_base_rate"] = DPPs.PPs_lst[3].dev_base_rate
pretrained_dict["PPs_lst.3.test_base_rate"] = DPPs.PPs_lst[3].test_base_rate
del pretrained_dict["end_time"]
del pretrained_dict["dev_end_time"]
DPPs.load_state_dict(pretrained_dict)
#-------------------------------------------------------------------

with torch.no_grad():
    predict_time_type(
        DPPs,
        pred_sample_size=200,
        test_evidences=test_evidences[0],
        batch_size=opt.batch_size,
        em_iters=1,
        virtual_only_to_predict=True,
        device=opt.device,
        log_folder_name=opt.log,
        top_base_rate=0.1,
    )

