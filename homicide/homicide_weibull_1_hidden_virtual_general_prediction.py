import modules.pps as pps
import torch
from modules.utils import prepare_dataloader
from collections import namedtuple
from modules.predict_time_type import predict_time_type
import time
import os

torch.set_default_dtype(torch.float32)
test_obs_size = 2
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
d_model = 32
d_inner = 64
DPPs.add_PP(
    id=0,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={5:0},
    parents_ids_string="5",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([1.4194e+01, 7.8432e-04, 9.4136e-16])[None, None, None, :],
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
    parents_ids_dict={5:0},
    parents_ids_string="5",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([1.5574e+01, 4.8724e-04, 4.9430e-15])[None, None, None, :],
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
    parents_ids_dict={5:0},
    parents_ids_string="5",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([1.5399e+01, 5.2702e-04, 2.5813e-15])[None, None, None, :],
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
    parents_ids_dict={5:0},
    parents_ids_string="5",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([1.6204e+01, 5.6320e-04, 4.2214e-15])[None, None, None, :],
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=4,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={5:0},
    parents_ids_string="5",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([1.7096e+01, 6.7924e-04, 3.6976e-15])[None, None, None, :],
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=5,
    end_time=None,
    top_base_rate=0.1,
    kernel_type="Weibull",
    parents_ids_dict=None,
    parents_ids_string=None,
    children_ids_dict={0:0, 1:1, 2:2, 3:3, 4:4},
    children_ids_string="0,1,2,3,4",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=True,
    bottom=False,
    virtual_kernel_params=torch.tensor([[4.6588e+00, 4.5732e-02, 1.8025e-16],
      [4.0986e+00, 4.1861e-02, 1.7031e-16],
      [6.8185e+00, 1.7065e-02, 2.3661e-16],
      [6.5204e+00, 1.7283e-02, 7.4959e-16],
      [3.4837e+01, 4.1731e-03, 3.3378e+02]])[None, None, ...],
    virtual_prop_background_rate=0.01,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    virtual_background_rate=torch.tensor([0.0134]),
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
    "../experiments/homicide/"
    + str(time.time())
    + "/"
)
os.makedirs(opt.log + "checkpoint")
opt.data = "/your/path/to/data_homicide/"
opt.train_obs_size = 0
opt.dev_obs_size = 0
opt.test_obs_size = 2
opt.training_data_ratio = 1
opt.dev_data_ratio = 1
opt.test_data_ratio = 1
opt.device = "cuda"
opt.time_scale = 1e-3
opt.batch_size = 2

_, _, testloader, num_types = prepare_dataloader(opt)

test_evidences = next(iter(testloader)) if testloader else None

#-------------------------------------------------------------------
check_point = torch.load("../pretrained_models/homicide/1_hidden_homicide_general")
pretrained_dict = check_point["model_state_dict"]
pretrained_dict["test_end_time"] = DPPs.test_end_time
pretrained_dict["PPs_lst.5.base_rate"] = DPPs.PPs_lst[5].base_rate
pretrained_dict["PPs_lst.5.dev_base_rate"] = DPPs.PPs_lst[5].dev_base_rate
pretrained_dict["PPs_lst.5.test_base_rate"] = DPPs.PPs_lst[5].test_base_rate
del pretrained_dict["end_time"]
del pretrained_dict["dev_end_time"]
DPPs.load_state_dict(pretrained_dict)
#-------------------------------------------------------------------

with torch.no_grad():
    predict_time_type(
        DPPs,
        pred_sample_size=3,
        test_evidences=test_evidences[0],
        batch_size=opt.batch_size,
        em_iters=5,
        virtual_only_to_predict=True,
        device=opt.device,
        log_folder_name=opt.log,
        top_base_rate=0.1,
    )

