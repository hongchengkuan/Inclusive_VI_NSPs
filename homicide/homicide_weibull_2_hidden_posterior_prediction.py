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
    kernel_params=torch.tensor([8.4119e+00, 7.5721e-03, 7.8059e+01])[None, None, None, :],
    dropout=0.0,
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
    parents_ids_dict={6:0},
    parents_ids_string="6",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([9.1914e+00, 8.6920e-03, 5.2562e+01])[None, None, None, :],
    dropout=0.0,
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
    parents_ids_dict={7:0},
    parents_ids_string="7",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([1.0448e+01, 7.5052e-03, 3.4270e+01])[None, None, None, :],
    dropout=0.0,
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
    parents_ids_dict={8:0},
    parents_ids_string="8",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([9.0904e+00, 5.2715e-03, 3.6869e+01])[None, None, None, :],
    dropout=0.0,
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
    parents_ids_dict={9:0},
    parents_ids_string="9",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([9.4265e+00, 6.0384e-03, 8.4225e+01])[None, None, None, :],
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    predict=True,
)
DPPs.add_PP(
    id=5,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={10:0},
    parents_ids_string="10",
    children_ids_dict={0:0},
    children_ids_string="0",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([8.2456e+00, 2.8169e-03, 2.0216e+01])[None, None, None, :],
    virtual_kernel_params=torch.tensor([1.2792e+01, 9.3789e-03, 3.2492e-13])[None, None, None, :],
    virtual_prop_background_rate=0.001,
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0019]),
    predict=True,
)
DPPs.add_PP(
    id=6,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={10:0},
    parents_ids_string="10",
    children_ids_dict={1:0},
    children_ids_string="1",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([7.7577e+00, 1.7265e-03, 1.8154e+01])[None, None, None, :],
    virtual_kernel_params=torch.tensor([1.1896e+01, 9.9669e-03, 2.6366e-20])[None, None, None, :],
    virtual_prop_background_rate=0.001,
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0011]),
    predict=True,
)
DPPs.add_PP(
    id=7,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={10:0},
    parents_ids_string="10",
    children_ids_dict={2:0},
    children_ids_string="2",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([8.1409e+00, 1.6933e-03, 7.5338e+00])[None, None, None, :],
    virtual_kernel_params=torch.tensor([1.2292e+01, 1.1989e-02, 4.0830e-04])[None, None, None, :],
    virtual_prop_background_rate=0.001,
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0008]),
    predict=True,
)
DPPs.add_PP(
    id=8,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={10:0},
    parents_ids_string="10",
    children_ids_dict={3:0},
    children_ids_string="3",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([8.2435e+00, 3.2405e-03, 2.0906e-17])[None, None, None, :],
    virtual_kernel_params=torch.tensor([12.0378,  0.0121, 11.6030])[None, None, None, :],
    virtual_prop_background_rate=0.001,
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0024]),
    predict=True,
)
DPPs.add_PP(
    id=9,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={10:0},
    parents_ids_string="10",
    children_ids_dict={4:0},
    children_ids_string="4",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([7.8726e+00, 3.5198e-03, 3.0154e-12])[None, None, None, :],
    virtual_kernel_params=torch.tensor([1.1724e+01, 1.0812e-02, 9.3363e+00])[None, None, None, :],
    virtual_prop_background_rate=0.001,
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0024]),
    predict=True,
)
DPPs.add_PP(
    id=10,
    end_time=None,
    top_base_rate=0.1,
    kernel_type="Weibull",
    parents_ids_dict=None,
    parents_ids_string=None,
    children_ids_dict={5:0, 6:1, 7:2, 8:3, 9:4},
    children_ids_string="5,6,7,8,9",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=True,
    bottom=False,
    virtual_kernel_params=torch.tensor([[4.1824e+00, 3.8484e-02, 2.0489e-19],
    [4.0347e+00, 3.8420e-02, 1.1096e-16],
    [1.0932e+01, 1.0640e-02, 6.4400e-11],
    [4.0917e+00, 4.7272e-02, 1.9550e-15],
    [1.1599e+01, 9.4511e-03, 3.3510e+02]])[None, None, ...],
    virtual_prop_background_rate=0.001,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    virtual_background_rate=torch.tensor([0.0091]),
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
    "../experiments/homicide_prediction/2_hidden/posterior/20/"
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
check_point = torch.load("../pretrained_models/homicide/2_hidden_homicide_nsp")
pretrained_dict = check_point["model_state_dict"]
pretrained_dict["test_end_time"] = DPPs.test_end_time
pretrained_dict["PPs_lst.10.base_rate"] = DPPs.PPs_lst[10].base_rate
pretrained_dict["PPs_lst.10.dev_base_rate"] = DPPs.PPs_lst[10].dev_base_rate
pretrained_dict["PPs_lst.10.test_base_rate"] = DPPs.PPs_lst[10].test_base_rate
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

