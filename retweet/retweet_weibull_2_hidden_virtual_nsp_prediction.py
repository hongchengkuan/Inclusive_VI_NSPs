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
    kernel_params=torch.tensor([9.1008, 0.2915, 2.6210])[None, None, None, :],
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
    parents_ids_dict={4:0},
    parents_ids_string="4",
    children_ids_dict=None,
    children_ids_string=None,
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=True,
    kernel_params=torch.tensor([6.6331, 0.2067, 1.1311])[None, None, None, :],
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
    kernel_params=torch.tensor([1.3897, 0.2326, 0.1403])[None, None, None, :],
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
    parents_ids_dict={6:0},
    parents_ids_string="6",
    children_ids_dict={0:0},
    children_ids_string="0",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([3.9000, 0.4303, 0.1875])[None, None, None, :],
    virtual_kernel_params=torch.tensor([0.1882, 0.2826, 0.0041])[None, None, None, :],
    virtual_prop_background_rate=0.05,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0002]),
    predict=True,
)

DPPs.add_PP(
    id=4,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={6:0},
    parents_ids_string="6",
    children_ids_dict={1:0},
    children_ids_string="1",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([4.5608, 0.7038, 0.842])[None, None, None, :],
    virtual_kernel_params=torch.tensor([0.3797, 0.1965, 0.0007])[None, None, None, :],
    virtual_prop_background_rate=0.05,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0003]),
    predict=True,
)

DPPs.add_PP(
    id=5,
    end_time=None,
    top_base_rate=None,
    kernel_type="Weibull",
    parents_ids_dict={6:0},
    parents_ids_string="6",
    children_ids_dict={2:0},
    children_ids_string="2",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=False,
    bottom=False,
    kernel_params=torch.tensor([18.574, 0.0365, 7.8641e-13])[None, None, None, :],
    virtual_kernel_params=torch.tensor([1.3582, 0.2370, 0.2318])[None, None, None, :],
    virtual_prop_background_rate=0.05,
    dropout=0.1,
    kernel_params_opt=True,
    base_rate_opt=True,
    background_rate_opt=False,
    virtual_background_rate=torch.tensor([0.0027]),
    predict=True,
)

DPPs.add_PP(
    id=6,
    end_time=None,
    top_base_rate=0.1,
    kernel_type="Weibull",
    parents_ids_dict=None,
    parents_ids_string=None,
    children_ids_dict={3:0, 4:1, 5:2},
    children_ids_string="3,4,5",
    attn_n_layers=1,
    d_model=d_model,
    d_inner=d_inner,
    n_head=4,
    top=True,
    bottom=False,
    virtual_kernel_params=torch.tensor([[1.9628e+01, 1.0807e-03, 1.8779e-11], [8.4594e-02, 4.8003e-01, 1.4625e-02], [8.8702e-01, 2.7878e-01, 3.7765e-05]])[None, None, ...],
    virtual_prop_background_rate=0.05,
    dropout=0.0,
    kernel_params_opt=True,
    base_rate_opt=True,
    virtual_background_rate=torch.tensor([1.2023e-06]),
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
    "../experiments/retweet_prediction/2_hidden/nsp/200/"
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
opt.batch_size = 2000

_, _, testloader, num_types = prepare_dataloader(opt)

test_evidences = next(iter(testloader)) if testloader else None

#-------------------------------------------------------------------
check_point = torch.load("../pretrained_models/retweet/2_hidden_retweet_nsp")
pretrained_dict = check_point["model_state_dict"]
pretrained_dict["test_end_time"] = DPPs.test_end_time
pretrained_dict["PPs_lst.6.base_rate"] = DPPs.PPs_lst[6].base_rate
pretrained_dict["PPs_lst.6.dev_base_rate"] = DPPs.PPs_lst[6].dev_base_rate
pretrained_dict["PPs_lst.6.test_base_rate"] = DPPs.PPs_lst[6].test_base_rate
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
        em_iters=5,
        virtual_only_to_predict=True,
        device=opt.device,
        log_folder_name=opt.log,
        top_base_rate=0.1,
    )

