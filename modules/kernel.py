import torch
import torch.nn as nn
import math
F = nn.functional
F_softplus = F.softplus
torch_exp = torch.exp
torch_pow = torch.pow
torch_log = torch.log

class WeibullShift:
    def __init__(self) -> None:
        pass

    @staticmethod
    def forward(x, params):
        # f(x) = s * k / \lambda * ((x+a) / \lambda) ^ (k - 1) * exp(-((x+a) / \lambda) ^ k)
        _params = F_softplus(params)
        s = _params[0]
        a = _params[1]
        k = _params[2]
        lam = _params[3]
        return s * k / lam * ((x + a) / lam) ** (k - 1) * torch_exp(-((x + a) / lam) ** k)

    @staticmethod
    def integral(x, params):
        # f(x) = s * (exp(-(a / lam) ^ k) - exp(-((a+x) / lam) ^ k))
        assert (x >= 0).all()
        _params = F_softplus(params)
        s = _params[0]
        a = _params[1]
        k = _params[2]
        lam = _params[3]
        return s * (torch_exp(-(a / lam) ** k) - torch_exp(-((a + x) / lam) ** k))

    @staticmethod
    def integral_inv(y, params):
        _params = F_softplus(params)
        s = _params[0]
        a = _params[1]
        k = _params[2]
        lam = _params[3]
        return lam * torch_pow(- torch_log((s * torch_exp(-(a / lam) ** k) - y) / s), 1 / k) - a

class Weibull:
    def __init__(self) -> None:
        pass

    @staticmethod
    def forward(x, params):
        # f(x) = s * k / \lambda * (x / \lambda) ^ (k - 1) * exp(-(x / \lambda) ^ k)
        _params = F_softplus(params)
        s = _params[..., 0]
        k = _params[..., 1]
        lam = _params[..., 2] + 1e-10
        x_pad_mask = x > 1e18
        x_calc = x.masked_fill(x_pad_mask, 1) + 1e-4

        x_calc_div_lam = x_calc / lam
        inf_check_1 = x_calc_div_lam ** (k - 1)
        inf_check_1_inf_mask = torch.isinf(inf_check_1 * torch.log(x_calc_div_lam))
        if inf_check_1_inf_mask.any():
            x_calc_div_lam_fake = x_calc_div_lam.masked_fill(inf_check_1_inf_mask, 1)
            inf_check_1 = x_calc_div_lam_fake ** (k - 1)
            inf_check_1 = inf_check_1.masked_fill(inf_check_1_inf_mask, 1e10)
        inf_check_2 = inf_check_1 * x_calc_div_lam
        inf_check_2_inf_mask = torch.isinf(inf_check_2 * torch.log(x_calc_div_lam))
        if inf_check_2_inf_mask.any():
            x_calc_div_lam_fake = x_calc_div_lam.masked_fill(inf_check_2_inf_mask, 1)
            inf_check_2 = inf_check_1 * x_calc_div_lam_fake
            inf_check_2 = inf_check_2.masked_fill(inf_check_2_inf_mask, 1e10)
        output = s * k / lam * inf_check_1 * torch_exp(-inf_check_2)

        # output = s * k / lam * (x_calc / lam) ** (k - 1) * torch_exp(-(x_calc / lam) ** k)
        output = output.masked_fill(x_pad_mask, 0)
        return output

    @staticmethod
    def integral(x, params):
        # f(x) = s * (1 - exp(-(x / lam) ^ k))
        # assert (x >= 0).all()
        _params = F_softplus(params)
        s = _params[..., 0]
        k = _params[..., 1]
        lam = _params[..., 2] + 1e-10
        # x_pad_mask = x >= 1e10
        # x_calc = x.masked_fill(x_pad_mask, 100) + 1e-20
        x_calc = x + 1e-4
        zero_mask = x == 0

        x_calc_div_lam = x_calc / lam
        inf_check = x_calc_div_lam ** k
        inf_check_inf_mask = torch.isinf(inf_check * torch.log(x_calc_div_lam))
        if inf_check_inf_mask.any():
            x_calc_div_lam_fake = x_calc_div_lam.masked_fill(inf_check_inf_mask, 1)
            inf_check = x_calc_div_lam_fake ** k
            inf_check = inf_check.masked_fill(inf_check_inf_mask, 1e10)
        output = s * (torch_exp(-(1e-4 / lam) ** k) - torch_exp(-inf_check))

        # output = s * (torch_exp(-(1e-4 / lam) ** k) - torch_exp(-(x_calc / lam) ** k))
        output = output.masked_fill(zero_mask, 0)
        # output = output.masked_fill(x_pad_mask, 10)
        return output

    @staticmethod
    def integral_inv(y, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        k = _params[..., 1]
        lam = _params[..., 2] + 1e-10
        log_input = ((s * torch_exp(-(1e-4 / lam) ** k) - y) / s)
        nan_mask = log_input <= 0
        log_input_fake = log_input.masked_fill(nan_mask, 1)
        output_fake = lam * torch_pow(- torch_log(log_input_fake), 1 / k) - 1e-4
        return output_fake.masked_fill(nan_mask, float('inf'))
        # return lam * torch_pow(- torch_log((s * torch_exp(-(1e-4 / lam) ** k) - y) / s), 1 / k) - 1e-4

class Gompertz:
    def __init__(self) -> None:
        pass

    @staticmethod
    def forward(x, params):
        # f(x) = s * b * eta * exp(eta + b * x - eta * exp(b * x))
        _params = F_softplus(params)
        s = _params[..., 0]
        b = _params[..., 1] + 1e-10
        eta = _params[..., 2]
        inner_exp_input = (b * x).clamp(max=70)
        return s * b * eta * torch_exp(eta + b * x - eta * torch_exp(inner_exp_input))
        # return s * b * eta * torch_exp(eta + b * x - eta * torch_exp(b * x))

    @staticmethod
    def integral(x, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        b = _params[..., 1] + 1e-10
        eta = _params[..., 2]
        inner_exp_input = (b * x).clamp(max=70)
        return s * (1 - torch_exp(-eta * (torch_exp(inner_exp_input) - 1)))
        # return s * (1 - torch_exp(-eta * (torch_exp(b * x) - 1)))

    @staticmethod
    def integral_inv(y, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        b = _params[..., 1] + 1e-10
        eta = _params[..., 2]
        log_input = (s - y) / s
        nan_mask = log_input <= 0
        log_input_fake = log_input.masked_fill(nan_mask, 1)
        output_fake = torch_log(1 - torch_log(log_input_fake) / eta) / b
        return output_fake.masked_fill(nan_mask, float('inf'))
        # return torch_log(1 - torch_log((s - y) / s) / eta) / b

class GompertzShift:
    def __init__(self) -> None:
        pass

    @staticmethod
    def forward(x, params):
        # f(x) = s * b * eta * exp(eta + b * (x + a) - eta * exp(b * (x + a)))
        _params = F_softplus(params)
        s = _params[0]
        a = _params[1]
        b = _params[2]
        eta = _params[3]
        return s * b * eta * torch_exp(eta + b * (x + a) - eta * torch_exp(b * (x + a)))

    @staticmethod
    def integral(x, params):
        _params = F_softplus(params)
        s = _params[0]
        a = _params[1]
        b = _params[2]
        eta = _params[3]
        return s * (torch_exp(-eta * (torch_exp(b * a) - 1)) - torch_exp(-eta * (torch_exp(b * (a + x)) - 1)))

    @staticmethod
    def integral_inv(y, params):
        _params = F_softplus(params)
        s = _params[0]
        a = _params[1]
        b = _params[2]
        eta = _params[3]
        return torch_log(1 - torch_log((s * torch_exp(-eta * (torch_exp(b * a) - 1)) - y) / s) / eta) / b - a

class Exponential:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def forward(x, params):
        # f(x) = s * b * exp( - b * x )
        _params = F_softplus(params)
        s = _params[..., 0]
        b = _params[..., 1] + 1e-10
        return s * b * torch.exp(-b * x)

    @staticmethod
    def integral(x, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        b = _params[..., 1] + 1e-10
        return s * (1 - torch.exp(-b * x))

    @staticmethod
    def integral_inv(y, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        b = _params[..., 1]
        # b = b.masked_fill(b == 0, 1e-20)
        b = b + 1e-10
        log_input = (1 - y / s)
        nan_mask = log_input <= 0
        log_input_fake = log_input.masked_fill(nan_mask, 1)
        output_fake = - 1 / b * torch.log(log_input_fake)
        return output_fake.masked_fill(nan_mask, float('inf'))
        # return - 1 / b * torch.log(1 - y / s)

class Normal:
    def __init__(self) -> None:
        pass

    @staticmethod
    def forward(x, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        mu = params[..., 1]
        sigma = _params[..., 2] + 1e-10
        return s / (sigma * math.sqrt(2 * math.pi)) * torch_exp(- 0.5 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def integral(x, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        mu = params[..., 1]
        sigma = _params[..., 2] + 1e-10
        zero_mask = x == 0
        output = s / 2 * (torch.erf((x - mu) / sigma / math.sqrt(2)) - torch.erf((-mu) / sigma / math.sqrt(2)))
        return output.masked_fill(zero_mask, 0)

    @staticmethod
    def integral_inv(y, params):
        _params = F_softplus(params)
        s = _params[..., 0]
        mu = params[..., 1]
        sigma = _params[..., 2] + 1e-10
        return mu + math.sqrt(2) * sigma * torch.erfinv(2 * y / s + torch.erf((-mu) / sigma / math.sqrt(2)))