import torch
import torch.nn as nn
import torchvision


# def ema_update(origin_model_para, new_model_para, moment=0.995):
#     assert origin_model_para.keys() == new_model_para.keys()
#     avg_model_para = nn.ModuleDict()
#     for k, v in origin_model_para.items():
#         avg_model_para[k] = v * moment + new_model_para[k] * (1 - moment)
#
#     return avg_model_para

def ema_update(model_T, model_S, moment=0.995):
    for mean_param, param in zip(model_T.parameters(), model_S.parameters()):
        mean_param.data.mul_(moment).add_(1 - moment, param.data)
