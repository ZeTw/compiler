import os
import random

import torch
from complier_utils.utils_base import focus
import numpy as np

csp6_scale = np.load('para/head.csp6.scale.npy')
print(csp6_scale)
csp6_zp = np.load('para/head.csp6.zero_point.npy')
print(csp6_zp)


# focus_csp_scale = np.load('para/backbone.backbone.stem.csp0.scale.npy')
# focus_csp_zp = np.load('para/backbone.backbone.stem.csp0.zero_point.npy')
# print(focus_csp_scale)
# print(focus_csp_zp)
# focus_conv_weight_scale = np.load('para/backbone.backbone.stem.conv.conv.weight.scale.npy')
# focus_conv_weight_zp = np.load('para/backbone.backbone.stem.conv.conv.weight.zero_point.npy')
# focus_conv_act_scale = np.load('para/backbone.backbone.stem.conv.conv.scale.npy')
# focus_conv_act_zp = np.load('para/backbone.backbone.stem.conv.conv.zero_point.npy')
#
# bias = np.load('para/backbone.backbone.stem.conv.conv.bias.npy')
# # print(focus_csp_scale, focus_csp_zp, focus_conv_weight_scale, focus_conv_weight_zp, focus_conv_act_scale,
# #       focus_conv_act_zp)
# print(bias)
# test_tensor = torch.arange(1, 65, dtype=torch.uint8)
# final_tensor = torch.reshape(test_tensor, [1, 1, 8, 8])
# print(final_tensor)
#
# a, b, c, d = focus(final_tensor)
#
# print(a)
# print(b)
# print(c)
# print(d)
# e = torch.cat([a, b, c, d], dim=1)
#
# print(e)
#
# a = 0b011111111111111111111111110001001
# print(a)
# c = a & 0b1
# if c == 0b1:
#     a += 0b1
# print(a)
# a = np.array([1], dtype=np.int32)
# M = np.array([0.5 * (2 ** 16)],dtype=np.float32)
# print(M.dtype)
# a = a *M
# print(a.dtype)

# file_path = 'model_structure/layer_name.txt'
# name_file = open(file_path)
# name_lines = name_file.readlines()
# name_dict = {}
#
# for index, name_line in enumerate(name_lines):
#     name_dict[index+1] = name_line.strip()
#
# print(name_dict)
