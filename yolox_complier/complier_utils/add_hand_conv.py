# -*- coding: UTF-8 -*-
# encoding=utf-8
import torch
import torch.nn as nn
import numpy as np

from complier_utils.utils_bbox import decode_outputs
from test_add2 import *

torch.set_printoptions(precision='full')  # torch打印18位
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys

sys.path.append("..")
import torch.quantization
from torch.nn.quantized import functional as qF
from utils_base import gen_coe, np2tensor, quant_cat, quant_add, reg_add
from picture_load import picture_load, focus, detect_img
from test_add2 import conv_naive_Simulation_speed, conv_naive_relu


class Conv2d_Q(nn.Module):
    def __init__(
            self,
            quant_scale1=None,
            quant_zero_point1=None,
            quant_scale2=None,
            quant_zero_point2=None,
            quant_scale3=None,
            quant_zero_point3=None,
            coe_name=None,
            operator=None,

    ):
        super(Conv2d_Q, self).__init__()
        self.quant_scale1 = quant_scale1
        self.quant_zero_point1 = quant_zero_point1
        self.quant_scale2 = quant_scale2
        self.quant_zero_point2 = quant_zero_point2
        self.quant_scale3 = quant_scale3
        self.quant_zero_point3 = quant_zero_point3
        self.coe_name = coe_name
        self.operator = operator

    def forward(self, q_feature, q_weight, bias, out_hand=1, write_coe_file=0, stride=2,
                padding=1, block=0):
        '''


        :param q_feature:  需要进入该层的量化后的输入
        :param q_weight:  该层量化后的权重
        :param bias:      bis
        :param out_hand:  是否输出
        :param write_coe_file: 是否写入coe文件
        :param stride: 该层卷积的stride
        :param padding: 该层卷积的padding
        :param block: 是否分块
        :return: 返回该层计算之后的feature map

        当out_hand 和 write_coe_file 为1的时候才会写入coe中
        '''

        # global feature_result

        if out_hand:
            if 'conv' == self.operator:
                feature_result = conv_naive_Simulation_speed(q_feature, q_weight, padding=padding, compare=0,
                                                             stride=stride,
                                                             bias=bias,
                                                             s1=self.quant_scale1,
                                                             s2=self.quant_scale2, s3=self.quant_scale3,
                                                             z1=int(self.quant_zero_point1),
                                                             z2=self.quant_zero_point2, z3=self.quant_zero_point3)

                if write_coe_file:
                    gen_coe(self.coe_name, feature_result)
                return feature_result
            elif 'conv2d' == self.operator:
                feature_result = conv_naive_relu(q_feature, q_weight, padding=padding, compare=0, stride=stride,
                                                 bias=bias,
                                                 s1=self.quant_scale1,
                                                 s2=self.quant_scale2, s3=self.quant_scale3,
                                                 z1=int(self.quant_zero_point1),
                                                 z2=self.quant_zero_point2, z3=self.quant_zero_point3)
                z3_conv11 = self.quant_zero_point3.numpy()
                shape = feature_result.shape
                # 将通道补为8的倍数
                new_shape_kenel = shape[1] + 8 - shape[1] % 8
                xxx = np.ones((shape[0], new_shape_kenel, shape[2], shape[3]),
                              dtype=np.uint8) * z3_conv11

                xxx[:, :shape[1], :, :] = feature_result
                feature_result = xxx
                feature_result = torch.tensor(feature_result, dtype=torch.uint8)

                if write_coe_file:
                    gen_coe(self.coe_name, feature_result)

                return feature_result


class QuantizableYolox(nn.Module):
    def __init__(self, img_path, show_result):
        super(QuantizableYolox, self).__init__()
        self.img_path = img_path
        self.show_result = show_result
        self.csp = torch.nn.quantized.FloatFunctional()
        # ==============start stem======================
        quant_scale = np2tensor('../para/quant.scale.npy')
        quant_zp = np2tensor('../para/quant.zero_point.npy')
        focus_conv_weight_scale = np2tensor('../para/backbone.backbone.stem.conv.conv.weight.scale.npy')
        focus_conv_weight_zp = np2tensor('../para/backbone.backbone.stem.conv.conv.weight.zero_point.npy')
        focus_act_scale = np2tensor('../para/backbone.backbone.stem.conv.conv.scale.npy')
        focus_act_zp = np2tensor('../para/backbone.backbone.stem.conv.conv.zero_point.npy')
        focus_conv_coe_name = '../hand_coe/focus_conv.coe'
        self.focus_conv = Conv2d_Q(quant_scale1=quant_scale, quant_zero_point1=quant_zp,
                                   quant_scale2=focus_conv_weight_scale,
                                   quant_zero_point2=focus_conv_weight_zp,
                                   quant_scale3=focus_act_scale, quant_zero_point3=focus_act_zp,
                                   coe_name=focus_conv_coe_name, operator='conv')

        # ==================start dark2====================
        # ======= BaseConv============

        dark2_BaseConv_weight_scale = np2tensor('../para/backbone.backbone.dark2.0.conv.weight.scale.npy')
        dark2_BaseConv_weight_zp = np2tensor('../para/backbone.backbone.dark2.0.conv.weight.zero_point.npy')
        dark2_BaseConv_act_scale = np2tensor('../para/backbone.backbone.dark2.0.conv.scale.npy')
        dark2_BaseConv_act_zp = np2tensor('../para/backbone.backbone.dark2.0.conv.zero_point.npy')
        dark2_BaseConv_coe_name = '../hand_coe/dark2_BaseConv.coe'
        self.dark2_baseconv = Conv2d_Q(quant_scale1=focus_act_scale, quant_zero_point1=focus_act_zp,
                                       quant_scale2=dark2_BaseConv_weight_scale,
                                       quant_zero_point2=dark2_BaseConv_weight_zp,
                                       quant_scale3=dark2_BaseConv_act_scale, quant_zero_point3=dark2_BaseConv_act_zp,
                                       coe_name=dark2_BaseConv_coe_name, operator='conv')
        # ========== CSPLayer ============
        dark2_csp_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.weight.scale.npy')
        dark2_csp_conv1_weight_zp = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.weight.zero_point.npy')
        dark2_csp_conv1_act_scale = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.scale.npy')
        dark2_csp_conv1_act_zp = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.zero_point.npy')

        dark2_csp_conv1_coe_name = '../hand_coe/dark2_CSP_Conv1.coe'
        self.dark2_csp_conv1 = Conv2d_Q(quant_scale1=dark2_BaseConv_act_scale, quant_zero_point1=dark2_BaseConv_act_zp,
                                        quant_scale2=dark2_csp_conv1_weight_scale,
                                        quant_zero_point2=dark2_csp_conv1_weight_zp,
                                        quant_scale3=dark2_csp_conv1_act_scale,
                                        quant_zero_point3=dark2_csp_conv1_act_zp, coe_name=dark2_csp_conv1_coe_name,
                                        operator='conv')

        dark2_csp_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.weight.scale.npy')
        dark2_csp_conv2_weight_zp = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.weight.zero_point.npy')
        dark2_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.scale.npy')
        dark2_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.zero_point.npy')
        dark2_csp_conv2_coe_name = '../hand_coe/dark2_CSP_Conv2.coe'
        self.dark2_csp_conv2 = Conv2d_Q(quant_scale1=dark2_BaseConv_act_scale,
                                        quant_zero_point1=dark2_BaseConv_act_zp,
                                        quant_scale2=dark2_csp_conv2_weight_scale,
                                        quant_zero_point2=dark2_csp_conv2_weight_zp,
                                        quant_scale3=dark2_csp_conv2_act_scale,
                                        quant_zero_point3=dark2_csp_conv2_act_zp, coe_name=dark2_csp_conv2_coe_name,
                                        operator='conv')

        # ========== dark2_csp_m =============
        # 进入csp_m的为csp_conv1的结果 则对应的scale1 和 zp1也应该是做完csp_conv1之后激活的scale和zp
        dark2_csp_m_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv1.conv.weight.scale.npy')
        dark2_csp_m_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark2.1.m.0.conv1.conv.weight.zero_point.npy')
        dark2_csp_m_conv1_act_scale = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv1.conv.scale.npy')
        dark2_csp_m_conv1_act_zp = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv1.conv.zero_point.npy')
        dark2_csp_m_conv1_coe_name = '../hand_coe/dark2_CSP_m_Conv1.coe'
        self.dark2_csp_m_conv1 = Conv2d_Q(quant_scale1=dark2_csp_conv1_act_scale,
                                          quant_zero_point1=dark2_csp_conv1_act_zp,
                                          quant_scale2=dark2_csp_m_conv1_weight_scale,
                                          quant_zero_point2=dark2_csp_m_conv1_weight_zp,
                                          quant_scale3=dark2_csp_m_conv1_act_scale,
                                          quant_zero_point3=dark2_csp_m_conv1_act_zp,
                                          coe_name=dark2_csp_m_conv1_coe_name,
                                          operator='conv')

        dark2_csp_m_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv2.conv.weight.scale.npy')
        dark2_csp_m_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark2.1.m.0.conv2.conv.weight.zero_point.npy')
        dark2_csp_m_conv2_act_scale = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv2.conv.scale.npy')
        dark2_csp_m_conv2_act_zp = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv2.conv.zero_point.npy')
        dark2_csp_m_conv2_coe_name = '../hand_coe/dark2_CSP_m_Conv2.coe'
        self.dark2_csp_m_conv2 = Conv2d_Q(quant_scale1=dark2_csp_m_conv1_act_scale,
                                          quant_zero_point1=dark2_csp_m_conv1_act_zp,
                                          quant_scale2=dark2_csp_m_conv2_weight_scale,
                                          quant_zero_point2=dark2_csp_m_conv2_weight_zp,
                                          quant_scale3=dark2_csp_m_conv2_act_scale,
                                          quant_zero_point3=dark2_csp_m_conv2_act_zp,
                                          coe_name=dark2_csp_m_conv2_coe_name, operator='conv')

        # ============ dark2 csp cat ==========
        dark2_csp_cat_scale = np2tensor('../para/backbone.backbone.dark2.1.csp1.scale.npy')
        dark2_csp_cat_zp = np2tensor('../para/backbone.backbone.dark2.1.csp1.zero_point.npy')

        # ============ dark2 conv3 ==========
        dark2_csp_conv3_weight_scale = np2tensor('../para/backbone.backbone.dark2.1.conv3.conv.weight.scale.npy')
        dark2_csp_conv3_weight_zp = np2tensor('../para/backbone.backbone.dark2.1.conv3.conv.weight.zero_point.npy')
        dark2_csp_conv3_act_scale = np2tensor('../para/backbone.backbone.dark2.1.conv3.conv.scale.npy')
        dark2_csp_conv3_act_zp = np2tensor('../para/backbone.backbone.dark2.1.conv3.conv.zero_point.npy')

        dark2_csp_conv3_coe_name = '../hand_coe/dark2_CSP_Conv3.coe'

        self.dark2_csp_conv3 = Conv2d_Q(quant_scale1=dark2_csp_cat_scale, quant_zero_point1=dark2_csp_cat_zp,
                                        quant_scale2=dark2_csp_conv3_weight_scale,
                                        quant_zero_point2=dark2_csp_conv3_weight_zp,
                                        quant_scale3=dark2_csp_conv3_act_scale,
                                        quant_zero_point3=dark2_csp_conv3_act_zp,
                                        coe_name=dark2_csp_conv3_coe_name, operator='conv')
        # ==================start dark3====================
        # ======= BaseConv============
        dark3_BaseConv_weight_scale = np2tensor('../para/backbone.backbone.dark3.0.conv.weight.scale.npy')
        dark3_BaseConv_weight_zp = np2tensor('../para/backbone.backbone.dark3.0.conv.weight.zero_point.npy')
        dark3_BaseConv_act_scale = np2tensor('../para/backbone.backbone.dark3.0.conv.scale.npy')
        dark3_BaseConv_act_zp = np2tensor('../para/backbone.backbone.dark3.0.conv.zero_point.npy')

        dark3_BaseConv_coe_name = '../hand_coe/dark3_BaseConv.coe'
        self.dark3_baseconv = Conv2d_Q(quant_scale1=dark2_csp_conv3_act_scale,
                                       quant_zero_point1=dark2_csp_conv3_act_zp,
                                       quant_scale2=dark3_BaseConv_weight_scale,
                                       quant_zero_point2=dark3_BaseConv_weight_zp,
                                       quant_scale3=dark3_BaseConv_act_scale,
                                       quant_zero_point3=dark3_BaseConv_act_zp,
                                       coe_name=dark3_BaseConv_coe_name, operator='conv')

        # ========== CSPLayer ============
        dark3_csp_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.weight.scale.npy')
        dark3_csp_conv1_weight_zp = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.weight.zero_point.npy')
        dark3_csp_conv1_act_scale = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.scale.npy')
        dark3_csp_conv1_act_zp = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.zero_point.npy')
        dark3_csp_conv1_coe_name = '../hand_coe/dark3_CSP_Conv1.coe'
        self.dark3_csp_conv1 = Conv2d_Q(quant_scale1=dark3_BaseConv_act_scale,
                                        quant_zero_point1=dark3_BaseConv_act_zp,
                                        quant_scale2=dark3_csp_conv1_weight_scale,
                                        quant_zero_point2=dark3_csp_conv1_weight_zp,
                                        quant_scale3=dark3_csp_conv1_act_scale,
                                        quant_zero_point3=dark3_csp_conv1_act_zp, coe_name=dark3_csp_conv1_coe_name,
                                        operator='conv')

        dark3_csp_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.weight.scale.npy')
        dark3_csp_conv2_weight_zp = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.weight.zero_point.npy')
        dark3_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.scale.npy')
        dark3_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.zero_point.npy')
        dark3_csp_conv2_coe_name = '../hand_coe/dark3_CSP_Conv2.coe'
        self.dark3_csp_conv2 = Conv2d_Q(quant_scale1=dark3_BaseConv_act_scale,
                                        quant_zero_point1=dark3_BaseConv_act_zp,
                                        quant_scale2=dark3_csp_conv2_weight_scale,
                                        quant_zero_point2=dark3_csp_conv2_weight_zp,
                                        quant_scale3=dark3_csp_conv2_act_scale,
                                        quant_zero_point3=dark3_csp_conv2_act_zp, coe_name=dark3_csp_conv2_coe_name,
                                        operator='conv')

        # ========== dark3_csp_m =============
        # m0
        dark3_csp_m0_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv1.conv.weight.scale.npy')
        dark3_csp_m0_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark3.1.m.0.conv1.conv.weight.zero_point.npy')
        dark3_csp_m0_conv1_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv1.conv.scale.npy')
        dark3_csp_m0_conv1_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv1.conv.zero_point.npy')
        dark3_csp_m0_conv1_coe_name = '../hand_coe/dark3_CSP_m0_Conv1.coe'
        self.dark3_csp_m0_conv1 = Conv2d_Q(quant_scale1=dark3_csp_conv1_act_scale,
                                           quant_zero_point1=dark3_csp_conv1_act_zp,
                                           quant_scale2=dark3_csp_m0_conv1_weight_scale,
                                           quant_zero_point2=dark3_csp_m0_conv1_weight_zp,
                                           quant_scale3=dark3_csp_m0_conv1_act_scale,
                                           quant_zero_point3=dark3_csp_m0_conv1_act_zp,
                                           coe_name=dark3_csp_m0_conv1_coe_name,
                                           operator='conv')

        dark3_csp_m0_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv2.conv.weight.scale.npy')
        dark3_csp_m0_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark3.1.m.0.conv2.conv.weight.zero_point.npy')
        dark3_csp_m0_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv2.conv.scale.npy')
        dark3_csp_m0_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv2.conv.zero_point.npy')

        dark3_csp_m0_conv2_coe_name = '../hand_coe/dark3_CSP_m0_Conv2.coe'
        self.dark3_csp_m0_conv2 = Conv2d_Q(quant_scale1=dark3_csp_m0_conv1_act_scale,
                                           quant_zero_point1=dark3_csp_m0_conv1_act_zp,
                                           quant_scale2=dark3_csp_m0_conv2_weight_scale,
                                           quant_zero_point2=dark3_csp_m0_conv2_weight_zp,
                                           quant_scale3=dark3_csp_m0_conv2_act_scale,
                                           quant_zero_point3=dark3_csp_m0_conv2_act_zp,
                                           coe_name=dark3_csp_m0_conv2_coe_name,
                                           operator='conv')
        dark3_csp_add0_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.csp.scale.npy')
        dark3_csp_add0_zp = np2tensor('../para/backbone.backbone.dark3.1.m.0.csp.zero_point.npy')

        # m1
        dark3_csp_m1_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv1.conv.weight.scale.npy')
        dark3_csp_m1_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark3.1.m.1.conv1.conv.weight.zero_point.npy')
        dark3_csp_m1_conv1_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv1.conv.scale.npy')
        dark3_csp_m1_conv1_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv1.conv.zero_point.npy')
        dark3_csp_m1_conv1_coe_name = '../hand_coe/dark3_CSP_m1_Conv1.coe'
        self.dark3_csp_m1_conv1 = Conv2d_Q(quant_scale1=dark3_csp_add0_scale, quant_zero_point1=dark3_csp_add0_zp,
                                           quant_scale2=dark3_csp_m1_conv1_weight_scale,
                                           quant_zero_point2=dark3_csp_m1_conv1_weight_zp,
                                           quant_scale3=dark3_csp_m1_conv1_act_scale,
                                           quant_zero_point3=dark3_csp_m1_conv1_act_zp,
                                           coe_name=dark3_csp_m1_conv1_coe_name,
                                           operator='conv')
        dark3_csp_m1_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv2.conv.weight.scale.npy')
        dark3_csp_m1_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark3.1.m.1.conv2.conv.weight.zero_point.npy')
        dark3_csp_m1_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv2.conv.scale.npy')
        dark3_csp_m1_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv2.conv.zero_point.npy')
        dark3_csp_m1_conv2_coe_name = '../hand_coe/dark3_CSP_m1_Conv2.coe'
        self.dark3_csp_m1_conv2 = Conv2d_Q(quant_scale1=dark3_csp_m1_conv1_act_scale,
                                           quant_zero_point1=dark3_csp_m1_conv1_act_zp,
                                           quant_scale2=dark3_csp_m1_conv2_weight_scale,
                                           quant_zero_point2=dark3_csp_m1_conv2_weight_zp,
                                           quant_scale3=dark3_csp_m1_conv2_act_scale,
                                           quant_zero_point3=dark3_csp_m1_conv2_act_zp,
                                           coe_name=dark3_csp_m1_conv2_coe_name,
                                           operator='conv')
        dark3_csp_add1_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.csp.scale.npy')
        dark3_csp_add1_zp = np2tensor('../para/backbone.backbone.dark3.1.m.1.csp.zero_point.npy')

        # m2 conv1
        dark3_csp_m2_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv1.conv.weight.scale.npy')
        dark3_csp_m2_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark3.1.m.2.conv1.conv.weight.zero_point.npy')
        dark3_csp_m2_conv1_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv1.conv.scale.npy')
        dark3_csp_m2_conv1_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv1.conv.zero_point.npy')
        dark3_csp_m2_conv1_coe_name = '../hand_coe/dark3_CSP_m2_Conv1.coe'
        self.dark3_csp_m2_conv1 = Conv2d_Q(quant_scale1=dark3_csp_add1_scale, quant_zero_point1=dark3_csp_add1_zp,
                                           quant_scale2=dark3_csp_m2_conv1_weight_scale,
                                           quant_zero_point2=dark3_csp_m2_conv1_weight_zp,
                                           quant_scale3=dark3_csp_m2_conv1_act_scale,
                                           quant_zero_point3=dark3_csp_m2_conv1_act_zp,
                                           coe_name=dark3_csp_m2_conv1_coe_name,
                                           operator='conv')
        # m2 conv2
        dark3_csp_m2_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv2.conv.weight.scale.npy')
        dark3_csp_m2_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark3.1.m.2.conv2.conv.weight.zero_point.npy')
        dark3_csp_m2_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv2.conv.scale.npy')
        dark3_csp_m2_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv2.conv.zero_point.npy')

        dark3_csp_m2_conv2_coe_name = '../hand_coe/dark3_CSP_m2_Conv2.coe'
        self.dark3_csp_m2_conv2 = Conv2d_Q(quant_scale1=dark3_csp_m2_conv1_act_scale,
                                           quant_zero_point1=dark3_csp_m2_conv1_act_zp,
                                           quant_scale2=dark3_csp_m2_conv2_weight_scale,
                                           quant_zero_point2=dark3_csp_m2_conv2_weight_zp,
                                           quant_scale3=dark3_csp_m2_conv2_act_scale,
                                           quant_zero_point3=dark3_csp_m2_conv2_act_zp,
                                           coe_name=dark3_csp_m2_conv2_coe_name,
                                           operator='conv')

        dark3_csp_add2_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.csp.scale.npy')
        dark3_csp_add2_zp = np2tensor('../para/backbone.backbone.dark3.1.m.2.csp.zero_point.npy')

        # ============ dark3 csp cat ==========
        dark3_csp_cat_scale = np2tensor('../para/backbone.backbone.dark3.1.csp1.scale.npy')
        dark3_csp_cat_zp = np2tensor('../para/backbone.backbone.dark3.1.csp1.zero_point.npy')

        # ============ dark3 conv3 ==========
        dark3_csp_conv3_weight_scale = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.weight.scale.npy')
        dark3_csp_conv3_weight_zp = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.weight.zero_point.npy')
        dark3_csp_conv3_act_scale = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.scale.npy')
        dark3_csp_conv3_act_zp = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.zero_point.npy')
        dark3_csp_conv3_coe_name = '../hand_coe/dark3_CSP_Conv3.coe'
        self.dark3_csp_conv3 = Conv2d_Q(quant_scale1=dark3_csp_cat_scale, quant_zero_point1=dark3_csp_cat_zp,
                                        quant_scale2=dark3_csp_conv3_weight_scale,
                                        quant_zero_point2=dark3_csp_conv3_weight_zp,
                                        quant_scale3=dark3_csp_conv3_act_scale,
                                        quant_zero_point3=dark3_csp_conv3_act_zp,
                                        coe_name=dark3_csp_conv3_coe_name,
                                        operator='conv')

        '''
         dark4 开始
        '''
        # ==================start dark4====================
        # ======= BaseConv============
        dark4_BaseConv_weight_scale = np2tensor('../para/backbone.backbone.dark4.0.conv.weight.scale.npy')
        dark4_BaseConv_weight_zp = np2tensor('../para/backbone.backbone.dark4.0.conv.weight.zero_point.npy')
        dark4_BaseConv_act_scale = np2tensor('../para/backbone.backbone.dark4.0.conv.scale.npy')
        dark4_BaseConv_act_zp = np2tensor('../para/backbone.backbone.dark4.0.conv.zero_point.npy')

        dark4_BaseConv_coe_name = '../hand_coe/dark4_BaseConv.coe'
        self.dark4_baseconv = Conv2d_Q(quant_scale1=dark3_csp_conv3_act_scale, quant_zero_point1=dark3_csp_conv3_act_zp,
                                       quant_scale2=dark4_BaseConv_weight_scale,
                                       quant_zero_point2=dark4_BaseConv_weight_zp,
                                       quant_scale3=dark4_BaseConv_act_scale, quant_zero_point3=dark4_BaseConv_act_zp,
                                       coe_name=dark4_BaseConv_coe_name, operator='conv')

        # ========== CSPLayer ============
        dark4_csp_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.weight.scale.npy')
        dark4_csp_conv1_weight_zp = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.weight.zero_point.npy')
        dark4_csp_conv1_act_scale = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.scale.npy')
        dark4_csp_conv1_act_zp = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.zero_point.npy')
        dark4_csp_conv1_coe_name = '../hand_coe/dark4_CSP_Conv1.coe'
        self.dark4_csp_conv1 = Conv2d_Q(quant_scale1=dark4_BaseConv_act_scale, quant_zero_point1=dark4_BaseConv_act_zp,
                                        quant_scale2=dark4_csp_conv1_weight_scale,
                                        quant_zero_point2=dark4_csp_conv1_weight_zp,
                                        quant_scale3=dark4_csp_conv1_act_scale,
                                        quant_zero_point3=dark4_csp_conv1_act_zp, coe_name=dark4_csp_conv1_coe_name,
                                        operator='conv')

        dark4_csp_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.weight.scale.npy')
        dark4_csp_conv2_weight_zp = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.weight.zero_point.npy')
        dark4_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.scale.npy')
        dark4_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.zero_point.npy')
        dark4_csp_conv2_coe_name = '../hand_coe/dark4_CSP_Conv2.coe'
        self.dark4_csp_conv2 = Conv2d_Q(quant_scale1=dark4_BaseConv_act_scale, quant_zero_point1=dark4_BaseConv_act_zp,
                                        quant_scale2=dark4_csp_conv2_weight_scale,
                                        quant_zero_point2=dark4_csp_conv2_weight_zp,
                                        quant_scale3=dark4_csp_conv2_act_scale,
                                        quant_zero_point3=dark4_csp_conv2_act_zp, coe_name=dark4_csp_conv2_coe_name,
                                        operator='conv')

        # ========== dark4_csp_m =============
        # m0
        dark4_csp_m0_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv1.conv.weight.scale.npy')
        dark4_csp_m0_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark4.1.m.0.conv1.conv.weight.zero_point.npy')
        dark4_csp_m0_conv1_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv1.conv.scale.npy')
        dark4_csp_m0_conv1_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv1.conv.zero_point.npy')
        dark4_csp_m0_conv1_coe_name = '../hand_coe/dark4_CSP_m0_Conv1.coe'
        self.dark4_csp_m0_conv1 = Conv2d_Q(quant_scale1=dark4_csp_conv1_act_scale,
                                           quant_zero_point1=dark4_csp_conv1_act_zp,
                                           quant_scale2=dark4_csp_m0_conv1_weight_scale,
                                           quant_zero_point2=dark4_csp_m0_conv1_weight_zp,
                                           quant_scale3=dark4_csp_m0_conv1_act_scale,
                                           quant_zero_point3=dark4_csp_m0_conv1_act_zp,
                                           coe_name=dark4_csp_m0_conv1_coe_name,
                                           operator='conv')

        dark4_csp_m0_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv2.conv.weight.scale.npy')
        dark4_csp_m0_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark4.1.m.0.conv2.conv.weight.zero_point.npy')
        dark4_csp_m0_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv2.conv.scale.npy')
        dark4_csp_m0_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv2.conv.zero_point.npy')
        dark4_csp_m0_conv2_coe_name = '../hand_coe/dark4_CSP_m0_Conv2.coe'
        self.dark4_csp_m0_conv2 = Conv2d_Q(quant_scale1=dark4_csp_m0_conv1_act_scale,
                                           quant_zero_point1=dark4_csp_m0_conv1_act_zp,
                                           quant_scale2=dark4_csp_m0_conv2_weight_scale,
                                           quant_zero_point2=dark4_csp_m0_conv2_weight_zp,
                                           quant_scale3=dark4_csp_m0_conv2_act_scale,
                                           quant_zero_point3=dark4_csp_m0_conv2_act_zp,
                                           coe_name=dark4_csp_m0_conv2_coe_name,
                                           operator='conv')
        dark4_csp_add0_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.csp.scale.npy')
        dark4_csp_add0_zp = np2tensor('../para/backbone.backbone.dark4.1.m.0.csp.zero_point.npy')

        # m1
        dark4_csp_m1_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv1.conv.weight.scale.npy')
        dark4_csp_m1_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark4.1.m.1.conv1.conv.weight.zero_point.npy')
        dark4_csp_m1_conv1_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv1.conv.scale.npy')
        dark4_csp_m1_conv1_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv1.conv.zero_point.npy')
        dark4_csp_m1_conv1_coe_name = '../hand_coe/dark4_CSP_m1_Conv1.coe'
        self.dark4_csp_m1_conv1 = Conv2d_Q(quant_scale1=dark4_csp_add0_scale, quant_zero_point1=dark4_csp_add0_zp,
                                           quant_scale2=dark4_csp_m1_conv1_weight_scale,
                                           quant_zero_point2=dark4_csp_m1_conv1_weight_zp,
                                           quant_scale3=dark4_csp_m1_conv1_act_scale,
                                           quant_zero_point3=dark4_csp_m1_conv1_act_zp,
                                           coe_name=dark4_csp_m1_conv1_coe_name,
                                           operator='conv')
        dark4_csp_m1_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv2.conv.weight.scale.npy')
        dark4_csp_m1_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark4.1.m.1.conv2.conv.weight.zero_point.npy')
        dark4_csp_m1_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv2.conv.scale.npy')
        dark4_csp_m1_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv2.conv.zero_point.npy')
        dark4_csp_m1_conv2_coe_name = '../hand_coe/dark4_CSP_m1_Conv2.coe'
        self.dark4_csp_m1_conv2 = Conv2d_Q(quant_scale1=dark4_csp_m1_conv1_act_scale,
                                           quant_zero_point1=dark4_csp_m1_conv1_act_zp,
                                           quant_scale2=dark4_csp_m1_conv2_weight_scale,
                                           quant_zero_point2=dark4_csp_m1_conv2_weight_zp,
                                           quant_scale3=dark4_csp_m1_conv2_act_scale,
                                           quant_zero_point3=dark4_csp_m1_conv2_act_zp,
                                           coe_name=dark4_csp_m1_conv2_coe_name,
                                           operator='conv')
        dark4_csp_add1_scale = np2tensor('../para/backbone.backbone.dark4.1.m.1.csp.scale.npy')
        dark4_csp_add1_zp = np2tensor('../para/backbone.backbone.dark4.1.m.1.csp.zero_point.npy')

        # m2 conv1
        dark4_csp_m2_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv1.conv.weight.scale.npy')
        dark4_csp_m2_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark4.1.m.2.conv1.conv.weight.zero_point.npy')
        dark4_csp_m2_conv1_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv1.conv.scale.npy')
        dark4_csp_m2_conv1_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv1.conv.zero_point.npy')
        dark4_csp_m2_conv1_coe_name = '../hand_coe/dark4_CSP_m2_Conv1.coe'
        self.dark4_csp_m2_conv1 = Conv2d_Q(quant_scale1=dark4_csp_add1_scale, quant_zero_point1=dark4_csp_add1_zp,
                                           quant_scale2=dark4_csp_m2_conv1_weight_scale,
                                           quant_zero_point2=dark4_csp_m2_conv1_weight_zp,
                                           quant_scale3=dark4_csp_m2_conv1_act_scale,
                                           quant_zero_point3=dark4_csp_m2_conv1_act_zp,
                                           coe_name=dark4_csp_m2_conv1_coe_name,
                                           operator='conv')
        # m2 conv2
        dark4_csp_m2_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv2.conv.weight.scale.npy')
        dark4_csp_m2_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark4.1.m.2.conv2.conv.weight.zero_point.npy')
        dark4_csp_m2_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv2.conv.scale.npy')
        dark4_csp_m2_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv2.conv.zero_point.npy')

        dark4_csp_m2_conv2_coe_name = '../hand_coe/dark4_CSP_m2_Conv2.coe'
        self.dark4_csp_m2_conv2 = Conv2d_Q(quant_scale1=dark4_csp_m2_conv1_act_scale,
                                           quant_zero_point1=dark4_csp_m2_conv1_act_zp,
                                           quant_scale2=dark4_csp_m2_conv2_weight_scale,
                                           quant_zero_point2=dark4_csp_m2_conv2_weight_zp,
                                           quant_scale3=dark4_csp_m2_conv2_act_scale,
                                           quant_zero_point3=dark4_csp_m2_conv2_act_zp,
                                           coe_name=dark4_csp_m2_conv2_coe_name,
                                           operator='conv')

        dark4_csp_add2_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.csp.scale.npy')
        dark4_csp_add2_zp = np2tensor('../para/backbone.backbone.dark4.1.m.2.csp.zero_point.npy')

        # ============ dark4 csp cat ==========
        dark4_csp_cat_scale = np2tensor('../para/backbone.backbone.dark4.1.csp1.scale.npy')
        dark4_csp_cat_zp = np2tensor('../para/backbone.backbone.dark4.1.csp1.zero_point.npy')

        # ============ dark4 conv3 ==========
        dark4_csp_conv3_weight_scale = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.weight.scale.npy')
        dark4_csp_conv3_weight_zp = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.weight.zero_point.npy')
        dark4_csp_conv3_act_scale = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.scale.npy')
        dark4_csp_conv3_act_zp = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.zero_point.npy')
        dark4_csp_conv3_coe_name = '../hand_coe/dark4_CSP_Conv3.coe'
        self.dark4_csp_conv3 = Conv2d_Q(quant_scale1=dark4_csp_cat_scale, quant_zero_point1=dark4_csp_cat_zp,
                                        quant_scale2=dark4_csp_conv3_weight_scale,
                                        quant_zero_point2=dark4_csp_conv3_weight_zp,
                                        quant_scale3=dark4_csp_conv3_act_scale,
                                        quant_zero_point3=dark4_csp_conv3_act_zp,
                                        coe_name=dark4_csp_conv3_coe_name,
                                        operator='conv')

        # ==================start dark5====================
        # ======= BaseConv============
        dark5_BaseConv_weight_scale = np2tensor('../para/backbone.backbone.dark5.0.conv.weight.scale.npy')
        dark5_BaseConv_weight_zp = np2tensor('../para/backbone.backbone.dark5.0.conv.weight.zero_point.npy')
        dark5_BaseConv_act_scale = np2tensor('../para/backbone.backbone.dark5.0.conv.scale.npy')
        dark5_BaseConv_act_zp = np2tensor('../para/backbone.backbone.dark5.0.conv.zero_point.npy')

        dark5_BaseConv_coe_name = '../hand_coe/dark5_BaseConv.coe'
        self.dark5_baseconv = Conv2d_Q(quant_scale1=dark4_csp_conv3_act_scale, quant_zero_point1=dark4_csp_conv3_act_zp,
                                       quant_scale2=dark5_BaseConv_weight_scale,
                                       quant_zero_point2=dark5_BaseConv_weight_zp,
                                       quant_scale3=dark5_BaseConv_act_scale, quant_zero_point3=dark5_BaseConv_act_zp,
                                       coe_name=dark5_BaseConv_coe_name, operator='conv')

        # ========== CSPLayer ============
        dark5_csp_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark5.1.conv1.conv.weight.scale.npy')
        dark5_csp_conv1_weight_zp = np2tensor('../para/backbone.backbone.dark5.1.conv1.conv.weight.zero_point.npy')
        dark5_csp_conv1_act_scale = np2tensor('../para/backbone.backbone.dark5.1.conv1.conv.scale.npy')
        dark5_csp_conv1_act_zp = np2tensor('../para/backbone.backbone.dark5.1.conv1.conv.zero_point.npy')
        dark5_csp_conv1_coe_name = '../hand_coe/dark5_CSP_Conv1.coe'
        self.dark5_csp_conv1 = Conv2d_Q(quant_scale1=dark5_BaseConv_act_scale, quant_zero_point1=dark5_BaseConv_act_zp,
                                        quant_scale2=dark5_csp_conv1_weight_scale,
                                        quant_zero_point2=dark5_csp_conv1_weight_zp,
                                        quant_scale3=dark5_csp_conv1_act_scale,
                                        quant_zero_point3=dark5_csp_conv1_act_zp, coe_name=dark5_csp_conv1_coe_name,
                                        operator='conv')

        dark5_csp_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.weight.scale.npy')
        dark5_csp_conv2_weight_zp = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.weight.zero_point.npy')
        dark5_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.scale.npy')
        dark5_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.zero_point.npy')
        dark5_csp_conv2_coe_name = '../hand_coe/dark5_CSP_Conv2.coe'
        self.dark5_csp_conv2 = Conv2d_Q(quant_scale1=dark5_BaseConv_act_scale, quant_zero_point1=dark5_BaseConv_act_zp,
                                        quant_scale2=dark5_csp_conv2_weight_scale,
                                        quant_zero_point2=dark5_csp_conv2_weight_zp,
                                        quant_scale3=dark5_csp_conv2_act_scale,
                                        quant_zero_point3=dark5_csp_conv2_act_zp, coe_name=dark5_csp_conv2_coe_name,
                                        operator='conv')

        # ========== dark5_csp_m =============
        # m0
        dark5_csp_m0_conv1_weight_scale = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv1.conv.weight.scale.npy')
        dark5_csp_m0_conv1_weight_zp = np2tensor(
            '../para/backbone.backbone.dark5.1.m.0.conv1.conv.weight.zero_point.npy')
        dark5_csp_m0_conv1_act_scale = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv1.conv.scale.npy')
        dark5_csp_m0_conv1_act_zp = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv1.conv.zero_point.npy')
        dark5_csp_m0_conv1_coe_name = '../hand_coe/dark5_CSP_m0_Conv1.coe'
        self.dark5_csp_m0_conv1 = Conv2d_Q(quant_scale1=dark5_csp_conv1_act_scale,
                                           quant_zero_point1=dark5_csp_conv1_act_zp,
                                           quant_scale2=dark5_csp_m0_conv1_weight_scale,
                                           quant_zero_point2=dark5_csp_m0_conv1_weight_zp,
                                           quant_scale3=dark5_csp_m0_conv1_act_scale,
                                           quant_zero_point3=dark5_csp_m0_conv1_act_zp,
                                           coe_name=dark5_csp_m0_conv1_coe_name,
                                           operator='conv')

        dark5_csp_m0_conv2_weight_scale = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv2.conv.weight.scale.npy')
        dark5_csp_m0_conv2_weight_zp = np2tensor(
            '../para/backbone.backbone.dark5.1.m.0.conv2.conv.weight.zero_point.npy')
        dark5_csp_m0_conv2_act_scale = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv2.conv.scale.npy')
        dark5_csp_m0_conv2_act_zp = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv2.conv.zero_point.npy')
        dark5_csp_m0_conv2_coe_name = '../hand_coe/dark5_CSP_m0_Conv2.coe'
        self.dark5_csp_m0_conv2 = Conv2d_Q(quant_scale1=dark5_csp_m0_conv1_act_scale,
                                           quant_zero_point1=dark5_csp_m0_conv1_act_zp,
                                           quant_scale2=dark5_csp_m0_conv2_weight_scale,
                                           quant_zero_point2=dark5_csp_m0_conv2_weight_zp,
                                           quant_scale3=dark5_csp_m0_conv2_act_scale,
                                           quant_zero_point3=dark5_csp_m0_conv2_act_zp,
                                           coe_name=dark5_csp_m0_conv2_coe_name,
                                           operator='conv')

        # ============ dark5 csp cat ==========
        dark5_csp_cat_scale = np2tensor('../para/backbone.backbone.dark5.1.csp1.scale.npy')
        dark5_csp_cat_zp = np2tensor('../para/backbone.backbone.dark5.1.csp1.zero_point.npy')

        # ============ dark5 conv3 ==========
        dark5_csp_conv3_weight_scale = np2tensor('../para/backbone.backbone.dark5.1.conv3.conv.weight.scale.npy')
        dark5_csp_conv3_weight_zp = np2tensor('../para/backbone.backbone.dark5.1.conv3.conv.weight.zero_point.npy')
        dark5_csp_conv3_act_scale = np2tensor('../para/backbone.backbone.dark5.1.conv3.conv.scale.npy')
        dark5_csp_conv3_act_zp = np2tensor('../para/backbone.backbone.dark5.1.conv3.conv.zero_point.npy')
        dark5_csp_conv3_coe_name = '../hand_coe/dark5_CSP_Conv3.coe'
        self.dark5_csp_conv3 = Conv2d_Q(quant_scale1=dark5_csp_cat_scale, quant_zero_point1=dark5_csp_cat_zp,
                                        quant_scale2=dark5_csp_conv3_weight_scale,
                                        quant_zero_point2=dark5_csp_conv3_weight_zp,
                                        quant_scale3=dark5_csp_conv3_act_scale,
                                        quant_zero_point3=dark5_csp_conv3_act_zp,
                                        coe_name=dark5_csp_conv3_coe_name,
                                        operator='conv')

        # ========================dark5 结束========================

        # =================start lateral_conv0 ====================
        lateral_conv0_weight_scale = np2tensor('../para/backbone.lateral_conv0.conv.weight.scale.npy')
        lateral_conv0_weight_zp = np2tensor('../para/backbone.lateral_conv0.conv.weight.zero_point.npy')
        lateral_conv0_act_scale = np2tensor('../para/backbone.lateral_conv0.conv.scale.npy')
        lateral_conv0_act_zp = np2tensor('../para/backbone.lateral_conv0.conv.zero_point.npy')
        lateral_conv0_coe_name = '../hand_coe/lateral_conv0.coe'
        self.lateral_conv0 = Conv2d_Q(quant_scale1=dark5_csp_conv3_act_scale, quant_zero_point1=dark5_csp_conv3_act_zp,
                                      quant_scale2=lateral_conv0_weight_scale,
                                      quant_zero_point2=lateral_conv0_weight_zp,
                                      quant_scale3=lateral_conv0_act_scale,
                                      quant_zero_point3=lateral_conv0_act_zp,
                                      coe_name=lateral_conv0_coe_name,
                                      operator='conv')

        # ========================C3_p4=========================
        YOLOPAFPN_csp2_cat_scale = np2tensor('../para/backbone.csp2.scale.npy')
        YOLOPAFPN_csp2_cat_zp = np2tensor('../para/backbone.csp2.zero_point.npy')
        c3_p4_csp_conv1_weight_scale = np2tensor('../para/backbone.C3_p4.conv1.conv.weight.scale.npy')
        c3_p4_csp_conv1_weight_zp = np2tensor('../para/backbone.C3_p4.conv1.conv.weight.zero_point.npy')
        c3_p4_csp_conv1_act_scale = np2tensor('../para/backbone.C3_p4.conv1.conv.scale.npy')
        c3_p4_csp_conv1_act_zp = np2tensor('../para/backbone.C3_p4.conv1.conv.zero_point.npy')
        c3_p4_csp_conv1_coe_name = '../hand_coe/C3_P4_CSP_Conv1.coe'
        self.c3_p4_csp_conv1 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp2_cat_scale, quant_zero_point1=YOLOPAFPN_csp2_cat_zp,
                                        quant_scale2=c3_p4_csp_conv1_weight_scale,
                                        quant_zero_point2=c3_p4_csp_conv1_weight_zp,
                                        quant_scale3=c3_p4_csp_conv1_act_scale,
                                        quant_zero_point3=c3_p4_csp_conv1_act_zp, coe_name=c3_p4_csp_conv1_coe_name,
                                        operator='conv')

        c3_p4_csp_conv2_weight_scale = np2tensor('../para/backbone.C3_p4.conv2.conv.weight.scale.npy')
        c3_p4_csp_conv2_weight_zp = np2tensor('../para/backbone.C3_p4.conv2.conv.weight.zero_point.npy')
        c3_p4_csp_conv2_act_scale = np2tensor('../para/backbone.C3_p4.conv2.conv.scale.npy')
        c3_p4_csp_conv2_act_zp = np2tensor('../para/backbone.C3_p4.conv2.conv.zero_point.npy')
        c3_p4_csp_conv2_coe_name = '../hand_coe/C3_P4_CSP_Conv2.coe'
        self.c3_p4_csp_conv2 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp2_cat_scale, quant_zero_point1=YOLOPAFPN_csp2_cat_zp,
                                        quant_scale2=c3_p4_csp_conv2_weight_scale,
                                        quant_zero_point2=c3_p4_csp_conv2_weight_zp,
                                        quant_scale3=c3_p4_csp_conv2_act_scale,
                                        quant_zero_point3=c3_p4_csp_conv2_act_zp, coe_name=c3_p4_csp_conv2_coe_name,
                                        operator='conv')
        # ============ start c3_p4_csp_m==================

        c3_p4_csp_m0_conv1_weight_scale = np2tensor('../para/backbone.C3_p4.m.0.conv1.conv.weight.scale.npy')
        c3_p4_csp_m0_conv1_weight_zp = np2tensor(
            '../para/backbone.C3_p4.m.0.conv1.conv.weight.zero_point.npy')
        c3_p4_csp_m0_conv1_act_scale = np2tensor('../para/backbone.C3_p4.m.0.conv1.conv.scale.npy')
        c3_p4_csp_m0_conv1_act_zp = np2tensor('../para/backbone.C3_p4.m.0.conv1.conv.zero_point.npy')
        c3_p4_csp_m0_conv1_coe_name = '../hand_coe/C3_P4_CSP_m0_Conv1.coe'
        self.c3_p4_csp_m0_conv1 = Conv2d_Q(quant_scale1=c3_p4_csp_conv1_act_scale,
                                           quant_zero_point1=c3_p4_csp_conv1_act_zp,
                                           quant_scale2=c3_p4_csp_m0_conv1_weight_scale,
                                           quant_zero_point2=c3_p4_csp_m0_conv1_weight_zp,
                                           quant_scale3=c3_p4_csp_m0_conv1_act_scale,
                                           quant_zero_point3=c3_p4_csp_m0_conv1_act_zp,
                                           coe_name=c3_p4_csp_m0_conv1_coe_name,
                                           operator='conv')

        c3_p4_csp_m0_conv2_weight_scale = np2tensor('../para/backbone.C3_p4.m.0.conv2.conv.weight.scale.npy')
        c3_p4_csp_m0_conv2_weight_zp = np2tensor(
            '../para/backbone.C3_p4.m.0.conv2.conv.weight.zero_point.npy')
        c3_p4_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_p4.m.0.conv2.conv.scale.npy')
        c3_p4_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_p4.m.0.conv2.conv.zero_point.npy')
        c3_p4_csp_m0_conv2_coe_name = '../hand_coe/C3_P4_CSP_m0_Conv2.coe'
        self.c3_p4_csp_m0_conv2 = Conv2d_Q(quant_scale1=c3_p4_csp_m0_conv1_act_scale,
                                           quant_zero_point1=c3_p4_csp_m0_conv1_act_zp,
                                           quant_scale2=c3_p4_csp_m0_conv2_weight_scale,
                                           quant_zero_point2=c3_p4_csp_m0_conv2_weight_zp,
                                           quant_scale3=c3_p4_csp_m0_conv2_act_scale,
                                           quant_zero_point3=c3_p4_csp_m0_conv2_act_zp,
                                           coe_name=c3_p4_csp_m0_conv2_coe_name,
                                           operator='conv')
        c3_p4_csp_cat_scale = np2tensor('../para/backbone.C3_p4.csp1.scale.npy')
        c3_p4_csp_cat_zp = np2tensor('../para/backbone.C3_p4.csp1.zero_point.npy')
        c3_p4_csp_conv3_weight_scale = np2tensor('../para/backbone.C3_p4.conv3.conv.weight.scale.npy')
        c3_p4_csp_conv3_weight_zp = np2tensor('../para/backbone.C3_p4.conv3.conv.weight.zero_point.npy')
        c3_p4_csp_conv3_act_scale = np2tensor('../para/backbone.C3_p4.conv3.conv.scale.npy')
        c3_p4_csp_conv3_act_zp = np2tensor('../para/backbone.C3_p4.conv3.conv.zero_point.npy')

        c3_p4_csp_conv3_coe_name = '../hand_coe/C3_P4_CSP_Conv3.coe'
        self.c3_p4_csp_conv3 = Conv2d_Q(quant_scale1=c3_p4_csp_cat_scale, quant_zero_point1=c3_p4_csp_cat_zp,
                                        quant_scale2=c3_p4_csp_conv3_weight_scale,
                                        quant_zero_point2=c3_p4_csp_conv3_weight_zp,
                                        quant_scale3=c3_p4_csp_conv3_act_scale,
                                        quant_zero_point3=c3_p4_csp_conv3_act_zp, coe_name=c3_p4_csp_conv3_coe_name,
                                        operator='conv')

        # ===============start reduce_conv1 ==============
        reduce_conv1_weight_scale = np2tensor('../para/backbone.reduce_conv1.conv.weight.scale.npy')
        reduce_conv1_weight_zp = np2tensor('../para/backbone.reduce_conv1.conv.weight.zero_point.npy')
        reduce_conv1_act_scale = np2tensor('../para/backbone.reduce_conv1.conv.scale.npy')
        reduce_conv1_act_zp = np2tensor('../para/backbone.reduce_conv1.conv.zero_point.npy')
        reduce_conv1_coe_name = '../hand_coe/reduce_conv1.coe'
        self.reduce_conv1 = Conv2d_Q(quant_scale1=c3_p4_csp_conv3_act_scale, quant_zero_point1=c3_p4_csp_conv3_act_zp,
                                     quant_scale2=reduce_conv1_weight_scale,
                                     quant_zero_point2=reduce_conv1_weight_zp,
                                     quant_scale3=reduce_conv1_act_scale,
                                     quant_zero_point3=reduce_conv1_act_zp, coe_name=reduce_conv1_coe_name,
                                     operator='conv')

        # ========================C3_p3=========================
        YOLOPAFPN_csp3_cat_scale = np2tensor('../para/backbone.csp3.scale.npy')
        YOLOPAFPN_csp3_cat_zp = np2tensor('../para/backbone.csp3.zero_point.npy')
        c3_p3_csp_conv1_weight_scale = np2tensor('../para/backbone.C3_p3.conv1.conv.weight.scale.npy')
        c3_p3_csp_conv1_weight_zp = np2tensor('../para/backbone.C3_p3.conv1.conv.weight.zero_point.npy')
        c3_p3_csp_conv1_act_scale = np2tensor('../para/backbone.C3_p3.conv1.conv.scale.npy')
        c3_p3_csp_conv1_act_zp = np2tensor('../para/backbone.C3_p3.conv1.conv.zero_point.npy')
        c3_p3_csp_conv1_coe_name = '../hand_coe/C3_P3_CSP_Conv1.coe'
        self.c3_p3_csp_conv1 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp3_cat_scale, quant_zero_point1=YOLOPAFPN_csp3_cat_zp,
                                        quant_scale2=c3_p3_csp_conv1_weight_scale,
                                        quant_zero_point2=c3_p3_csp_conv1_weight_zp,
                                        quant_scale3=c3_p3_csp_conv1_act_scale,
                                        quant_zero_point3=c3_p3_csp_conv1_act_zp, coe_name=c3_p3_csp_conv1_coe_name,
                                        operator='conv')

        c3_p3_csp_conv2_weight_scale = np2tensor('../para/backbone.C3_p3.conv2.conv.weight.scale.npy')
        c3_p3_csp_conv2_weight_zp = np2tensor('../para/backbone.C3_p3.conv2.conv.weight.zero_point.npy')
        c3_p3_csp_conv2_act_scale = np2tensor('../para/backbone.C3_p3.conv2.conv.scale.npy')
        c3_p3_csp_conv2_act_zp = np2tensor('../para/backbone.C3_p3.conv2.conv.zero_point.npy')
        c3_p3_csp_conv2_coe_name = '../hand_coe/C3_P3_CSP_Conv2.coe'
        self.c3_p3_csp_conv2 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp3_cat_scale, quant_zero_point1=YOLOPAFPN_csp3_cat_zp,
                                        quant_scale2=c3_p3_csp_conv2_weight_scale,
                                        quant_zero_point2=c3_p3_csp_conv2_weight_zp,
                                        quant_scale3=c3_p3_csp_conv2_act_scale,
                                        quant_zero_point3=c3_p3_csp_conv2_act_zp, coe_name=c3_p3_csp_conv2_coe_name,
                                        operator='conv')

        # ============ start c3_p3_csp_m==================

        c3_p3_csp_m0_conv1_weight_scale = np2tensor('../para/backbone.C3_p3.m.0.conv1.conv.weight.scale.npy')
        c3_p3_csp_m0_conv1_weight_zp = np2tensor(
            '../para/backbone.C3_p3.m.0.conv1.conv.weight.zero_point.npy')
        c3_p3_csp_m0_conv1_act_scale = np2tensor('../para/backbone.C3_p3.m.0.conv1.conv.scale.npy')
        c3_p3_csp_m0_conv1_act_zp = np2tensor('../para/backbone.C3_p3.m.0.conv1.conv.zero_point.npy')
        c3_p3_csp_m0_conv1_coe_name = '../hand_coe/C3_P3_CSP_m0_Conv1.coe'
        self.c3_p3_csp_m0_conv1 = Conv2d_Q(quant_scale1=c3_p3_csp_conv1_act_scale,
                                           quant_zero_point1=c3_p3_csp_conv1_act_zp,
                                           quant_scale2=c3_p3_csp_m0_conv1_weight_scale,
                                           quant_zero_point2=c3_p3_csp_m0_conv1_weight_zp,
                                           quant_scale3=c3_p3_csp_m0_conv1_act_scale,
                                           quant_zero_point3=c3_p3_csp_m0_conv1_act_zp,
                                           coe_name=c3_p3_csp_m0_conv1_coe_name,
                                           operator='conv')

        c3_p3_csp_m0_conv2_weight_scale = np2tensor('../para/backbone.C3_p3.m.0.conv2.conv.weight.scale.npy')
        c3_p3_csp_m0_conv2_weight_zp = np2tensor(
            '../para/backbone.C3_p3.m.0.conv2.conv.weight.zero_point.npy')
        c3_p3_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_p3.m.0.conv2.conv.scale.npy')
        c3_p3_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_p3.m.0.conv2.conv.zero_point.npy')
        c3_p3_csp_m0_conv2_coe_name = '../hand_coe/C3_P3_CSP_m0_Conv2.coe'
        self.c3_p3_csp_m0_conv2 = Conv2d_Q(quant_scale1=c3_p3_csp_m0_conv1_act_scale,
                                           quant_zero_point1=c3_p3_csp_m0_conv1_act_zp,
                                           quant_scale2=c3_p3_csp_m0_conv2_weight_scale,
                                           quant_zero_point2=c3_p3_csp_m0_conv2_weight_zp,
                                           quant_scale3=c3_p3_csp_m0_conv2_act_scale,
                                           quant_zero_point3=c3_p3_csp_m0_conv2_act_zp,
                                           coe_name=c3_p3_csp_m0_conv2_coe_name,
                                           operator='conv')
        # =========== c3_p3_csp_conv3==================
        c3_p3_csp_cat_scale = np2tensor('../para/backbone.C3_p3.csp1.scale.npy')
        c3_p3_csp_cat_zp = np2tensor('../para/backbone.C3_p3.csp1.zero_point.npy')
        c3_p3_csp_conv3_weight_scale = np2tensor('../para/backbone.C3_p3.conv3.conv.weight.scale.npy')
        c3_p3_csp_conv3_weight_zp = np2tensor('../para/backbone.C3_p3.conv3.conv.weight.zero_point.npy')
        c3_p3_csp_conv3_act_scale = np2tensor('../para/backbone.C3_p3.conv3.conv.scale.npy')
        c3_p3_csp_conv3_act_zp = np2tensor('../para/backbone.C3_p3.conv3.conv.zero_point.npy')
        c3_p3_csp_conv3_coe_name = '../hand_coe/C3_P3_CSP_Conv3.coe'
        self.c3_p3_csp_conv3 = Conv2d_Q(quant_scale1=c3_p3_csp_cat_scale, quant_zero_point1=c3_p3_csp_cat_zp,
                                        quant_scale2=c3_p3_csp_conv3_weight_scale,
                                        quant_zero_point2=c3_p3_csp_conv3_weight_zp,
                                        quant_scale3=c3_p3_csp_conv3_act_scale,
                                        quant_zero_point3=c3_p3_csp_conv3_act_zp, coe_name=c3_p3_csp_conv3_coe_name,
                                        operator='conv')
        # ==========start bu_conv2 ===============
        bu_conv2_weight_scale = np2tensor('../para/backbone.bu_conv2.conv.weight.scale.npy')
        bu_conv2_weight_zp = np2tensor('../para/backbone.bu_conv2.conv.weight.zero_point.npy')
        bu_conv2_act_scale = np2tensor('../para/backbone.bu_conv2.conv.scale.npy')
        bu_conv2_act_zp = np2tensor('../para/backbone.bu_conv2.conv.zero_point.npy')
        bu_conv2_coe_name = '../hand_coe/bu_conv2.coe'
        self.bu_conv2 = Conv2d_Q(quant_scale1=c3_p3_csp_conv3_act_scale, quant_zero_point1=c3_p3_csp_conv3_act_zp,
                                 quant_scale2=bu_conv2_weight_scale,
                                 quant_zero_point2=bu_conv2_weight_zp,
                                 quant_scale3=bu_conv2_act_scale,
                                 quant_zero_point3=bu_conv2_act_zp, coe_name=bu_conv2_coe_name,
                                 operator='conv')

        # ==================== start c3_n3 ===================
        YOLOPAFPN_csp4_cat_scale = np2tensor('../para/backbone.csp4.scale.npy')
        YOLOPAFPN_csp4_cat_zp = np2tensor('../para/backbone.csp4.zero_point.npy')
        c3_n3_csp_conv1_weight_scale = np2tensor('../para/backbone.C3_n3.conv1.conv.weight.scale.npy')
        c3_n3_csp_conv1_weight_zp = np2tensor('../para/backbone.C3_n3.conv1.conv.weight.zero_point.npy')
        c3_n3_csp_conv1_act_scale = np2tensor('../para/backbone.C3_n3.conv1.conv.scale.npy')
        c3_n3_csp_conv1_act_zp = np2tensor('../para/backbone.C3_n3.conv1.conv.zero_point.npy')
        c3_n3_csp_conv1_coe_name = '../hand_coe/C3_n3_CSP_Conv1.coe'
        self.c3_n3_csp_conv1 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp4_cat_scale, quant_zero_point1=YOLOPAFPN_csp4_cat_zp,
                                        quant_scale2=c3_n3_csp_conv1_weight_scale,
                                        quant_zero_point2=c3_n3_csp_conv1_weight_zp,
                                        quant_scale3=c3_n3_csp_conv1_act_scale,
                                        quant_zero_point3=c3_n3_csp_conv1_act_zp, coe_name=c3_n3_csp_conv1_coe_name,
                                        operator='conv')

        c3_n3_csp_conv2_weight_scale = np2tensor('../para/backbone.C3_n3.conv2.conv.weight.scale.npy')
        c3_n3_csp_conv2_weight_zp = np2tensor('../para/backbone.C3_n3.conv2.conv.weight.zero_point.npy')
        c3_n3_csp_conv2_act_scale = np2tensor('../para/backbone.C3_n3.conv2.conv.scale.npy')
        c3_n3_csp_conv2_act_zp = np2tensor('../para/backbone.C3_n3.conv2.conv.zero_point.npy')
        c3_n3_csp_conv2_coe_name = '../hand_coe/C3_n3_CSP_Conv2.coe'
        self.c3_n3_csp_conv2 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp4_cat_scale, quant_zero_point1=YOLOPAFPN_csp4_cat_zp,
                                        quant_scale2=c3_n3_csp_conv2_weight_scale,
                                        quant_zero_point2=c3_n3_csp_conv2_weight_zp,
                                        quant_scale3=c3_n3_csp_conv2_act_scale,
                                        quant_zero_point3=c3_n3_csp_conv2_act_zp, coe_name=c3_n3_csp_conv2_coe_name,
                                        operator='conv')

        # ============ start c3_n3_csp_m==================

        c3_n3_csp_m0_conv1_weight_scale = np2tensor('../para/backbone.C3_n3.m.0.conv1.conv.weight.scale.npy')
        c3_n3_csp_m0_conv1_weight_zp = np2tensor(
            '../para/backbone.C3_n3.m.0.conv1.conv.weight.zero_point.npy')
        c3_n3_csp_m0_conv1_act_scale = np2tensor('../para/backbone.C3_n3.m.0.conv1.conv.scale.npy')
        c3_n3_csp_m0_conv1_act_zp = np2tensor('../para/backbone.C3_n3.m.0.conv1.conv.zero_point.npy')
        c3_n3_csp_m0_conv1_coe_name = '../hand_coe/C3_n3_CSP_m0_Conv1.coe'
        self.c3_n3_csp_m0_conv1 = Conv2d_Q(quant_scale1=c3_n3_csp_conv1_act_scale,
                                           quant_zero_point1=c3_n3_csp_conv1_act_zp,
                                           quant_scale2=c3_n3_csp_m0_conv1_weight_scale,
                                           quant_zero_point2=c3_n3_csp_m0_conv1_weight_zp,
                                           quant_scale3=c3_n3_csp_m0_conv1_act_scale,
                                           quant_zero_point3=c3_n3_csp_m0_conv1_act_zp,
                                           coe_name=c3_n3_csp_m0_conv1_coe_name,
                                           operator='conv')

        c3_n3_csp_m0_conv2_weight_scale = np2tensor('../para/backbone.C3_n3.m.0.conv2.conv.weight.scale.npy')
        c3_n3_csp_m0_conv2_weight_zp = np2tensor(
            '../para/backbone.C3_n3.m.0.conv2.conv.weight.zero_point.npy')
        c3_n3_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_n3.m.0.conv2.conv.scale.npy')
        c3_n3_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_n3.m.0.conv2.conv.zero_point.npy')
        c3_n3_csp_m0_conv2_coe_name = '../hand_coe/C3_n3_CSP_m0_Conv2.coe'
        self.c3_n3_csp_m0_conv2 = Conv2d_Q(quant_scale1=c3_n3_csp_m0_conv1_act_scale,
                                           quant_zero_point1=c3_n3_csp_m0_conv1_act_zp,
                                           quant_scale2=c3_n3_csp_m0_conv2_weight_scale,
                                           quant_zero_point2=c3_n3_csp_m0_conv2_weight_zp,
                                           quant_scale3=c3_n3_csp_m0_conv2_act_scale,
                                           quant_zero_point3=c3_n3_csp_m0_conv2_act_zp,
                                           coe_name=c3_n3_csp_m0_conv2_coe_name,
                                           operator='conv')

        # =========== c3_n3_csp_conv3==================
        c3_n3_csp_cat_scale = np2tensor('../para/backbone.C3_n3.csp1.scale.npy')
        c3_n3_csp_cat_zp = np2tensor('../para/backbone.C3_n3.csp1.zero_point.npy')
        c3_n3_csp_conv3_weight_scale = np2tensor('../para/backbone.C3_n3.conv3.conv.weight.scale.npy')
        c3_n3_csp_conv3_weight_zp = np2tensor('../para/backbone.C3_n3.conv3.conv.weight.zero_point.npy')
        c3_n3_csp_conv3_act_scale = np2tensor('../para/backbone.C3_n3.conv3.conv.scale.npy')
        c3_n3_csp_conv3_act_zp = np2tensor('../para/backbone.C3_n3.conv3.conv.zero_point.npy')
        c3_n3_csp_conv3_coe_name = '../hand_coe/C3_n3_CSP_Conv3.coe'
        self.c3_n3_csp_conv3 = Conv2d_Q(quant_scale1=c3_n3_csp_cat_scale, quant_zero_point1=c3_n3_csp_cat_zp,
                                        quant_scale2=c3_n3_csp_conv3_weight_scale,
                                        quant_zero_point2=c3_n3_csp_conv3_weight_zp,
                                        quant_scale3=c3_n3_csp_conv3_act_scale,
                                        quant_zero_point3=c3_n3_csp_conv3_act_zp, coe_name=c3_n3_csp_conv3_coe_name,
                                        operator='conv')

        # ==================bu_conv1=====================

        bu_conv1_weight_scale = np2tensor('../para/backbone.bu_conv1.conv.weight.scale.npy')
        bu_conv1_weight_zp = np2tensor('../para/backbone.bu_conv1.conv.weight.zero_point.npy')
        bu_conv1_act_scale = np2tensor('../para/backbone.bu_conv1.conv.scale.npy')
        bu_conv1_act_zp = np2tensor('../para/backbone.bu_conv1.conv.zero_point.npy')
        bu_conv1_coe_name = '../hand_coe/bu_conv1.coe'
        self.bu_conv1 = Conv2d_Q(quant_scale1=c3_n3_csp_conv3_act_scale, quant_zero_point1=c3_n3_csp_conv3_act_zp,
                                 quant_scale2=bu_conv1_weight_scale,
                                 quant_zero_point2=bu_conv1_weight_zp,
                                 quant_scale3=bu_conv1_act_scale,
                                 quant_zero_point3=bu_conv1_act_zp, coe_name=bu_conv1_coe_name,
                                 operator='conv')

        # ==================== start c3_n4 ===================
        YOLOPAFPN_csp5_cat_scale = np2tensor('../para/backbone.csp5.scale.npy')
        YOLOPAFPN_csp5_cat_zp = np2tensor('../para/backbone.csp5.zero_point.npy')
        c3_n4_csp_conv1_weight_scale = np2tensor('../para/backbone.C3_n4.conv1.conv.weight.scale.npy')
        c3_n4_csp_conv1_weight_zp = np2tensor('../para/backbone.C3_n4.conv1.conv.weight.zero_point.npy')
        c3_n4_csp_conv1_act_scale = np2tensor('../para/backbone.C3_n4.conv1.conv.scale.npy')
        c3_n4_csp_conv1_act_zp = np2tensor('../para/backbone.C3_n4.conv1.conv.zero_point.npy')
        c3_n4_csp_conv1_coe_name = '../hand_coe/C3_n4_CSP_Conv1.coe'
        self.c3_n4_csp_conv1 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp5_cat_scale, quant_zero_point1=YOLOPAFPN_csp5_cat_zp,
                                        quant_scale2=c3_n4_csp_conv1_weight_scale,
                                        quant_zero_point2=c3_n4_csp_conv1_weight_zp,
                                        quant_scale3=c3_n4_csp_conv1_act_scale,
                                        quant_zero_point3=c3_n4_csp_conv1_act_zp, coe_name=c3_n4_csp_conv1_coe_name,
                                        operator='conv')

        c3_n4_csp_conv2_weight_scale = np2tensor('../para/backbone.C3_n4.conv2.conv.weight.scale.npy')
        c3_n4_csp_conv2_weight_zp = np2tensor('../para/backbone.C3_n4.conv2.conv.weight.zero_point.npy')
        c3_n4_csp_conv2_act_scale = np2tensor('../para/backbone.C3_n4.conv2.conv.scale.npy')
        c3_n4_csp_conv2_act_zp = np2tensor('../para/backbone.C3_n4.conv2.conv.zero_point.npy')
        c3_n4_csp_conv2_coe_name = '../hand_coe/C3_n4_CSP_Conv2.coe'
        self.c3_n4_csp_conv2 = Conv2d_Q(quant_scale1=YOLOPAFPN_csp5_cat_scale, quant_zero_point1=YOLOPAFPN_csp5_cat_zp,
                                        quant_scale2=c3_n4_csp_conv2_weight_scale,
                                        quant_zero_point2=c3_n4_csp_conv2_weight_zp,
                                        quant_scale3=c3_n4_csp_conv2_act_scale,
                                        quant_zero_point3=c3_n4_csp_conv2_act_zp, coe_name=c3_n4_csp_conv2_coe_name,
                                        operator='conv')

        # ============ start c3_n4_csp_m==================

        c3_n4_csp_m0_conv1_weight_scale = np2tensor('../para/backbone.C3_n4.m.0.conv1.conv.weight.scale.npy')
        c3_n4_csp_m0_conv1_weight_zp = np2tensor(
            '../para/backbone.C3_n4.m.0.conv1.conv.weight.zero_point.npy')
        c3_n4_csp_m0_conv1_act_scale = np2tensor('../para/backbone.C3_n4.m.0.conv1.conv.scale.npy')
        c3_n4_csp_m0_conv1_act_zp = np2tensor('../para/backbone.C3_n4.m.0.conv1.conv.zero_point.npy')
        c3_n4_csp_m0_conv1_coe_name = '../hand_coe/C3_n4_CSP_m0_Conv1.coe'
        self.c3_n4_csp_m0_conv1 = Conv2d_Q(quant_scale1=c3_n4_csp_conv1_act_scale,
                                           quant_zero_point1=c3_n4_csp_conv1_act_zp,
                                           quant_scale2=c3_n4_csp_m0_conv1_weight_scale,
                                           quant_zero_point2=c3_n4_csp_m0_conv1_weight_zp,
                                           quant_scale3=c3_n4_csp_m0_conv1_act_scale,
                                           quant_zero_point3=c3_n4_csp_m0_conv1_act_zp,
                                           coe_name=c3_n4_csp_m0_conv1_coe_name,
                                           operator='conv')

        c3_n4_csp_m0_conv2_weight_scale = np2tensor('../para/backbone.C3_n4.m.0.conv2.conv.weight.scale.npy')
        c3_n4_csp_m0_conv2_weight_zp = np2tensor(
            '../para/backbone.C3_n4.m.0.conv2.conv.weight.zero_point.npy')
        c3_n4_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_n4.m.0.conv2.conv.scale.npy')
        c3_n4_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_n4.m.0.conv2.conv.zero_point.npy')
        c3_n4_csp_m0_conv2_coe_name = '../hand_coe/C3_n4_CSP_m0_Conv2.coe'
        self.c3_n4_csp_m0_conv2 = Conv2d_Q(quant_scale1=c3_n4_csp_m0_conv1_act_scale,
                                           quant_zero_point1=c3_n4_csp_m0_conv1_act_zp,
                                           quant_scale2=c3_n4_csp_m0_conv2_weight_scale,
                                           quant_zero_point2=c3_n4_csp_m0_conv2_weight_zp,
                                           quant_scale3=c3_n4_csp_m0_conv2_act_scale,
                                           quant_zero_point3=c3_n4_csp_m0_conv2_act_zp,
                                           coe_name=c3_n4_csp_m0_conv2_coe_name,
                                           operator='conv')

        # =========== c3_n4_csp_conv3==================
        c3_n4_csp_cat_scale = np2tensor('../para/backbone.C3_n4.csp1.scale.npy')
        c3_n4_csp_cat_zp = np2tensor('../para/backbone.C3_n4.csp1.zero_point.npy')
        c3_n4_csp_conv3_weight_scale = np2tensor('../para/backbone.C3_n4.conv3.conv.weight.scale.npy')
        c3_n4_csp_conv3_weight_zp = np2tensor('../para/backbone.C3_n4.conv3.conv.weight.zero_point.npy')
        c3_n4_csp_conv3_act_scale = np2tensor('../para/backbone.C3_n4.conv3.conv.scale.npy')
        c3_n4_csp_conv3_act_zp = np2tensor('../para/backbone.C3_n4.conv3.conv.zero_point.npy')
        c3_n4_csp_conv3_coe_name = '../hand_coe/C3_n4_CSP_Conv3.coe'
        self.c3_n4_csp_conv3 = Conv2d_Q(quant_scale1=c3_n4_csp_cat_scale, quant_zero_point1=c3_n4_csp_cat_zp,
                                        quant_scale2=c3_n4_csp_conv3_weight_scale,
                                        quant_zero_point2=c3_n4_csp_conv3_weight_zp,
                                        quant_scale3=c3_n4_csp_conv3_act_scale,
                                        quant_zero_point3=c3_n4_csp_conv3_act_zp, coe_name=c3_n4_csp_conv3_coe_name,
                                        operator='conv')

        # =============start yolo_head===============
        # ================P3_out======================
        c3_p3_csp_conv3_act_scale = np2tensor('../para/backbone.C3_p3.conv3.conv.scale.npy')
        c3_p3_csp_conv3_act_zp = np2tensor('../para/backbone.C3_p3.conv3.conv.zero_point.npy')
        p3_stem_conv_weight_scale = np2tensor('../para/head.stems.0.conv.weight.scale.npy')
        p3_stem_conv_weight_zp = np2tensor('../para/head.stems.0.conv.weight.zero_point.npy')
        p3_stem_conv_act_scale = np2tensor('../para/head.stems.0.conv.scale.npy')
        p3_stem_conv_act_zp = np2tensor('../para/head.stems.0.conv.zero_point.npy')
        p3_stem_coe_name = '../hand_coe/P3_stem.coe'
        self.p3_stem_conv = Conv2d_Q(quant_scale1=c3_p3_csp_conv3_act_scale, quant_zero_point1=c3_p3_csp_conv3_act_zp,
                                     quant_scale2=p3_stem_conv_weight_scale,
                                     quant_zero_point2=p3_stem_conv_weight_zp,
                                     quant_scale3=p3_stem_conv_act_scale,
                                     quant_zero_point3=p3_stem_conv_act_zp, coe_name=p3_stem_coe_name,
                                     operator='conv')

        # =================P3_cls_conv=============
        p3_cls_conv0_weight_scale = np2tensor('../para/head.cls_convs.0.0.conv.weight.scale.npy')
        p3_cls_conv0_weight_zp = np2tensor('../para/head.cls_convs.0.0.conv.weight.zero_point.npy')
        p3_cls_conv0_act_scale = np2tensor('../para/head.cls_convs.0.0.conv.scale.npy')
        p3_cls_conv0_act_zp = np2tensor('../para/head.cls_convs.0.0.conv.zero_point.npy')
        p3_cls_conv0_coe_name = '../hand_coe/P3_cls_conv0.coe'
        self.p3_cls_conv0 = Conv2d_Q(quant_scale1=p3_stem_conv_act_scale, quant_zero_point1=p3_stem_conv_act_zp,
                                     quant_scale2=p3_cls_conv0_weight_scale,
                                     quant_zero_point2=p3_cls_conv0_weight_zp,
                                     quant_scale3=p3_cls_conv0_act_scale,
                                     quant_zero_point3=p3_cls_conv0_act_zp, coe_name=p3_cls_conv0_coe_name,
                                     operator='conv')
        p3_cls_conv1_weight_scale = np2tensor('../para/head.cls_convs.0.1.conv.weight.scale.npy')
        p3_cls_conv1_weight_zp = np2tensor('../para/head.cls_convs.0.1.conv.weight.zero_point.npy')
        p3_cls_conv1_act_scale = np2tensor('../para/head.cls_convs.0.1.conv.scale.npy')
        p3_cls_conv1_act_zp = np2tensor('../para/head.cls_convs.0.1.conv.zero_point.npy')
        p3_cls_conv1_coe_name = '../hand_coe/P3_cls_feat.coe'
        self.p3_cls_conv1 = Conv2d_Q(quant_scale1=p3_cls_conv0_act_scale, quant_zero_point1=p3_cls_conv0_act_zp,
                                     quant_scale2=p3_cls_conv1_weight_scale,
                                     quant_zero_point2=p3_cls_conv1_weight_zp,
                                     quant_scale3=p3_cls_conv1_act_scale,
                                     quant_zero_point3=p3_cls_conv1_act_zp, coe_name=p3_cls_conv1_coe_name,
                                     operator='conv')

        # ===============P3_out_cls_preds==============
        p3_cls_preds_weight_scale = np2tensor('../para/head.cls_preds.0.weight.scale.npy')
        p3_cls_preds_weight_zp = np2tensor('../para/head.cls_preds.0.weight.zero_point.npy')
        p3_cls_preds_act_scale = np2tensor('../para/head.cls_preds.0.scale.npy')
        p3_cls_preds_act_zp = np2tensor('../para/head.cls_preds.0.zero_point.npy')
        p3_cls_preds_coe_name = '../hand_coe/P3_cls_output.coe'
        self.p3_cls_preds = Conv2d_Q(quant_scale1=p3_cls_conv1_act_scale, quant_zero_point1=p3_cls_conv1_act_zp,
                                     quant_scale2=p3_cls_preds_weight_scale,
                                     quant_zero_point2=p3_cls_preds_weight_zp,
                                     quant_scale3=p3_cls_preds_act_scale,
                                     quant_zero_point3=p3_cls_preds_act_zp, coe_name=p3_cls_preds_coe_name,
                                     operator='conv2d')

        # ===============P3_out_reg_convs==============

        p3_reg_conv0_weight_scale = np2tensor('../para/head.reg_convs.0.0.conv.weight.scale.npy')
        p3_reg_conv0_weight_zp = np2tensor('../para/head.reg_convs.0.0.conv.weight.zero_point.npy')
        p3_reg_conv0_act_scale = np2tensor('../para/head.reg_convs.0.0.conv.scale.npy')
        p3_reg_conv0_act_zp = np2tensor('../para/head.reg_convs.0.0.conv.zero_point.npy')
        p3_reg_conv0_coe_name = '../hand_coe/P3_reg_conv0.coe'
        self.p3_reg_conv0 = Conv2d_Q(quant_scale1=p3_stem_conv_act_scale, quant_zero_point1=p3_stem_conv_act_zp,
                                     quant_scale2=p3_reg_conv0_weight_scale,
                                     quant_zero_point2=p3_reg_conv0_weight_zp,
                                     quant_scale3=p3_reg_conv0_act_scale,
                                     quant_zero_point3=p3_reg_conv0_act_zp, coe_name=p3_reg_conv0_coe_name,
                                     operator='conv')
        p3_reg_conv1_weight_scale = np2tensor('../para/head.reg_convs.0.1.conv.weight.scale.npy')
        p3_reg_conv1_weight_zp = np2tensor('../para/head.reg_convs.0.1.conv.weight.zero_point.npy')
        p3_reg_conv1_act_scale = np2tensor('../para/head.reg_convs.0.1.conv.scale.npy')
        p3_reg_conv1_act_zp = np2tensor('../para/head.reg_convs.0.1.conv.zero_point.npy')
        p3_reg_conv1_coe_name = '../hand_coe/P3_reg_feat.coe'
        self.p3_reg_conv1 = Conv2d_Q(quant_scale1=p3_reg_conv0_act_scale, quant_zero_point1=p3_reg_conv0_act_zp,
                                     quant_scale2=p3_reg_conv1_weight_scale,
                                     quant_zero_point2=p3_reg_conv1_weight_zp,
                                     quant_scale3=p3_reg_conv1_act_scale,
                                     quant_zero_point3=p3_reg_conv1_act_zp, coe_name=p3_reg_conv1_coe_name,
                                     operator='conv')

        # ===================P3_out_reg_preds=================
        p3_reg_preds_weight_scale = np2tensor('../para/head.reg_preds.0.weight.scale.npy')
        p3_reg_preds_weight_zp = np2tensor('../para/head.reg_preds.0.weight.zero_point.npy')
        p3_reg_preds_act_scale = np2tensor('../para/head.reg_preds.0.scale.npy')
        p3_reg_preds_act_zp = np2tensor('../para/head.reg_preds.0.zero_point.npy')
        p3_reg_preds_coe_name = '../hand_coe/P3_reg_output.coe'
        self.p3_reg_preds = Conv2d_Q(quant_scale1=p3_reg_conv1_act_scale, quant_zero_point1=p3_reg_conv1_act_zp,
                                     quant_scale2=p3_reg_preds_weight_scale,
                                     quant_zero_point2=p3_reg_preds_weight_zp,
                                     quant_scale3=p3_reg_preds_act_scale,
                                     quant_zero_point3=p3_reg_preds_act_zp, coe_name=p3_reg_preds_coe_name,
                                     operator='conv2d')

        # =============== P3_out_obj_preds ==================

        p3_obj_preds_weight_scale = np2tensor('../para/head.obj_preds.0.weight.scale.npy')
        p3_obj_preds_weight_zp = np2tensor('../para/head.obj_preds.0.weight.zero_point.npy')
        p3_obj_preds_act_scale = np2tensor('../para/head.obj_preds.0.scale.npy')
        p3_obj_preds_act_zp = np2tensor('../para/head.obj_preds.0.zero_point.npy')
        p3_obj_preds_coe_name = '../hand_coe/P3_obj_output.coe'
        self.p3_obj_preds = Conv2d_Q(quant_scale1=p3_reg_conv1_act_scale, quant_zero_point1=p3_reg_conv1_act_zp,
                                     quant_scale2=p3_obj_preds_weight_scale,
                                     quant_zero_point2=p3_obj_preds_weight_zp,
                                     quant_scale3=p3_obj_preds_act_scale,
                                     quant_zero_point3=p3_obj_preds_act_zp, coe_name=p3_obj_preds_coe_name,
                                     operator='conv2d')

        # ==================================P3_out 结束 =====================================

        # ================P4_out======================
        c3_n3_csp_conv3_act_scale = np2tensor('../para/backbone.C3_n3.conv3.conv.scale.npy')
        c3_n3_csp_conv3_act_zp = np2tensor('../para/backbone.C3_n3.conv3.conv.zero_point.npy')
        p4_stem_conv_weight_scale = np2tensor('../para/head.stems.1.conv.weight.scale.npy')
        p4_stem_conv_weight_zp = np2tensor('../para/head.stems.1.conv.weight.zero_point.npy')
        p4_stem_conv_act_scale = np2tensor('../para/head.stems.1.conv.scale.npy')
        p4_stem_conv_act_zp = np2tensor('../para/head.stems.1.conv.zero_point.npy')
        p4_stem_coe_name = '../hand_coe/P4_stem.coe'
        self.p4_stem_conv = Conv2d_Q(quant_scale1=c3_n3_csp_conv3_act_scale, quant_zero_point1=c3_n3_csp_conv3_act_zp,
                                     quant_scale2=p4_stem_conv_weight_scale,
                                     quant_zero_point2=p4_stem_conv_weight_zp,
                                     quant_scale3=p4_stem_conv_act_scale,
                                     quant_zero_point3=p4_stem_conv_act_zp, coe_name=p4_stem_coe_name,
                                     operator='conv')

        # =================P4_cls_conv=============
        p4_cls_conv0_weight_scale = np2tensor('../para/head.cls_convs.1.0.conv.weight.scale.npy')
        p4_cls_conv0_weight_zp = np2tensor('../para/head.cls_convs.1.0.conv.weight.zero_point.npy')
        p4_cls_conv0_act_scale = np2tensor('../para/head.cls_convs.1.0.conv.scale.npy')
        p4_cls_conv0_act_zp = np2tensor('../para/head.cls_convs.1.0.conv.zero_point.npy')
        p4_cls_conv0_coe_name = '../hand_coe/P4_cls_conv0.coe'
        self.p4_cls_conv0 = Conv2d_Q(quant_scale1=p4_stem_conv_act_scale, quant_zero_point1=p4_stem_conv_act_zp,
                                     quant_scale2=p4_cls_conv0_weight_scale,
                                     quant_zero_point2=p4_cls_conv0_weight_zp,
                                     quant_scale3=p4_cls_conv0_act_scale,
                                     quant_zero_point3=p4_cls_conv0_act_zp, coe_name=p4_cls_conv0_coe_name,
                                     operator='conv')
        p4_cls_conv1_weight_scale = np2tensor('../para/head.cls_convs.1.1.conv.weight.scale.npy')
        p4_cls_conv1_weight_zp = np2tensor('../para/head.cls_convs.1.1.conv.weight.zero_point.npy')
        p4_cls_conv1_act_scale = np2tensor('../para/head.cls_convs.1.1.conv.scale.npy')
        p4_cls_conv1_act_zp = np2tensor('../para/head.cls_convs.1.1.conv.zero_point.npy')
        p4_cls_conv1_coe_name = '../hand_coe/P4_cls_feat.coe'
        self.p4_cls_conv1 = Conv2d_Q(quant_scale1=p4_cls_conv0_act_scale, quant_zero_point1=p4_cls_conv0_act_zp,
                                     quant_scale2=p4_cls_conv1_weight_scale,
                                     quant_zero_point2=p4_cls_conv1_weight_zp,
                                     quant_scale3=p4_cls_conv1_act_scale,
                                     quant_zero_point3=p4_cls_conv1_act_zp, coe_name=p4_cls_conv1_coe_name,
                                     operator='conv')

        # ===============P4_out_cls_preds==============
        p4_cls_preds_weight_scale = np2tensor('../para/head.cls_preds.1.weight.scale.npy')
        p4_cls_preds_weight_zp = np2tensor('../para/head.cls_preds.1.weight.zero_point.npy')
        p4_cls_preds_act_scale = np2tensor('../para/head.cls_preds.1.scale.npy')
        p4_cls_preds_act_zp = np2tensor('../para/head.cls_preds.1.zero_point.npy')
        p4_cls_preds_coe_name = '../hand_coe/P4_cls_output.coe'
        self.p4_cls_preds = Conv2d_Q(quant_scale1=p4_cls_conv1_act_scale, quant_zero_point1=p4_cls_conv1_act_zp,
                                     quant_scale2=p4_cls_preds_weight_scale,
                                     quant_zero_point2=p4_cls_preds_weight_zp,
                                     quant_scale3=p4_cls_preds_act_scale,
                                     quant_zero_point3=p4_cls_preds_act_zp, coe_name=p4_cls_preds_coe_name,
                                     operator='conv2d')

        # ===============P4_out_reg_convs==============

        p4_reg_conv0_weight_scale = np2tensor('../para/head.reg_convs.1.0.conv.weight.scale.npy')
        p4_reg_conv0_weight_zp = np2tensor('../para/head.reg_convs.1.0.conv.weight.zero_point.npy')
        p4_reg_conv0_act_scale = np2tensor('../para/head.reg_convs.1.0.conv.scale.npy')
        p4_reg_conv0_act_zp = np2tensor('../para/head.reg_convs.1.0.conv.zero_point.npy')
        p4_reg_conv0_coe_name = '../hand_coe/P4_reg_conv0.coe'
        self.p4_reg_conv0 = Conv2d_Q(quant_scale1=p4_stem_conv_act_scale, quant_zero_point1=p4_stem_conv_act_zp,
                                     quant_scale2=p4_reg_conv0_weight_scale,
                                     quant_zero_point2=p4_reg_conv0_weight_zp,
                                     quant_scale3=p4_reg_conv0_act_scale,
                                     quant_zero_point3=p4_reg_conv0_act_zp, coe_name=p4_reg_conv0_coe_name,
                                     operator='conv')
        p4_reg_conv1_weight_scale = np2tensor('../para/head.reg_convs.1.1.conv.weight.scale.npy')
        p4_reg_conv1_weight_zp = np2tensor('../para/head.reg_convs.1.1.conv.weight.zero_point.npy')
        p4_reg_conv1_act_scale = np2tensor('../para/head.reg_convs.1.1.conv.scale.npy')
        p4_reg_conv1_act_zp = np2tensor('../para/head.reg_convs.1.1.conv.zero_point.npy')
        p4_reg_conv1_coe_name = '../hand_coe/P4_reg_feat.coe'
        self.p4_reg_conv1 = Conv2d_Q(quant_scale1=p4_reg_conv0_act_scale, quant_zero_point1=p4_reg_conv0_act_zp,
                                     quant_scale2=p4_reg_conv1_weight_scale,
                                     quant_zero_point2=p4_reg_conv1_weight_zp,
                                     quant_scale3=p4_reg_conv1_act_scale,
                                     quant_zero_point3=p4_reg_conv1_act_zp, coe_name=p4_reg_conv1_coe_name,
                                     operator='conv')

        # ===================P4_out_reg_preds=================
        p4_reg_preds_weight_scale = np2tensor('../para/head.reg_preds.1.weight.scale.npy')
        p4_reg_preds_weight_zp = np2tensor('../para/head.reg_preds.1.weight.zero_point.npy')
        p4_reg_preds_act_scale = np2tensor('../para/head.reg_preds.1.scale.npy')
        p4_reg_preds_act_zp = np2tensor('../para/head.reg_preds.1.zero_point.npy')
        p4_reg_preds_coe_name = '../hand_coe/P4_reg_output.coe'
        self.p4_reg_preds = Conv2d_Q(quant_scale1=p4_reg_conv1_act_scale, quant_zero_point1=p4_reg_conv1_act_zp,
                                     quant_scale2=p4_reg_preds_weight_scale,
                                     quant_zero_point2=p4_reg_preds_weight_zp,
                                     quant_scale3=p4_reg_preds_act_scale,
                                     quant_zero_point3=p4_reg_preds_act_zp, coe_name=p4_reg_preds_coe_name,
                                     operator='conv2d')

        # =============== P4_out_obj_preds ==================

        p4_obj_preds_weight_scale = np2tensor('../para/head.obj_preds.1.weight.scale.npy')
        p4_obj_preds_weight_zp = np2tensor('../para/head.obj_preds.1.weight.zero_point.npy')
        p4_obj_preds_act_scale = np2tensor('../para/head.obj_preds.1.scale.npy')
        p4_obj_preds_act_zp = np2tensor('../para/head.obj_preds.1.zero_point.npy')
        p4_obj_preds_coe_name = '../hand_coe/P4_obj_output.coe'
        self.p4_obj_preds = Conv2d_Q(quant_scale1=p4_reg_conv1_act_scale, quant_zero_point1=p4_reg_conv1_act_zp,
                                     quant_scale2=p4_obj_preds_weight_scale,
                                     quant_zero_point2=p4_obj_preds_weight_zp,
                                     quant_scale3=p4_obj_preds_act_scale,
                                     quant_zero_point3=p4_obj_preds_act_zp, coe_name=p4_obj_preds_coe_name,
                                     operator='conv2d')

        # ================P5_out======================
        c3_n4_csp_conv3_act_scale = np2tensor('../para/backbone.C3_n4.conv3.conv.scale.npy')
        c3_n4_csp_conv3_act_zp = np2tensor('../para/backbone.C3_n4.conv3.conv.zero_point.npy')
        p5_stem_conv_weight_scale = np2tensor('../para/head.stems.2.conv.weight.scale.npy')
        p5_stem_conv_weight_zp = np2tensor('../para/head.stems.2.conv.weight.zero_point.npy')
        p5_stem_conv_act_scale = np2tensor('../para/head.stems.2.conv.scale.npy')
        p5_stem_conv_act_zp = np2tensor('../para/head.stems.2.conv.zero_point.npy')
        p5_stem_coe_name = '../hand_coe/P5_stem.coe'
        self.p5_stem_conv = Conv2d_Q(quant_scale1=c3_n4_csp_conv3_act_scale, quant_zero_point1=c3_n4_csp_conv3_act_zp,
                                     quant_scale2=p5_stem_conv_weight_scale,
                                     quant_zero_point2=p5_stem_conv_weight_zp,
                                     quant_scale3=p5_stem_conv_act_scale,
                                     quant_zero_point3=p5_stem_conv_act_zp, coe_name=p5_stem_coe_name,
                                     operator='conv')

        # =================P5_cls_conv=============
        p5_cls_conv0_weight_scale = np2tensor('../para/head.cls_convs.2.0.conv.weight.scale.npy')
        p5_cls_conv0_weight_zp = np2tensor('../para/head.cls_convs.2.0.conv.weight.zero_point.npy')
        p5_cls_conv0_act_scale = np2tensor('../para/head.cls_convs.2.0.conv.scale.npy')
        p5_cls_conv0_act_zp = np2tensor('../para/head.cls_convs.2.0.conv.zero_point.npy')
        p5_cls_conv0_coe_name = '../hand_coe/P5_cls_conv0.coe'
        self.p5_cls_conv0 = Conv2d_Q(quant_scale1=p5_stem_conv_act_scale, quant_zero_point1=p5_stem_conv_act_zp,
                                     quant_scale2=p5_cls_conv0_weight_scale,
                                     quant_zero_point2=p5_cls_conv0_weight_zp,
                                     quant_scale3=p5_cls_conv0_act_scale,
                                     quant_zero_point3=p5_cls_conv0_act_zp, coe_name=p5_cls_conv0_coe_name,
                                     operator='conv')
        p5_cls_conv1_weight_scale = np2tensor('../para/head.cls_convs.2.1.conv.weight.scale.npy')
        p5_cls_conv1_weight_zp = np2tensor('../para/head.cls_convs.2.1.conv.weight.zero_point.npy')
        p5_cls_conv1_act_scale = np2tensor('../para/head.cls_convs.2.1.conv.scale.npy')
        p5_cls_conv1_act_zp = np2tensor('../para/head.cls_convs.2.1.conv.zero_point.npy')
        p5_cls_conv1_coe_name = '../hand_coe/P5_cls_feat.coe'
        self.p5_cls_conv1 = Conv2d_Q(quant_scale1=p5_cls_conv0_act_scale, quant_zero_point1=p5_cls_conv0_act_zp,
                                     quant_scale2=p5_cls_conv1_weight_scale,
                                     quant_zero_point2=p5_cls_conv1_weight_zp,
                                     quant_scale3=p5_cls_conv1_act_scale,
                                     quant_zero_point3=p5_cls_conv1_act_zp, coe_name=p5_cls_conv1_coe_name,
                                     operator='conv')

        # ===============P5_out_cls_preds==============
        p5_cls_preds_weight_scale = np2tensor('../para/head.cls_preds.2.weight.scale.npy')
        p5_cls_preds_weight_zp = np2tensor('../para/head.cls_preds.2.weight.zero_point.npy')
        p5_cls_preds_act_scale = np2tensor('../para/head.cls_preds.2.scale.npy')
        p5_cls_preds_act_zp = np2tensor('../para/head.cls_preds.2.zero_point.npy')
        p5_cls_preds_coe_name = '../hand_coe/P5_cls_output.coe'
        self.p5_cls_preds = Conv2d_Q(quant_scale1=p5_cls_conv1_act_scale, quant_zero_point1=p5_cls_conv1_act_zp,
                                     quant_scale2=p5_cls_preds_weight_scale,
                                     quant_zero_point2=p5_cls_preds_weight_zp,
                                     quant_scale3=p5_cls_preds_act_scale,
                                     quant_zero_point3=p5_cls_preds_act_zp, coe_name=p5_cls_preds_coe_name,
                                     operator='conv2d')

        # ===============P5_out_reg_convs==============

        p5_reg_conv0_weight_scale = np2tensor('../para/head.reg_convs.2.0.conv.weight.scale.npy')
        p5_reg_conv0_weight_zp = np2tensor('../para/head.reg_convs.2.0.conv.weight.zero_point.npy')
        p5_reg_conv0_act_scale = np2tensor('../para/head.reg_convs.2.0.conv.scale.npy')
        p5_reg_conv0_act_zp = np2tensor('../para/head.reg_convs.2.0.conv.zero_point.npy')
        p5_reg_conv0_coe_name = '../hand_coe/P5_reg_conv0.coe'
        self.p5_reg_conv0 = Conv2d_Q(quant_scale1=p5_stem_conv_act_scale, quant_zero_point1=p5_stem_conv_act_zp,
                                     quant_scale2=p5_reg_conv0_weight_scale,
                                     quant_zero_point2=p5_reg_conv0_weight_zp,
                                     quant_scale3=p5_reg_conv0_act_scale,
                                     quant_zero_point3=p5_reg_conv0_act_zp, coe_name=p5_reg_conv0_coe_name,
                                     operator='conv')
        p5_reg_conv1_weight_scale = np2tensor('../para/head.reg_convs.2.1.conv.weight.scale.npy')
        p5_reg_conv1_weight_zp = np2tensor('../para/head.reg_convs.2.1.conv.weight.zero_point.npy')
        p5_reg_conv1_act_scale = np2tensor('../para/head.reg_convs.2.1.conv.scale.npy')
        p5_reg_conv1_act_zp = np2tensor('../para/head.reg_convs.2.1.conv.zero_point.npy')
        p5_reg_conv1_coe_name = '../hand_coe/P5_reg_feat.coe'
        self.p5_reg_conv1 = Conv2d_Q(quant_scale1=p5_reg_conv0_act_scale, quant_zero_point1=p5_reg_conv0_act_zp,
                                     quant_scale2=p5_reg_conv1_weight_scale,
                                     quant_zero_point2=p5_reg_conv1_weight_zp,
                                     quant_scale3=p5_reg_conv1_act_scale,
                                     quant_zero_point3=p5_reg_conv1_act_zp, coe_name=p5_reg_conv1_coe_name,
                                     operator='conv')

        # ===================P5_out_reg_preds=================
        p5_reg_preds_weight_scale = np2tensor('../para/head.reg_preds.2.weight.scale.npy')
        p5_reg_preds_weight_zp = np2tensor('../para/head.reg_preds.2.weight.zero_point.npy')
        p5_reg_preds_act_scale = np2tensor('../para/head.reg_preds.2.scale.npy')
        p5_reg_preds_act_zp = np2tensor('../para/head.reg_preds.2.zero_point.npy')
        p5_reg_preds_coe_name = '../hand_coe/P5_reg_output.coe'
        self.p5_reg_preds = Conv2d_Q(quant_scale1=p5_reg_conv1_act_scale, quant_zero_point1=p5_reg_conv1_act_zp,
                                     quant_scale2=p5_reg_preds_weight_scale,
                                     quant_zero_point2=p5_reg_preds_weight_zp,
                                     quant_scale3=p5_reg_preds_act_scale,
                                     quant_zero_point3=p5_reg_preds_act_zp, coe_name=p5_reg_preds_coe_name,
                                     operator='conv2d')

        # =============== P5_out_obj_preds ==================

        p5_obj_preds_weight_scale = np2tensor('../para/head.obj_preds.2.weight.scale.npy')
        p5_obj_preds_weight_zp = np2tensor('../para/head.obj_preds.2.weight.zero_point.npy')
        p5_obj_preds_act_scale = np2tensor('../para/head.obj_preds.2.scale.npy')
        p5_obj_preds_act_zp = np2tensor('../para/head.obj_preds.2.zero_point.npy')
        p5_obj_preds_coe_name = '../hand_coe/P5_obj_output.coe'
        self.p5_obj_preds = Conv2d_Q(quant_scale1=p5_reg_conv1_act_scale, quant_zero_point1=p5_reg_conv1_act_zp,
                                     quant_scale2=p5_obj_preds_weight_scale,
                                     quant_zero_point2=p5_obj_preds_weight_zp,
                                     quant_scale3=p5_obj_preds_act_scale,
                                     quant_zero_point3=p5_obj_preds_act_zp, coe_name=p5_obj_preds_coe_name,
                                     operator='conv2d')

    def forward(self):
        yolox_quant_pth = '../yolox_quant_pth523/Epoch2-yolox_quantization_post.pth'
        model = torch.jit.load(yolox_quant_pth)
        model.eval()
        CSPDarknet = model.backbone.backbone
        stem = CSPDarknet.stem
        focus_csp = stem.csp0
        # ============dark2 模块 主要使用里面的add 和cat操作===============
        # dark2 = CSPDarknet.dark2
        # dark2_list = list(dark2.children())
        # dark2_CSP = dark2_list[1]
        # dark2_CSP_csp_cat1 = dark2_CSP.csp1
        # dark2_CSP_m = dark2_CSP.m
        # dark2_CSP_m_Bottleneck = list(dark2_CSP_m.children())[0]
        # dark2_CSP_m_Bottleneck_add = dark2_CSP_m_Bottleneck.csp
        #
        # # ============dark3 模块 主要使用里面的add 和cat操作===============
        # dark3 = CSPDarknet.dark3
        # dark3_list = list(dark3.children())
        # dark3_CSP = dark3_list[1]
        # dark3_CSP_csp_cat1 = dark3_CSP.csp1
        # dark3_CSP_m = dark3_CSP.m
        # dark3_CSP_m_Bottleneck_list = list(dark3_CSP_m.children())
        # dark3_CSP_m_Bottleneck_item0 = dark3_CSP_m_Bottleneck_list[0]
        # dark3_Bottleneck_item0_add = dark3_CSP_m_Bottleneck_item0.csp
        # dark3_CSP_m_Bottleneck_item1 = dark3_CSP_m_Bottleneck_list[1]
        # dark3_Bottleneck_item1_add = dark3_CSP_m_Bottleneck_item1.csp
        # dark3_CSP_m_Bottleneck_item2 = dark3_CSP_m_Bottleneck_list[2]
        # dark3_Bottleneck_item2_add = dark3_CSP_m_Bottleneck_item2.csp
        #
        # # ============dark4 模块 主要使用里面的add 和cat操作===============
        # dark4 = CSPDarknet.dark4
        # dark4_list = list(dark4.children())
        # dark4_CSP = dark4_list[1]
        # dark4_CSP_csp_cat1 = dark4_CSP.csp1
        # dark4_CSP_m = dark4_CSP.m
        # dark4_CSP_m_Bottleneck_list = list(dark4_CSP_m.children())
        # dark4_CSP_m_Bottleneck_item0 = dark4_CSP_m_Bottleneck_list[0]
        # dark4_Bottleneck_item0_add = dark4_CSP_m_Bottleneck_item0.csp
        # dark4_CSP_m_Bottleneck_item1 = dark4_CSP_m_Bottleneck_list[1]
        # dark4_Bottleneck_item1_add = dark4_CSP_m_Bottleneck_item1.csp
        # dark4_CSP_m_Bottleneck_item2 = dark4_CSP_m_Bottleneck_list[2]
        # dark4_Bottleneck_item2_add = dark4_CSP_m_Bottleneck_item2.csp
        #
        # # ============dark5 模块 主要使用里面的add 和cat操作===============
        # dark5 = CSPDarknet.dark5
        # dark5_list = list(dark5.children())
        # dark5_CSP = dark5_list[1]
        # dark5_CSP_csp_cat1 = dark5_CSP.csp1
        # dark5_CSP_m = dark5_CSP.m
        # dark5_CSP_m_Bottleneck_list = list(dark5_CSP_m.children())
        # dark5_CSP_m_Bottleneck_item0 = dark5_CSP_m_Bottleneck_list[0]
        # dark5_Bottleneck_item0_add = dark5_CSP_m_Bottleneck_item0.csp
        #
        # YOLOPAFPN = model.backbone
        # upsample = YOLOPAFPN.upsample
        # YOLOPAFPN_csp2_cat = YOLOPAFPN.csp2
        # YOLOPAFPN_csp3_cat = YOLOPAFPN.csp3
        # YOLOPAFPN_csp4_cat = YOLOPAFPN.csp4
        # YOLOPAFPN_csp5_cat = YOLOPAFPN.csp5
        #
        # # ===========C3_P4===================
        # C3_P4 = YOLOPAFPN.C3_p4
        # C3_P4_CSP_cat = C3_P4.csp1
        # C3_P4_m = C3_P4.m
        # C3_P4_m_Bottleneck_list = list(C3_P4_m.children())
        # C3_P4_m_Bottleneck_item0 = C3_P4_m_Bottleneck_list[0]
        # C3_P4_m_Bottleneck_item0_add = C3_P4_m_Bottleneck_item0.csp
        #
        # # ==============C3_P3===================
        # C3_P3 = YOLOPAFPN.C3_p3
        # C3_P3_CSP_cat = C3_P3.csp1
        # C3_P3_m = C3_P3.m
        # C3_P3_m_Bottleneck_list = list(C3_P3_m.children())
        # C3_P3_m_Bottleneck_item0 = C3_P3_m_Bottleneck_list[0]
        # C3_P3_m_Bottleneck_item0_add = C3_P3_m_Bottleneck_item0.csp
        #
        # # ===========start C3_n3===========
        # C3_n3 = YOLOPAFPN.C3_n3
        # C3_n3_CSP_cat = C3_n3.csp1
        # C3_n3_m = C3_n3.m
        # C3_n3_m_Bottleneck_list = list(C3_n3_m.children())
        # C3_n3_m_Bottleneck_item0 = C3_n3_m_Bottleneck_list[0]
        # C3_n3_m_Bottleneck_item0_add = C3_n3_m_Bottleneck_item0.csp
        #
        # # ===========start C3_n4===========
        # C3_n4 = YOLOPAFPN.C3_n4
        # C3_n4_CSP_cat = C3_n4.csp1
        # C3_n4_m = C3_n4.m
        # C3_n4_m_Bottleneck_list = list(C3_n4_m.children())
        # C3_n4_m_Bottleneck_item0 = C3_n4_m_Bottleneck_list[0]
        # C3_n4_m_Bottleneck_item0_add = C3_n4_m_Bottleneck_item0.csp
        #
        # head = model.head
        # head_cat = head.csp6
        # dequant = model.dequant

        with torch.no_grad():
            img = picture_load(self.img_path)
            img_feature = model.quant(img)
            patch_top_left, patch_bot_left, patch_top_right, patch_bot_right = focus(img_feature)

            focus_conv_quant_weight = np2tensor('../para/backbone.backbone.stem.conv.conv.weight.int.npy')
            focus_conv_bias = np2tensor('../para/backbone.backbone.stem.conv.conv.bias.npy')
            focus_cat_feature = focus_csp.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,),
                                              dim=1, )

            # cat focus_conv s=1,p=1
            focus_conv_feature = self.focus_conv(focus_cat_feature.int_repr(), focus_conv_quant_weight, focus_conv_bias,
                                                 out_hand=1,
                                                 write_coe_file=0, stride=1, padding=1)

            # BaseConv  s=2 p=1
            dark2_BaseConv_quant_weight = np2tensor('../para/backbone.backbone.dark2.0.conv.weight.int.npy')
            dark2_BaseConv_bias = np2tensor('../para/backbone.backbone.dark2.0.conv.bias.npy')

            dark2_BaseConv_feature = self.dark2_baseconv(focus_conv_feature, dark2_BaseConv_quant_weight,
                                                         dark2_BaseConv_bias, out_hand=1, write_coe_file=0, stride=2,
                                                         padding=1)

            # ================dark2 csp======================
            # csp_conv1 s=1 p=0
            dark2_csp_conv1_quant_weight = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.weight.int.npy')
            dark2_csp_conv1_bias = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.bias.npy')
            dark2_csp_conv1_feature = self.dark2_csp_conv1(dark2_BaseConv_feature.numpy(), dark2_csp_conv1_quant_weight,
                                                           dark2_csp_conv1_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)

            # csp_conv2 s=1 p=0
            dark2_csp_conv2_quant_weight = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.weight.int.npy')
            dark2_csp_conv2_bias = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.bias.npy')
            dark2_csp_conv2_feature = self.dark2_csp_conv2(dark2_BaseConv_feature.numpy(), dark2_csp_conv2_quant_weight,
                                                           dark2_csp_conv2_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)

            # dark2_csp_m_conv1 s=1 p=0 输入为csp_conv1
            dark2_csp_m_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark2.1.m.0.conv1.conv.weight.int.npy')
            dark2_csp_m_conv1_bias = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv1.conv.bias.npy')
            dark2_csp_m_conv1_feature = self.dark2_csp_m_conv1(dark2_csp_conv1_feature, dark2_csp_m_conv1_quant_weight,
                                                               dark2_csp_m_conv1_bias, out_hand=1, write_coe_file=0,
                                                               stride=1, padding=0)
            # dark2_csp_m_conv2 s=1 p=1
            dark2_csp_m_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark2.1.m.0.conv2.conv.weight.int.npy')
            dark2_csp_m_conv2_bias = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv2.conv.bias.npy')
            dark2_csp_m_conv2_feature = self.dark2_csp_m_conv2(dark2_csp_m_conv1_feature,
                                                               dark2_csp_m_conv2_quant_weight, dark2_csp_m_conv2_bias,
                                                               out_hand=1, write_coe_file=0, stride=1, padding=1)
            # exit()
            # dark2_csp_m_add  将m的输入和输出相加 输入为csp_conv1 输出为 m_conv2
            dark2_csp_conv1_act_scale = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.scale.npy')
            dark2_csp_conv1_act_zp = np2tensor('../para/backbone.backbone.dark2.1.conv1.conv.zero_point.npy')
            dark2_csp_m_conv2_act_scale = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv2.conv.scale.npy')
            dark2_csp_m_conv2_act_zp = np2tensor('../para/backbone.backbone.dark2.1.m.0.conv2.conv.zero_point.npy')
            dark2_csp_add_scale = np2tensor('../para/backbone.backbone.dark2.1.m.0.csp.scale.npy')
            dark2_csp_add_zp = np2tensor('../para/backbone.backbone.dark2.1.m.0.csp.zero_point.npy')

            dark2_csp_m_add_feature = quant_add(dark2_csp_conv1_feature, dark2_csp_m_conv2_feature,
                                                dark2_csp_conv1_act_scale, dark2_csp_m_conv2_act_scale,
                                                dark2_csp_add_scale, dark2_csp_conv1_act_zp, dark2_csp_m_conv2_act_zp,
                                                dark2_csp_add_zp)

            # gen_coe('../hand_coe/dark2_CSP_m_add.coe', dark2_csp_m_add_feature)
            # exit()
            # dark2_csp_m_add_feature = dark2_CSP_m_Bottleneck_add.add(dark2_csp_conv1_feature, dark2_csp_m_conv2_feature)

            #  cat

            dark2_csp_add_scale = np2tensor('../para/backbone.backbone.dark2.1.m.0.csp.scale.npy')
            dark2_csp_add_zp = np2tensor('../para/backbone.backbone.dark2.1.m.0.csp.zero_point.npy')
            dark2_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.scale.npy')
            dark2_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark2.1.conv2.conv.zero_point.npy')
            dark2_csp_cat_scale = np2tensor('../para/backbone.backbone.dark2.1.csp1.scale.npy')
            dark2_csp_cat_zp = np2tensor('../para/backbone.backbone.dark2.1.csp1.zero_point.npy')
            dark2_csp_cat_feature = quant_cat(dark2_csp_m_add_feature, dark2_csp_conv2_feature, dark2_csp_add_scale,
                                              dark2_csp_conv2_act_scale, dark2_csp_cat_scale, dark2_csp_add_zp,
                                              dark2_csp_conv2_act_zp, dark2_csp_cat_zp)
            # gen_coe('../hand_coe/dark2_CSP_cat.coe', dark2_csp_cat_feature)

            # cat结束之后 进行 conv3 输入为cat之后的值

            dark2_csp_conv3_bias = np2tensor('../para/backbone.backbone.dark2.1.conv3.conv.bias.npy')
            dark2_csp_conv3_quant_weight = np2tensor('../para/backbone.backbone.dark2.1.conv3.conv.weight.int.npy')
            dark2_csp_conv3_feature = self.dark2_csp_conv3(dark2_csp_cat_feature, dark2_csp_conv3_quant_weight,
                                                           dark2_csp_conv3_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)
            # exit()

            # =========================start dark3============================
            dark3_BaseConv_quant_weight = np2tensor('../para/backbone.backbone.dark3.0.conv.weight.int.npy')
            dark3_BaseConv_bias = np2tensor('../para/backbone.backbone.dark3.0.conv.bias.npy')
            dark3_BaseConv_feature = self.dark3_baseconv(dark2_csp_conv3_feature, dark3_BaseConv_quant_weight,
                                                         dark3_BaseConv_bias, out_hand=1, write_coe_file=0, stride=2,
                                                         padding=1)

            # =================dark3 csplayer =================

            # csp_conv1 s=1 p=0
            dark3_csp_conv1_quant_weight = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.weight.int.npy')
            dark3_csp_conv1_bias = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.bias.npy')
            dark3_csp_conv1_feature = self.dark3_csp_conv1(dark3_BaseConv_feature, dark3_csp_conv1_quant_weight,
                                                           dark3_csp_conv1_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)
            # csp_conv2 s=1 p=0
            dark3_csp_conv2_quant_weight = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.weight.int.npy')
            dark3_csp_conv2_bias = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.bias.npy')
            dark3_csp_conv2_feature = self.dark3_csp_conv2(dark3_BaseConv_feature, dark3_csp_conv2_quant_weight,
                                                           dark3_csp_conv2_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)
            # exit()
            # =================dark3 csp m =================
            # m0
            # m0 conv1 s=1 p=0

            dark3_csp_m0_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark3.1.m.0.conv1.conv.weight.int.npy')
            dark3_csp_m0_conv1_bias = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv1.conv.bias.npy')
            dark3_csp_m0_conv1_feature = self.dark3_csp_m0_conv1(dark3_csp_conv1_feature,
                                                                 dark3_csp_m0_conv1_quant_weight,
                                                                 dark3_csp_m0_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # m0 conv2 s=1 p=1
            dark3_csp_m0_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark3.1.m.0.conv2.conv.weight.int.npy')
            dark3_csp_m0_conv2_bias = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv2.conv.bias.npy')
            dark3_csp_m0_conv2_feature = self.dark3_csp_m0_conv2(dark3_csp_m0_conv1_feature,
                                                                 dark3_csp_m0_conv2_quant_weight,
                                                                 dark3_csp_m0_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)

            # dark3_csp_m0_add  将m0的输入和输出相加 输入为csp_conv1 输出为 m0_conv2
            dark3_csp_conv1_act_scale = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.scale.npy')
            dark3_csp_conv1_act_zp = np2tensor('../para/backbone.backbone.dark3.1.conv1.conv.zero_point.npy')
            dark3_csp_m0_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv2.conv.scale.npy')
            dark3_csp_m0_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.0.conv2.conv.zero_point.npy')
            dark3_csp_m0_add_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.csp.scale.npy')
            dark3_csp_m0_add_zp = np2tensor('../para/backbone.backbone.dark3.1.m.0.csp.zero_point.npy')
            dark3_csp_m0_add_feature = quant_add(dark3_csp_conv1_feature, dark3_csp_m0_conv2_feature,
                                                 dark3_csp_conv1_act_scale, dark3_csp_m0_conv2_act_scale,
                                                 dark3_csp_m0_add_scale, dark3_csp_conv1_act_zp,
                                                 dark3_csp_m0_conv2_act_zp, dark3_csp_m0_add_zp)
            # gen_coe('../hand_coe/dark3_CSP_m0_add.coe', dark3_csp_m0_add_feature)

            # m1
            # m0 add之后的结果作为 m1的输入
            # m1 conv1  s=1 p=0
            dark3_csp_m1_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark3.1.m.1.conv1.conv.weight.int.npy')
            dark3_csp_m1_conv1_bias = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv1.conv.bias.npy')
            dark3_csp_m1_conv1_feature = self.dark3_csp_m1_conv1(dark3_csp_m0_add_feature,
                                                                 dark3_csp_m1_conv1_quant_weight,
                                                                 dark3_csp_m1_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # m1 conv2  s=1 p=1
            dark3_csp_m1_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark3.1.m.1.conv2.conv.weight.int.npy')
            dark3_csp_m1_conv2_bias = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv2.conv.bias.npy')
            dark3_csp_m1_conv2_feature = self.dark3_csp_m1_conv2(dark3_csp_m1_conv1_feature,
                                                                 dark3_csp_m1_conv2_quant_weight,
                                                                 dark3_csp_m1_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)

            # dark_csp_m1_add  将m1的输入dark3_csp_m0_add_feature 和 dark3_csp_m1_conv2_feature 相加
            dark3_csp_m0_add_scale = np2tensor('../para/backbone.backbone.dark3.1.m.0.csp.scale.npy')
            dark3_csp_m0_add_zp = np2tensor('../para/backbone.backbone.dark3.1.m.0.csp.zero_point.npy')
            dark3_csp_m1_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv2.conv.scale.npy')
            dark3_csp_m1_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.1.conv2.conv.zero_point.npy')
            dark3_csp_m1_add_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.csp.scale.npy')
            dark3_csp_m1_add_zp = np2tensor('../para/backbone.backbone.dark3.1.m.1.csp.zero_point.npy')
            dark3_csp_m1_add_feature = quant_add(dark3_csp_m0_add_feature, dark3_csp_m1_conv2_feature,
                                                 dark3_csp_m0_add_scale, dark3_csp_m1_conv2_act_scale,
                                                 dark3_csp_m1_add_scale, dark3_csp_m0_add_zp,
                                                 dark3_csp_m1_conv2_act_zp, dark3_csp_m1_add_zp)
            # gen_coe('../hand_coe/dark3_CSP_m1_add.coe', dark3_csp_m1_add_feature)
            # exit()
            # dark3_csp_m1_add_feature = dark3_Bottleneck_item1_add.add(dark3_csp_m0_add_feature,
            #                                                           dark3_csp_m1_conv2_feature)

            # m2    m1 add之后的输出作为m2的输入
            # m2 conv1 s=1 p=0
            dark3_csp_m2_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark3.1.m.2.conv1.conv.weight.int.npy')
            dark3_csp_m2_conv1_bias = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv1.conv.bias.npy')
            dark3_csp_m2_conv1_feature = self.dark3_csp_m2_conv1(dark3_csp_m1_add_feature,
                                                                 dark3_csp_m2_conv1_quant_weight,
                                                                 dark3_csp_m2_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)

            # m2 conv2 s=1 p=1
            dark3_csp_m2_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark3.1.m.2.conv2.conv.weight.int.npy')
            dark3_csp_m2_conv2_bias = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv2.conv.bias.npy')
            dark3_csp_m2_conv2_feature = self.dark3_csp_m2_conv2(dark3_csp_m2_conv1_feature,
                                                                 dark3_csp_m2_conv2_quant_weight,
                                                                 dark3_csp_m2_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)
            # m2 add conv1的输入dark3_csp_m1_add_feature 和 conv2的输出 dark3_csp_m1_conv2_feature相加
            dark3_csp_m1_add_scale = np2tensor('../para/backbone.backbone.dark3.1.m.1.csp.scale.npy')
            dark3_csp_m1_add_zp = np2tensor('../para/backbone.backbone.dark3.1.m.1.csp.zero_point.npy')
            dark3_csp_m2_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv2.conv.scale.npy')
            dark3_csp_m2_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.m.2.conv2.conv.zero_point.npy')
            dark3_csp_m2_add_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.csp.scale.npy')
            dark3_csp_m2_add_zp = np2tensor('../para/backbone.backbone.dark3.1.m.2.csp.zero_point.npy')
            dark3_csp_m2_add_feature = quant_add(dark3_csp_m1_add_feature, dark3_csp_m2_conv2_feature,
                                                 dark3_csp_m1_add_scale, dark3_csp_m2_conv2_act_scale,
                                                 dark3_csp_m2_add_scale, dark3_csp_m1_add_zp, dark3_csp_m2_conv2_act_zp,
                                                 dark3_csp_m2_add_zp)
            # gen_coe('../hand_coe/dark3_CSP_m2_add.coe', dark3_csp_m2_add_feature)
            # exit()

            # dark3 csp cat 将m的最终输出dark3_csp_m2_add_feature和conv2的输出dark3_csp_conv2_feature进行cat
            dark3_csp_m2_add_scale = np2tensor('../para/backbone.backbone.dark3.1.m.2.csp.scale.npy')
            dark3_csp_m2_add_zp = np2tensor('../para/backbone.backbone.dark3.1.m.2.csp.zero_point.npy')
            dark3_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.scale.npy')
            dark3_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark3.1.conv2.conv.zero_point.npy')
            dark3_csp_cat_scale = np2tensor('../para/backbone.backbone.dark3.1.csp1.scale.npy')
            dark3_csp_cat_zp = np2tensor('../para/backbone.backbone.dark3.1.csp1.zero_point.npy')
            dark3_csp_cat_feature = quant_cat(dark3_csp_m2_add_feature, dark3_csp_conv2_feature, dark3_csp_m2_add_scale,
                                              dark3_csp_conv2_act_scale, dark3_csp_cat_scale, dark3_csp_m2_add_zp,
                                              dark3_csp_conv2_act_zp, dark3_csp_cat_zp)

            # gen_coe('../hand_coe/dark3_CSP_cat.coe', dark3_csp_cat_feature)
            # dark3 csp conv3 s=1 p=0  cat之后的输出作为conv3的输入
            dark3_csp_conv3_quant_weight = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.weight.int.npy')
            dark3_csp_conv3_bias = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.bias.npy')
            dark3_csp_conv3_feature = self.dark3_csp_conv3(dark3_csp_cat_feature, dark3_csp_conv3_quant_weight,
                                                           dark3_csp_conv3_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)
            # exit()

            # =========================start dark4============================
            # BaseConv s=2 p=1
            dark4_BaseConv_quant_weight = np2tensor('../para/backbone.backbone.dark4.0.conv.weight.int.npy')
            dark4_BaseConv_bias = np2tensor('../para/backbone.backbone.dark4.0.conv.bias.npy')
            dark4_BaseConv_feature = self.dark4_baseconv(dark3_csp_conv3_feature, dark4_BaseConv_quant_weight,
                                                         dark4_BaseConv_bias, out_hand=1, write_coe_file=0, stride=2,
                                                         padding=1)
            # =================dark4 csp=================

            # csp_conv1 s=1 p=0
            dark4_csp_conv1_quant_weight = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.weight.int.npy')
            dark4_csp_conv1_bias = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.bias.npy')
            dark4_csp_conv1_feature = self.dark4_csp_conv1(dark4_BaseConv_feature, dark4_csp_conv1_quant_weight,
                                                           dark4_csp_conv1_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)
            # csp_conv2 s=1 p=0
            dark4_csp_conv2_quant_weight = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.weight.int.npy')
            dark4_csp_conv2_bias = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.bias.npy')
            dark4_csp_conv2_feature = self.dark4_csp_conv2(dark4_BaseConv_feature, dark4_csp_conv2_quant_weight,
                                                           dark4_csp_conv2_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)
            # =================dark4 csp m =================
            # m0
            # m0 conv1 s=1 p=0

            dark4_csp_m0_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark4.1.m.0.conv1.conv.weight.int.npy')
            dark4_csp_m0_conv1_bias = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv1.conv.bias.npy')
            dark4_csp_m0_conv1_feature = self.dark4_csp_m0_conv1(dark4_csp_conv1_feature,
                                                                 dark4_csp_m0_conv1_quant_weight,
                                                                 dark4_csp_m0_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # m0 conv2 s=1 p=1
            dark4_csp_m0_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark4.1.m.0.conv2.conv.weight.int.npy')
            dark4_csp_m0_conv2_bias = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv2.conv.bias.npy')
            dark4_csp_m0_conv2_feature = self.dark4_csp_m0_conv2(dark4_csp_m0_conv1_feature,
                                                                 dark4_csp_m0_conv2_quant_weight,
                                                                 dark4_csp_m0_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)
            # dark4_csp_m0_add  将m0的输入和输出相加 输入为csp_conv1 输出为 m0_conv2
            dark4_csp_conv1_act_scale = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.scale.npy')
            dark4_csp_conv1_act_zp = np2tensor('../para/backbone.backbone.dark4.1.conv1.conv.zero_point.npy')
            dark4_csp_m0_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv2.conv.scale.npy')
            dark4_csp_m0_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.0.conv2.conv.zero_point.npy')
            dark4_csp_add0_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.csp.scale.npy')
            dark4_csp_add0_zp = np2tensor('../para/backbone.backbone.dark4.1.m.0.csp.zero_point.npy')

            dark4_csp_m0_add_feature = quant_add(dark4_csp_conv1_feature, dark4_csp_m0_conv2_feature,
                                                 dark4_csp_conv1_act_scale, dark4_csp_m0_conv2_act_scale,
                                                 dark4_csp_add0_scale, dark4_csp_conv1_act_zp,
                                                 dark4_csp_m0_conv2_act_zp, dark4_csp_add0_zp)

            # dark4_csp_m0_add_feature = dark4_Bottleneck_item0_add.add(dark4_csp_conv1_feature,
            #                                                           dark4_csp_m0_conv2_feature)
            # gen_coe('../hand_coe/dark4_CSP_m0_add.coe', dark4_csp_m0_add_feature)

            # m1
            # m0 add之后的结果作为 m1的输入
            # m1 conv1  s=1 p=0
            dark4_csp_m1_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark4.1.m.1.conv1.conv.weight.int.npy')
            dark4_csp_m1_conv1_bias = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv1.conv.bias.npy')
            dark4_csp_m1_conv1_feature = self.dark4_csp_m1_conv1(dark4_csp_m0_add_feature,
                                                                 dark4_csp_m1_conv1_quant_weight,
                                                                 dark4_csp_m1_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # m1 conv2  s=1 p=1
            dark4_csp_m1_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark4.1.m.1.conv2.conv.weight.int.npy')
            dark4_csp_m1_conv2_bias = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv2.conv.bias.npy')
            dark4_csp_m1_conv2_feature = self.dark4_csp_m1_conv2(dark4_csp_m1_conv1_feature,
                                                                 dark4_csp_m1_conv2_quant_weight,
                                                                 dark4_csp_m1_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)

            # dark4_csp_m1_add  将m1的输入dark4_csp_m0_add_feature 和 dark4_csp_m1_conv2_feature 相加
            dark4_csp_add0_scale = np2tensor('../para/backbone.backbone.dark4.1.m.0.csp.scale.npy')
            dark4_csp_add0_zp = np2tensor('../para/backbone.backbone.dark4.1.m.0.csp.zero_point.npy')
            dark4_csp_m1_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv2.conv.scale.npy')
            dark4_csp_m1_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.1.conv2.conv.zero_point.npy')
            dark4_csp_add1_scale = np2tensor('../para/backbone.backbone.dark4.1.m.1.csp.scale.npy')
            dark4_csp_add1_zp = np2tensor('../para/backbone.backbone.dark4.1.m.1.csp.zero_point.npy')

            dark4_csp_m1_add_feature = quant_add(dark4_csp_m0_add_feature, dark4_csp_m1_conv2_feature,
                                                 dark4_csp_add0_scale, dark4_csp_m1_conv2_act_scale,
                                                 dark4_csp_add1_scale, dark4_csp_add0_zp, dark4_csp_m1_conv2_act_zp,
                                                 dark4_csp_add1_zp)

            # dark4_csp_m1_add_feature = dark4_Bottleneck_item1_add.add(dark4_csp_m0_add_feature,
            #                                                           dark4_csp_m1_conv2_feature)
            # gen_coe('../hand_coe/dark4_CSP_m1_add.coe', dark4_csp_m1_add_feature)

            # m2    m1 add之后的输出作为m2的输入
            # m2 conv1 s=1 p=0
            dark4_csp_m2_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark4.1.m.2.conv1.conv.weight.int.npy')
            dark4_csp_m2_conv1_bias = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv1.conv.bias.npy')
            dark4_csp_m2_conv1_feature = self.dark4_csp_m2_conv1(dark4_csp_m1_add_feature,
                                                                 dark4_csp_m2_conv1_quant_weight,
                                                                 dark4_csp_m2_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)

            # m2 conv2 s=1 p=1
            dark4_csp_m2_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark4.1.m.2.conv2.conv.weight.int.npy')
            dark4_csp_m2_conv2_bias = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv2.conv.bias.npy')
            dark4_csp_m2_conv2_feature = self.dark4_csp_m2_conv2(dark4_csp_m2_conv1_feature,
                                                                 dark4_csp_m2_conv2_quant_weight,
                                                                 dark4_csp_m2_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)
            # m2 add conv1的输入dark4_csp_m1_add_feature 和 conv2的输出 dark4_csp_m1_conv2_feature相加

            dark4_csp_m2_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv2.conv.scale.npy')
            dark4_csp_m2_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.m.2.conv2.conv.zero_point.npy')
            dark4_csp_add2_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.csp.scale.npy')
            dark4_csp_add2_zp = np2tensor('../para/backbone.backbone.dark4.1.m.2.csp.zero_point.npy')
            dark4_csp_m2_add_feature = quant_add(dark4_csp_m1_add_feature, dark4_csp_m2_conv2_feature,
                                                 dark4_csp_add1_scale, dark4_csp_m2_conv2_act_scale,
                                                 dark4_csp_add2_scale, dark4_csp_add1_zp, dark4_csp_m2_conv2_act_zp,
                                                 dark4_csp_add2_zp)
            # dark4_csp_m2_add_feature = dark4_Bottleneck_item2_add.add(dark4_csp_m1_add_feature,
            #                                                           dark4_csp_m2_conv2_feature)
            # gen_coe('../hand_coe/dark4_CSP_m2_add.coe', dark4_csp_m2_add_feature)

            # dark4 csp cat 将m的最终输出dark4_csp_m2_add_feature和conv2的输出dark4_csp_conv2_feature进行cat
            dark4_csp_add2_scale = np2tensor('../para/backbone.backbone.dark4.1.m.2.csp.scale.npy')
            dark4_csp_add2_zp = np2tensor('../para/backbone.backbone.dark4.1.m.2.csp.zero_point.npy')
            dark4_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.scale.npy')
            dark4_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark4.1.conv2.conv.zero_point.npy')
            dark4_csp_cat_scale = np2tensor('../para/backbone.backbone.dark4.1.csp1.scale.npy')
            dark4_csp_cat_zp = np2tensor('../para/backbone.backbone.dark4.1.csp1.zero_point.npy')

            dark4_csp_cat_feature = quant_cat(dark4_csp_m2_add_feature, dark4_csp_conv2_feature, dark4_csp_add2_scale,
                                              dark4_csp_conv2_act_scale, dark4_csp_cat_scale, dark4_csp_add2_zp,
                                              dark4_csp_conv2_act_zp, dark4_csp_cat_zp)

            # gen_coe('../hand_coe/dark4_CSP_cat.coe', dark4_csp_cat_feature)
            # dark4 csp conv3 s=1 p=0  cat之后的输出作为conv3的输入
            dark4_csp_conv3_quant_weight = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.weight.int.npy')
            dark4_csp_conv3_bias = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.bias.npy')
            dark4_csp_conv3_feature = self.dark4_csp_conv3(dark4_csp_cat_feature, dark4_csp_conv3_quant_weight,
                                                           dark4_csp_conv3_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # =========================start dark5============================
            #  dark4的输出作为dark5的输入
            # BaseConv s=2 p=1
            dark5_BaseConv_quant_weight = np2tensor('../para/backbone.backbone.dark5.0.conv.weight.int.npy')
            dark5_BaseConv_bias = np2tensor('../para/backbone.backbone.dark5.0.conv.bias.npy')
            dark5_BaseConv_feature = self.dark5_baseconv(dark4_csp_conv3_feature, dark5_BaseConv_quant_weight,
                                                         dark5_BaseConv_bias, out_hand=1, write_coe_file=0, stride=2,
                                                         padding=1)
            # =================dark5 csp=================

            # csp_conv1 s=1 p=0
            dark5_csp_conv1_quant_weight = np2tensor('../para/backbone.backbone.dark5.1.conv1.conv.weight.int.npy')
            dark5_csp_conv1_bias = np2tensor('../para/backbone.backbone.dark5.1.conv1.conv.bias.npy')
            dark5_csp_conv1_feature = self.dark5_csp_conv1(dark5_BaseConv_feature, dark5_csp_conv1_quant_weight,
                                                           dark5_csp_conv1_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)
            # csp_conv2 s=1 p=0
            dark5_csp_conv2_quant_weight = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.weight.int.npy')
            dark5_csp_conv2_bias = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.bias.npy')
            dark5_csp_conv2_feature = self.dark5_csp_conv2(dark5_BaseConv_feature, dark5_csp_conv2_quant_weight,
                                                           dark5_csp_conv2_bias,
                                                           out_hand=1, write_coe_file=0, stride=1, padding=0)

            # =========== start dark5 csp m0===================

            # m0 conv1 s=1 p=0

            dark5_csp_m0_conv1_quant_weight = np2tensor(
                '../para/backbone.backbone.dark5.1.m.0.conv1.conv.weight.int.npy')
            dark5_csp_m0_conv1_bias = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv1.conv.bias.npy')
            dark5_csp_m0_conv1_feature = self.dark5_csp_m0_conv1(dark5_csp_conv1_feature,
                                                                 dark5_csp_m0_conv1_quant_weight,
                                                                 dark5_csp_m0_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # m0 conv2 s=1 p=1
            dark5_csp_m0_conv2_quant_weight = np2tensor(
                '../para/backbone.backbone.dark5.1.m.0.conv2.conv.weight.int.npy')
            dark5_csp_m0_conv2_bias = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv2.conv.bias.npy')
            dark5_csp_m0_conv2_feature = self.dark5_csp_m0_conv2(dark5_csp_m0_conv1_feature,
                                                                 dark5_csp_m0_conv2_quant_weight,
                                                                 dark5_csp_m0_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)
            # m0卷积结束 开始add,add对象为 csp_conv1的输入dark5_csp_conv1_feature 和 m0_conv2的输出 dark5_csp_m0_conv2_feature

            # cat 将m的输出dark5_csp_m0_conv2 和 csp_conv2的输出dark5_csp_conv2_feature  进行cat
            dark5_csp_m0_conv2_act_scale = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv2.conv.scale.npy')
            dark5_csp_m0_conv2_act_zp = np2tensor('../para/backbone.backbone.dark5.1.m.0.conv2.conv.zero_point.npy')
            dark5_csp_conv2_act_scale = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.scale.npy')
            dark5_csp_conv2_act_zp = np2tensor('../para/backbone.backbone.dark5.1.conv2.conv.zero_point.npy')
            dark5_csp_cat_scale = np2tensor('../para/backbone.backbone.dark5.1.csp1.scale.npy')
            dark5_csp_cat_zp = np2tensor('../para/backbone.backbone.dark5.1.csp1.zero_point.npy')

            dark5_csp_cat_feature = quant_cat(dark5_csp_m0_conv2_feature, dark5_csp_conv2_feature,
                                              dark5_csp_m0_conv2_act_scale, dark5_csp_conv2_act_scale,
                                              dark5_csp_cat_scale, dark5_csp_m0_conv2_act_zp, dark5_csp_conv2_act_zp,
                                              dark5_csp_cat_zp)

            # gen_coe('../hand_coe/dark5_CSP_cat.coe', dark5_csp_cat_feature)

            # dark5_csp_conv3  s=1 p=0
            dark5_csp_conv3_quant_weight = np2tensor('../para/backbone.backbone.dark5.1.conv3.conv.weight.int.npy')
            dark5_csp_conv3_bias = np2tensor('../para/backbone.backbone.dark5.1.conv3.conv.bias.npy')
            dark5_csp_conv3_feature = self.dark5_csp_conv3(dark5_csp_cat_feature, dark5_csp_conv3_quant_weight,
                                                           dark5_csp_conv3_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # =============CSPDarknet 结束  ====================
            # ============= lateral_conv0 =================
            feat1 = dark3_csp_conv3_feature
            feat2 = dark4_csp_conv3_feature
            feat3 = dark5_csp_conv3_feature

            # s=1 p=0
            lateral_conv0_quant_weight = np2tensor('../para/backbone.lateral_conv0.conv.weight.int.npy')
            lateral_conv0_bias = np2tensor('../para/backbone.lateral_conv0.conv.bias.npy')
            P5 = self.lateral_conv0(feat3, lateral_conv0_quant_weight, lateral_conv0_bias, out_hand=1,
                                    write_coe_file=0, stride=1, padding=0)
            torch_upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

            P5_upsample = torch_upsample(P5)
            # gen_coe('../hand_coe/P5_upsample.coe', P5_upsample)
            lateral_conv0_act_scale = np2tensor('../para/backbone.lateral_conv0.conv.scale.npy')
            lateral_conv0_act_zp = np2tensor('../para/backbone.lateral_conv0.conv.zero_point.npy')
            dark4_csp_conv3_act_scale = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.scale.npy')
            dark4_csp_conv3_act_zp = np2tensor('../para/backbone.backbone.dark4.1.conv3.conv.zero_point.npy')
            YOLOPAFPN_csp2_cat_scale = np2tensor('../para/backbone.csp2.scale.npy')
            YOLOPAFPN_csp2_cat_zp = np2tensor('../para/backbone.csp2.zero_point.npy')

            P5_upsample_cat = quant_cat(P5_upsample, feat2, lateral_conv0_act_scale, dark4_csp_conv3_act_scale,
                                        YOLOPAFPN_csp2_cat_scale, lateral_conv0_act_zp, dark4_csp_conv3_act_zp,
                                        YOLOPAFPN_csp2_cat_zp)
            # P5_upsample_cat = YOLOPAFPN_csp2_cat.cat([P5_upsample, feat2], 1)
            # gen_coe('../hand_coe/P5_upsample_cat.coe', P5_upsample_cat)
            # exit()

            # =====================start C3_p4 =======================
            # C3_p4的输入为cat之后的P5_upsample
            c3_p4_csp_conv1_quant_weight = np2tensor('../para/backbone.C3_p4.conv1.conv.weight.int.npy')
            c3_p4_csp_conv1_bias = np2tensor('../para/backbone.C3_p4.conv1.conv.bias.npy')
            c3_p4_csp_conv1_feature = self.c3_p4_csp_conv1(P5_upsample_cat, c3_p4_csp_conv1_quant_weight,
                                                           c3_p4_csp_conv1_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)
            c3_p4_csp_conv2_quant_weight = np2tensor('../para/backbone.C3_p4.conv2.conv.weight.int.npy')
            c3_p4_csp_conv2_bias = np2tensor('../para/backbone.C3_p4.conv2.conv.bias.npy')
            c3_p4_csp_conv2_feature = self.c3_p4_csp_conv2(P5_upsample_cat, c3_p4_csp_conv2_quant_weight,
                                                           c3_p4_csp_conv2_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)
            # ==============start c3_p4_csp_m0=====================
            #  进入到 m0_conv1的为 csp_conv1的输出  s=1 p=0
            c3_p4_csp_m0_conv1_quant_weight = np2tensor('../para/backbone.C3_p4.m.0.conv1.conv.weight.int.npy')
            c3_p4_csp_m0_conv1_bias = np2tensor('../para/backbone.C3_p4.m.0.conv1.conv.bias.npy')
            c3_p4_csp_m0_conv1_feature = self.c3_p4_csp_m0_conv1(c3_p4_csp_conv1_feature,
                                                                 c3_p4_csp_m0_conv1_quant_weight,
                                                                 c3_p4_csp_m0_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # s=1 p=1
            c3_p4_csp_m0_conv2_quant_weight = np2tensor('../para/backbone.C3_p4.m.0.conv2.conv.weight.int.npy')
            c3_p4_csp_m0_conv2_bias = np2tensor('../para/backbone.C3_p4.m.0.conv2.conv.bias.npy')
            c3_p4_csp_m0_conv2_feature = self.c3_p4_csp_m0_conv2(c3_p4_csp_m0_conv1_feature,
                                                                 c3_p4_csp_m0_conv2_quant_weight,
                                                                 c3_p4_csp_m0_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)

            # cat  cat的两个分别是 m0的输出c3_p4_csp_m0_add_feature   和   csp_conv2的输出c3_p4_csp_conv2_feature
            c3_p4_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_p4.m.0.conv2.conv.scale.npy')
            c3_p4_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_p4.m.0.conv2.conv.zero_point.npy')
            c3_p4_csp_conv2_act_scale = np2tensor('../para/backbone.C3_p4.conv2.conv.scale.npy')
            c3_p4_csp_conv2_act_zp = np2tensor('../para/backbone.C3_p4.conv2.conv.zero_point.npy')
            c3_p4_csp_cat_scale = np2tensor('../para/backbone.C3_p4.csp1.scale.npy')
            c3_p4_csp_cat_zp = np2tensor('../para/backbone.C3_p4.csp1.zero_point.npy')
            c3_p4_csp_cat_feature = quant_cat(c3_p4_csp_m0_conv2_feature, c3_p4_csp_conv2_feature,
                                              c3_p4_csp_m0_conv2_act_scale, c3_p4_csp_conv2_act_scale,
                                              c3_p4_csp_cat_scale, c3_p4_csp_m0_conv2_act_zp, c3_p4_csp_conv2_act_zp,
                                              c3_p4_csp_cat_zp)

            # gen_coe('../hand_coe/C3_P4_CSP_cat.coe', c3_p4_csp_cat_feature)

            # c3_p4_csp_conv3
            c3_p4_csp_conv3_quant_weight = np2tensor('../para/backbone.C3_p4.conv3.conv.weight.int.npy')
            c3_p4_csp_conv3_bias = np2tensor('../para/backbone.C3_p4.conv3.conv.bias.npy')
            c3_p4_csp_conv3_feature = self.c3_p4_csp_conv3(c3_p4_csp_cat_feature, c3_p4_csp_conv3_quant_weight,
                                                           c3_p4_csp_conv3_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # ==========start reduce_conv1=================
            # reduce_conv1 的输入为 c3_p4的输出 c3_p4_csp_conv3_feature
            # s=1 p=0

            reduce_conv1_quant_weight = np2tensor('../para/backbone.reduce_conv1.conv.weight.int.npy')
            reduce_conv1_bias = np2tensor('../para/backbone.reduce_conv1.conv.bias.npy')
            reduce_conv1_feature = self.reduce_conv1(c3_p4_csp_conv3_feature, reduce_conv1_quant_weight,
                                                     reduce_conv1_bias, out_hand=1, write_coe_file=0, stride=1,
                                                     padding=0)
            # =========reduce_conv1结束  upsample ============
            # upsample的输入为 reduce_conv1的输出
            P4 = reduce_conv1_feature
            P4_upsample = torch_upsample(P4)
            # gen_coe('../hand_coe/P4_upsample.coe', P4_upsample)

            reduce_conv1_act_scale = np2tensor('../para/backbone.reduce_conv1.conv.scale.npy')
            reduce_conv1_act_zp = np2tensor('../para/backbone.reduce_conv1.conv.zero_point.npy')
            dark3_csp_conv3_act_scale = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.scale.npy')
            dark3_csp_conv3_act_zp = np2tensor('../para/backbone.backbone.dark3.1.conv3.conv.zero_point.npy')
            YOLOPAFPN_csp3_cat_scale = np2tensor('../para/backbone.csp3.scale.npy')
            YOLOPAFPN_csp3_cat_zp = np2tensor('../para/backbone.csp3.zero_point.npy')
            P4_upsample_cat = quant_cat(P4_upsample, feat1, reduce_conv1_act_scale, dark3_csp_conv3_act_scale,
                                        YOLOPAFPN_csp3_cat_scale, reduce_conv1_act_zp, dark3_csp_conv3_act_zp,
                                        YOLOPAFPN_csp3_cat_zp)
            # P4_upsample_cat = YOLOPAFPN_csp3_cat.cat([P4_upsample, feat1], 1)
            # gen_coe('../hand_coe/P4_upsample_cat.coe', P4_upsample_cat)

            # ===============start c3_p3=================
            # ============== c3_p3_csp==============
            # 输入为 cat之后的P4_upsample  P4_upsample_cat s=1 p=0
            c3_p3_csp_conv1_quant_weight = np2tensor('../para/backbone.C3_p3.conv1.conv.weight.int.npy')
            c3_p3_csp_conv1_bias = np2tensor('../para/backbone.C3_p3.conv1.conv.bias.npy')
            c3_p3_csp_conv1_feature = self.c3_p3_csp_conv1(P4_upsample_cat, c3_p3_csp_conv1_quant_weight,
                                                           c3_p3_csp_conv1_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            c3_p3_csp_conv2_quant_weight = np2tensor('../para/backbone.C3_p3.conv2.conv.weight.int.npy')
            c3_p3_csp_conv2_bias = np2tensor('../para/backbone.C3_p3.conv2.conv.bias.npy')
            c3_p3_csp_conv2_feature = self.c3_p3_csp_conv2(P4_upsample_cat, c3_p3_csp_conv2_quant_weight,
                                                           c3_p3_csp_conv2_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # =========c3_p3_csp_m0 =================
            # m0的输入为csp_conv1的输出
            # m0_conv1 s=1 p=0
            c3_p3_csp_m0_conv1_quant_weight = np2tensor('../para/backbone.C3_p3.m.0.conv1.conv.weight.int.npy')
            c3_p3_csp_m0_conv1_bias = np2tensor('../para/backbone.C3_p3.m.0.conv1.conv.bias.npy')
            c3_p3_csp_m0_conv1_feature = self.c3_p3_csp_m0_conv1(c3_p3_csp_conv1_feature,
                                                                 c3_p3_csp_m0_conv1_quant_weight,
                                                                 c3_p3_csp_m0_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)

            # m0_conv2 s=1 p=1
            c3_p3_csp_m0_conv2_quant_weight = np2tensor('../para/backbone.C3_p3.m.0.conv2.conv.weight.int.npy')
            c3_p3_csp_m0_conv2_bias = np2tensor('../para/backbone.C3_p3.m.0.conv2.conv.bias.npy')
            c3_p3_csp_m0_conv2_feature = self.c3_p3_csp_m0_conv2(c3_p3_csp_m0_conv1_feature,
                                                                 c3_p3_csp_m0_conv2_quant_weight,
                                                                 c3_p3_csp_m0_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)

            # csp_cat 对象为m0 的输出c3_p3_csp_m0_conv2_feature 和csp_conv2的输出 c3_p3_csp_conv2_feature
            c3_p3_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_p3.m.0.conv2.conv.scale.npy')
            c3_p3_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_p3.m.0.conv2.conv.zero_point.npy')
            c3_p3_csp_conv2_act_scale = np2tensor('../para/backbone.C3_p3.conv2.conv.scale.npy')
            c3_p3_csp_conv2_act_zp = np2tensor('../para/backbone.C3_p3.conv2.conv.zero_point.npy')
            c3_p3_csp_cat_scale = np2tensor('../para/backbone.C3_p3.csp1.scale.npy')
            c3_p3_csp_cat_zp = np2tensor('../para/backbone.C3_p3.csp1.zero_point.npy')

            c3_p3_csp_cat_feature = quant_cat(c3_p3_csp_m0_conv2_feature, c3_p3_csp_conv2_feature,
                                              c3_p3_csp_m0_conv2_act_scale, c3_p3_csp_conv2_act_scale,
                                              c3_p3_csp_cat_scale, c3_p3_csp_m0_conv2_act_zp, c3_p3_csp_conv2_act_zp,
                                              c3_p3_csp_cat_zp)

            # gen_coe('../hand_coe/C3_P3_CSP_cat.coe', c3_p3_csp_cat_feature)

            # ===========c3_p3_csp_conv3===============
            c3_p3_csp_conv3_quant_weight = np2tensor('../para/backbone.C3_p3.conv3.conv.weight.int.npy')
            c3_p3_csp_conv3_bias = np2tensor('../para/backbone.C3_p3.conv3.conv.bias.npy')
            c3_p3_csp_conv3_feature = self.c3_p3_csp_conv3(c3_p3_csp_cat_feature, c3_p3_csp_conv3_quant_weight,
                                                           c3_p3_csp_conv3_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # ==========bu_conv2===============
            # bu_conv2的输入为c3_p3的输出 即 c3_p3_csp_conv3_feature = P3_out
            # s=2 p=1
            bu_conv2_quant_weight = np2tensor('../para/backbone.bu_conv2.conv.weight.int.npy')
            bu_conv2_bias = np2tensor('../para/backbone.bu_conv2.conv.bias.npy')
            bu_conv2_feature = self.bu_conv2(c3_p3_csp_conv3_feature, bu_conv2_quant_weight, bu_conv2_bias,
                                             out_hand=1, write_coe_file=0, stride=2, padding=1)

            # ============= csp4.cat==============
            # cat的对象为bu_conv2的输出bu_conv2_feature (P3_downsample) 和P4(reduce_conv1的输出)
            bu_conv2_act_scale = np2tensor('../para/backbone.bu_conv2.conv.scale.npy')
            bu_conv2_act_zp = np2tensor('../para/backbone.bu_conv2.conv.zero_point.npy')
            reduce_conv1_act_scale = np2tensor('../para/backbone.reduce_conv1.conv.scale.npy')
            reduce_conv1_act_zp = np2tensor('../para/backbone.reduce_conv1.conv.zero_point.npy')
            YOLOPAFPN_csp4_cat_scale = np2tensor('../para/backbone.csp4.scale.npy')
            YOLOPAFPN_csp4_cat_zp = np2tensor('../para/backbone.csp4.zero_point.npy')
            P3_downsample_cat = quant_cat(bu_conv2_feature, P4, bu_conv2_act_scale, reduce_conv1_act_scale,
                                          YOLOPAFPN_csp4_cat_scale, bu_conv2_act_zp, reduce_conv1_act_zp,
                                          YOLOPAFPN_csp4_cat_zp)
            # P3_downsample_cat = YOLOPAFPN_csp4_cat.cat([bu_conv2_feature, P4], 1)
            # gen_coe('../hand_coe/P3_downsample_cat.coe', P3_downsample_cat)
            # exit()

            # ===============start c3_n3=================
            # ============== c3_n3_csp==============
            # 输入为 cat之后的P3_downsample  P3_downsample_cat s=1 p=0
            c3_n3_csp_conv1_quant_weight = np2tensor('../para/backbone.C3_n3.conv1.conv.weight.int.npy')
            c3_n3_csp_conv1_bias = np2tensor('../para/backbone.C3_n3.conv1.conv.bias.npy')
            c3_n3_csp_conv1_feature = self.c3_n3_csp_conv1(P3_downsample_cat, c3_n3_csp_conv1_quant_weight,
                                                           c3_n3_csp_conv1_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # s=1 p=0
            c3_n3_csp_conv2_quant_weight = np2tensor('../para/backbone.C3_n3.conv2.conv.weight.int.npy')
            c3_n3_csp_conv2_bias = np2tensor('../para/backbone.C3_n3.conv2.conv.bias.npy')
            c3_n3_csp_conv2_feature = self.c3_n3_csp_conv2(P3_downsample_cat, c3_n3_csp_conv2_quant_weight,
                                                           c3_n3_csp_conv2_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # ==============start c3_n3_csp_m===============
            # m的输入为 conv1的输出 c3_n3_csp_conv1_feature  s=1 p=0
            c3_n3_csp_m0_conv1_quant_weight = np2tensor('../para/backbone.C3_n3.m.0.conv1.conv.weight.int.npy')
            c3_n3_csp_m0_conv1_bias = np2tensor('../para/backbone.C3_n3.m.0.conv1.conv.bias.npy')
            c3_n3_csp_m0_conv1_feature = self.c3_n3_csp_m0_conv1(c3_n3_csp_conv1_feature,
                                                                 c3_n3_csp_m0_conv1_quant_weight,
                                                                 c3_n3_csp_m0_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # m0_conv2  s=1 p=1
            c3_n3_csp_m0_conv2_quant_weight = np2tensor('../para/backbone.C3_n3.m.0.conv2.conv.weight.int.npy')
            c3_n3_csp_m0_conv2_bias = np2tensor('../para/backbone.C3_n3.m.0.conv2.conv.bias.npy')
            c3_n3_csp_m0_conv2_feature = self.c3_n3_csp_m0_conv2(c3_n3_csp_m0_conv1_feature,
                                                                 c3_n3_csp_m0_conv2_quant_weight,
                                                                 c3_n3_csp_m0_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)

            # csp_cat  cat的对象为 m0的输出 c3_n3_csp_m0_conv2_feature 和csp_conv2的输出 c3_n3_csp_conv2_feature
            c3_n3_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_n3.m.0.conv2.conv.scale.npy')
            c3_n3_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_n3.m.0.conv2.conv.zero_point.npy')
            c3_n3_csp_conv2_act_scale = np2tensor('../para/backbone.C3_n3.conv2.conv.scale.npy')
            c3_n3_csp_conv2_act_zp = np2tensor('../para/backbone.C3_n3.conv2.conv.zero_point.npy')
            c3_n3_csp_cat_scale = np2tensor('../para/backbone.C3_n3.csp1.scale.npy')
            c3_n3_csp_cat_zp = np2tensor('../para/backbone.C3_n3.csp1.zero_point.npy')
            c3_n3_csp_cat_feature = quant_cat(c3_n3_csp_m0_conv2_feature, c3_n3_csp_conv2_feature,
                                              c3_n3_csp_m0_conv2_act_scale, c3_n3_csp_conv2_act_scale,
                                              c3_n3_csp_cat_scale, c3_n3_csp_m0_conv2_act_zp, c3_n3_csp_conv2_act_zp,
                                              c3_n3_csp_cat_zp)
            # c3_n3_csp_cat_feature = C3_n3_CSP_cat.cat((c3_n3_csp_m0_conv2_feature, c3_n3_csp_conv2_feature), 1)
            # gen_coe('../hand_coe/C3_n3_CSP_cat.coe', c3_n3_csp_cat_feature)

            # =======start c3_n3_csp_conv3=================
            # s=1 p=0
            c3_n3_csp_conv3_quant_weight = np2tensor('../para/backbone.C3_n3.conv3.conv.weight.int.npy')
            c3_n3_csp_conv3_bias = np2tensor('../para/backbone.C3_n3.conv3.conv.bias.npy')
            # P4_out
            c3_n3_csp_conv3_feature = self.c3_n3_csp_conv3(c3_n3_csp_cat_feature, c3_n3_csp_conv3_quant_weight,
                                                           c3_n3_csp_conv3_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            bu_conv1_quant_weight = np2tensor('../para/backbone.bu_conv1.conv.weight.int.npy')
            bu_conv1_bias = np2tensor('../para/backbone.bu_conv1.conv.bias.npy')
            # P4_downsample  s=2 p=1
            bu_conv1_feature = self.bu_conv1(c3_n3_csp_conv3_feature, bu_conv1_quant_weight, bu_conv1_bias,
                                             out_hand=1, write_coe_file=0, stride=2, padding=1)

            bu_conv1_act_scale = np2tensor('../para/backbone.bu_conv1.conv.scale.npy')
            bu_conv1_act_zp = np2tensor('../para/backbone.bu_conv1.conv.zero_point.npy')
            lateral_conv0_act_scale = np2tensor('../para/backbone.lateral_conv0.conv.scale.npy')
            lateral_conv0_act_zp = np2tensor('../para/backbone.lateral_conv0.conv.zero_point.npy')
            YOLOPAFPN_csp5_cat_scale = np2tensor('../para/backbone.csp5.scale.npy')
            YOLOPAFPN_csp5_cat_zp = np2tensor('../para/backbone.csp5.zero_point.npy')
            P4_downsample_cat = quant_cat(bu_conv1_feature, P5, bu_conv1_act_scale, lateral_conv0_act_scale,
                                          YOLOPAFPN_csp5_cat_scale, bu_conv1_act_zp, lateral_conv0_act_zp,
                                          YOLOPAFPN_csp5_cat_zp)
            # P4_downsample_cat = YOLOPAFPN_csp5_cat.cat([bu_conv1_feature, P5], 1)
            # gen_coe('../hand_coe/P4_downsample_cat.coe', P4_downsample_cat)
            # exit()

            # ==========start  C3_n4 ====================
            # c3_n4 的输入为 P4_downsample_cat  s=1 p=0

            c3_n4_csp_conv1_quant_weight = np2tensor('../para/backbone.C3_n4.conv1.conv.weight.int.npy')
            c3_n4_csp_conv1_bias = np2tensor('../para/backbone.C3_n4.conv1.conv.bias.npy')
            c3_n4_csp_conv1_feature = self.c3_n4_csp_conv1(P4_downsample_cat, c3_n4_csp_conv1_quant_weight,
                                                           c3_n4_csp_conv1_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            c3_n4_csp_conv2_quant_weight = np2tensor('../para/backbone.C3_n4.conv2.conv.weight.int.npy')
            c3_n4_csp_conv2_bias = np2tensor('../para/backbone.C3_n4.conv2.conv.bias.npy')
            c3_n4_csp_conv2_feature = self.c3_n4_csp_conv2(P4_downsample_cat, c3_n4_csp_conv2_quant_weight,
                                                           c3_n4_csp_conv2_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            # ============start c3_n4_m0===============
            # s=1 p=0
            c3_n4_csp_m0_conv1_quant_weight = np2tensor('../para/backbone.C3_n4.m.0.conv1.conv.weight.int.npy')
            c3_n4_csp_m0_conv1_bias = np2tensor('../para/backbone.C3_n4.m.0.conv1.conv.bias.npy')
            c3_n4_csp_m0_conv1_feature = self.c3_n4_csp_m0_conv1(c3_n4_csp_conv1_feature,
                                                                 c3_n4_csp_m0_conv1_quant_weight,
                                                                 c3_n4_csp_m0_conv1_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=0)
            # s=1 p=1
            c3_n4_csp_m0_conv2_quant_weight = np2tensor('../para/backbone.C3_n4.m.0.conv2.conv.weight.int.npy')
            c3_n4_csp_m0_conv2_bias = np2tensor('../para/backbone.C3_n4.m.0.conv2.conv.bias.npy')
            c3_n4_csp_m0_conv2_feature = self.c3_n4_csp_m0_conv2(c3_n4_csp_m0_conv1_feature,
                                                                 c3_n4_csp_m0_conv2_quant_weight,
                                                                 c3_n4_csp_m0_conv2_bias, out_hand=1, write_coe_file=0,
                                                                 stride=1, padding=1)

            # cat  cat的对象为 m0的输出 c3_n4_csp_m0_conv2_feature 和 csp_conv2的输出c3_n4_csp_conv2_feature
            c3_n4_csp_m0_conv2_act_scale = np2tensor('../para/backbone.C3_n4.m.0.conv2.conv.scale.npy')
            c3_n4_csp_m0_conv2_act_zp = np2tensor('../para/backbone.C3_n4.m.0.conv2.conv.zero_point.npy')
            c3_n4_csp_conv2_act_scale = np2tensor('../para/backbone.C3_n4.conv2.conv.scale.npy')
            c3_n4_csp_conv2_act_zp = np2tensor('../para/backbone.C3_n4.conv2.conv.zero_point.npy')
            c3_n4_csp_cat_scale = np2tensor('../para/backbone.C3_n4.csp1.scale.npy')
            c3_n4_csp_cat_zp = np2tensor('../para/backbone.C3_n4.csp1.zero_point.npy')

            c3_n4_csp_cat_feature = quant_cat(c3_n4_csp_m0_conv2_feature, c3_n4_csp_conv2_feature,
                                              c3_n4_csp_m0_conv2_act_scale, c3_n4_csp_conv2_act_scale,
                                              c3_n4_csp_cat_scale, c3_n4_csp_m0_conv2_act_zp, c3_n4_csp_conv2_act_zp,
                                              c3_n4_csp_cat_zp)
            # c3_n4_csp_cat_feature = C3_n4_CSP_cat.cat((c3_n4_csp_m0_conv2_feature, c3_n4_csp_conv2_feature), dim=1)
            # gen_coe('../hand_coe/C3_n4_CSP_cat.coe', c3_n4_csp_cat_feature)

            # =============== c3_n4 csp_conv3================
            c3_n4_csp_conv3_quant_weight = np2tensor('../para/backbone.C3_n4.conv3.conv.weight.int.npy')
            c3_n4_csp_conv3_bias = np2tensor('../para/backbone.C3_n4.conv3.conv.bias.npy')
            c3_n4_csp_conv3_feature = self.c3_n4_csp_conv3(c3_n4_csp_cat_feature, c3_n4_csp_conv3_quant_weight,
                                                           c3_n4_csp_conv3_bias, out_hand=1, write_coe_file=0, stride=1,
                                                           padding=0)

            P3_out = c3_p3_csp_conv3_feature
            P4_out = c3_n3_csp_conv3_feature
            P5_out = c3_n4_csp_conv3_feature

            # ================ start yolo head===============
            # ============P3 head===========
            # stem的输入为P3_out s=1 p=0
            p3_stem_conv_quant_weight = np2tensor('../para/head.stems.0.conv.weight.int.npy')
            p3_stem_conv_bias = np2tensor('../para/head.stems.0.conv.bias.npy')
            p3_stem_conv_feature = self.p3_stem_conv(P3_out, p3_stem_conv_quant_weight, p3_stem_conv_bias, out_hand=1,
                                                     write_coe_file=0, stride=1, padding=0)
            # ============p3_cls_conv==========
            # p3_cls_conv0 s=1 p=1
            p3_cls_conv0_quant_weight = np2tensor('../para/head.cls_convs.0.0.conv.weight.int.npy')
            p3_cls_conv0_bias = np2tensor('../para/head.cls_convs.0.0.conv.bias.npy')
            p3_cls_conv0_feature = self.p3_cls_conv0(p3_stem_conv_feature, p3_cls_conv0_quant_weight, p3_cls_conv0_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            # p3_cls_conv1 s=1 p=1
            p3_cls_conv1_quant_weight = np2tensor('../para/head.cls_convs.0.1.conv.weight.int.npy')
            p3_cls_conv1_bias = np2tensor('../para/head.cls_convs.0.1.conv.bias.npy')
            p3_cls_conv1_feature = self.p3_cls_conv1(p3_cls_conv0_feature, p3_cls_conv1_quant_weight, p3_cls_conv1_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            p3_cls_feat = p3_cls_conv1_feature

            # =============== p3_cls_preds==========
            p3_cls_preds_quant_weight = np2tensor('../para/head.cls_preds.0.weight.int.npy')
            p3_cls_preds_bias = np2tensor('../para/head.cls_preds.0.bias.npy')
            p3_cls_output = self.p3_cls_preds(p3_cls_feat.numpy(), p3_cls_preds_quant_weight, p3_cls_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            # =========== p3_reg_convs0============
            # s=1 p=1
            p3_reg_conv0_quant_weight = np2tensor('../para/head.reg_convs.0.0.conv.weight.int.npy')
            p3_reg_conv0_bias = np2tensor('../para/head.reg_convs.0.0.conv.bias.npy')
            p3_reg_conv0_feature = self.p3_reg_conv0(p3_stem_conv_feature, p3_reg_conv0_quant_weight, p3_reg_conv0_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            # =========== p3_reg_convs1============
            # s=1 p=1
            p3_reg_conv1_quant_weight = np2tensor('../para/head.reg_convs.0.1.conv.weight.int.npy')
            p3_reg_conv1_bias = np2tensor('../para/head.reg_convs.0.1.conv.bias.npy')
            p3_reg_conv1_feature = self.p3_reg_conv1(p3_reg_conv0_feature, p3_reg_conv1_quant_weight, p3_reg_conv1_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            p3_reg_feat = p3_reg_conv1_feature
            # =========== p3_reg_preds============
            # s=1 p=0
            p3_reg_preds_quant_weight = np2tensor('../para/head.reg_preds.0.weight.int.npy')
            p3_reg_preds_bias = np2tensor('../para/head.reg_preds.0.bias.npy')
            p3_reg_output = self.p3_reg_preds(p3_reg_feat.numpy(), p3_reg_preds_quant_weight, p3_reg_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            # =========== p3_obj_preds============
            # s=1 p=0
            p3_obj_preds_quant_weight = np2tensor('../para/head.obj_preds.0.weight.int.npy')
            p3_obj_preds_bias = np2tensor('../para/head.obj_preds.0.bias.npy')
            p3_obj_output = self.p3_obj_preds(p3_reg_feat.numpy(), p3_obj_preds_quant_weight, p3_obj_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            p3_reg_output_act_scale = np2tensor('../para/head.reg_preds.0.scale.npy')
            p3_reg_output_act_zp = np2tensor('../para/head.reg_preds.0.zero_point.npy')
            p3_obj_output_act_scale = np2tensor('../para/head.obj_preds.0.scale.npy')
            p3_obj_output_act_zp = np2tensor('../para/head.obj_preds.0.zero_point.npy')
            head_cat_scale = np2tensor('../para/head.csp6.scale.npy')
            head_cat_zp = np2tensor('../para/head.csp6.zero_point.npy')
            # print('p3_reg_output  shape',p3_reg_output.shape)
            # print('p3_obj_output  shape',p3_obj_output.shape)
            # print(p3_obj_output)
            # exit()

            p3_reg_obj_cat_feature = quant_cat(p3_reg_output, p3_obj_output, p3_reg_output_act_scale,
                                               p3_obj_output_act_scale, head_cat_scale, p3_reg_output_act_zp,
                                               p3_obj_output_act_zp, head_cat_zp)
            # gen_coe('../hand_coe/p3_reg_obj_cat.coe', p3_reg_obj_cat_feature)

            p3_cls_output_act_scale = np2tensor('../para/head.cls_preds.0.scale.npy')
            p3_cls_output_act_zp = np2tensor('../para/head.cls_preds.0.zero_point.npy')

            p3_output = quant_cat(p3_reg_obj_cat_feature, p3_cls_output, head_cat_scale, p3_cls_output_act_scale,
                                  head_cat_scale, head_cat_zp, p3_cls_output_act_zp, head_cat_zp)
            # print(p3_output)
            # gen_coe('../hand_coe/P3_output.coe', p3_output)
            # exit()

            # exit()
            # # head cat 的对象分别是 reg_preds obj_preds  cls_preds 的输出
            # p3_output = head_cat.cat([p3_reg_preds_feature, p3_obj_preds_feature, p3_cls_output], 1)
            # gen_coe('../hand_coe/p3_output.coe', p3_output.int_repr())

            # ============P4 head===========
            # stem的输入为P4_out s=1 p=0
            p4_stem_conv_quant_weight = np2tensor('../para/head.stems.1.conv.weight.int.npy')
            p4_stem_conv_bias = np2tensor('../para/head.stems.1.conv.bias.npy')
            p4_stem_conv_feature = self.p4_stem_conv(P4_out, p4_stem_conv_quant_weight, p4_stem_conv_bias, out_hand=1,
                                                     write_coe_file=0, stride=1, padding=0)
            # ============p4_cls_conv==========
            # p4_cls_conv0 s=1 p=1
            p4_cls_conv0_quant_weight = np2tensor('../para/head.cls_convs.1.0.conv.weight.int.npy')
            p4_cls_conv0_bias = np2tensor('../para/head.cls_convs.1.0.conv.bias.npy')
            p4_cls_conv0_feature = self.p4_cls_conv0(p4_stem_conv_feature, p4_cls_conv0_quant_weight, p4_cls_conv0_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            # p4_cls_conv1 s=1 p=1
            p4_cls_conv1_quant_weight = np2tensor('../para/head.cls_convs.1.1.conv.weight.int.npy')
            p4_cls_conv1_bias = np2tensor('../para/head.cls_convs.1.1.conv.bias.npy')
            p4_cls_conv1_feature = self.p4_cls_conv1(p4_cls_conv0_feature, p4_cls_conv1_quant_weight, p4_cls_conv1_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            p4_cls_feat = p4_cls_conv1_feature

            # =============== p4_cls_preds==========
            p4_cls_preds_quant_weight = np2tensor('../para/head.cls_preds.1.weight.int.npy')
            p4_cls_preds_bias = np2tensor('../para/head.cls_preds.1.bias.npy')
            p4_cls_output = self.p4_cls_preds(p4_cls_feat.numpy(), p4_cls_preds_quant_weight, p4_cls_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            # =========== p4_reg_convs0============
            # s=1 p=1
            p4_reg_conv0_quant_weight = np2tensor('../para/head.reg_convs.1.0.conv.weight.int.npy')
            p4_reg_conv0_bias = np2tensor('../para/head.reg_convs.1.0.conv.bias.npy')
            p4_reg_conv0_feature = self.p4_reg_conv0(p4_stem_conv_feature, p4_reg_conv0_quant_weight, p4_reg_conv0_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            # =========== p4_reg_convs1============
            # s=1 p=1
            p4_reg_conv1_quant_weight = np2tensor('../para/head.reg_convs.1.1.conv.weight.int.npy')
            p4_reg_conv1_bias = np2tensor('../para/head.reg_convs.1.1.conv.bias.npy')
            p4_reg_conv1_feature = self.p4_reg_conv1(p4_reg_conv0_feature, p4_reg_conv1_quant_weight, p4_reg_conv1_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            p4_reg_feat = p4_reg_conv1_feature
            # =========== p4_reg_preds============
            # s=1 p=0
            p4_reg_preds_quant_weight = np2tensor('../para/head.reg_preds.1.weight.int.npy')
            p4_reg_preds_bias = np2tensor('../para/head.reg_preds.1.bias.npy')
            p4_reg_output = self.p4_reg_preds(p4_reg_feat.numpy(), p4_reg_preds_quant_weight, p4_reg_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            # =========== p4_obj_preds============
            # s=1 p=0
            p4_obj_preds_quant_weight = np2tensor('../para/head.obj_preds.1.weight.int.npy')
            p4_obj_preds_bias = np2tensor('../para/head.obj_preds.1.bias.npy')
            p4_obj_output = self.p4_obj_preds(p4_reg_feat.numpy(), p4_obj_preds_quant_weight, p4_obj_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            p4_reg_output_act_scale = np2tensor('../para/head.reg_preds.1.scale.npy')
            p4_reg_output_act_zp = np2tensor('../para/head.reg_preds.1.zero_point.npy')
            p4_obj_output_act_scale = np2tensor('../para/head.obj_preds.1.scale.npy')
            p4_obj_output_act_zp = np2tensor('../para/head.obj_preds.1.zero_point.npy')
            head_cat_scale = np2tensor('../para/head.csp6.scale.npy')
            head_cat_zp = np2tensor('../para/head.csp6.zero_point.npy')

            p4_reg_obj_cat_feature = quant_cat(p4_reg_output, p4_obj_output, p4_reg_output_act_scale,
                                               p4_obj_output_act_scale, head_cat_scale, p4_reg_output_act_zp,
                                               p4_obj_output_act_zp, head_cat_zp)
            # gen_coe('../hand_coe/p4_reg_obj_cat.coe', p4_reg_obj_cat_feature)

            p4_cls_output_act_scale = np2tensor('../para/head.cls_preds.1.scale.npy')
            p4_cls_output_act_zp = np2tensor('../para/head.cls_preds.1.zero_point.npy')

            p4_output = quant_cat(p4_reg_obj_cat_feature, p4_cls_output, head_cat_scale, p4_cls_output_act_scale,
                                  head_cat_scale, head_cat_zp, p4_cls_output_act_zp, head_cat_zp)
            # gen_coe('../hand_coe/P4_output.coe', p4_output)

            # ============P5 head===========
            # stem的输入为P5_out s=1 p=0
            p5_stem_conv_quant_weight = np2tensor('../para/head.stems.2.conv.weight.int.npy')
            p5_stem_conv_bias = np2tensor('../para/head.stems.2.conv.bias.npy')
            p5_stem_conv_feature = self.p5_stem_conv(P5_out, p5_stem_conv_quant_weight, p5_stem_conv_bias, out_hand=1,
                                                     write_coe_file=0, stride=1, padding=0)
            # ============p5_cls_conv==========
            # p5_cls_conv0 s=1 p=1
            p5_cls_conv0_quant_weight = np2tensor('../para/head.cls_convs.2.0.conv.weight.int.npy')
            p5_cls_conv0_bias = np2tensor('../para/head.cls_convs.2.0.conv.bias.npy')
            p5_cls_conv0_feature = self.p5_cls_conv0(p5_stem_conv_feature, p5_cls_conv0_quant_weight, p5_cls_conv0_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            # p5_cls_conv1 s=1 p=1
            p5_cls_conv1_quant_weight = np2tensor('../para/head.cls_convs.2.1.conv.weight.int.npy')
            p5_cls_conv1_bias = np2tensor('../para/head.cls_convs.2.1.conv.bias.npy')
            p5_cls_conv1_feature = self.p5_cls_conv1(p5_cls_conv0_feature, p5_cls_conv1_quant_weight, p5_cls_conv1_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            p5_cls_feat = p5_cls_conv1_feature

            # =============== p5_cls_preds==========
            p5_cls_preds_quant_weight = np2tensor('../para/head.cls_preds.2.weight.int.npy')
            p5_cls_preds_bias = np2tensor('../para/head.cls_preds.2.bias.npy')
            p5_cls_output = self.p5_cls_preds(p5_cls_feat.numpy(), p5_cls_preds_quant_weight, p5_cls_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            # =========== p5_reg_convs0============
            # s=1 p=1
            p5_reg_conv0_quant_weight = np2tensor('../para/head.reg_convs.2.0.conv.weight.int.npy')
            p5_reg_conv0_bias = np2tensor('../para/head.reg_convs.2.0.conv.bias.npy')
            p5_reg_conv0_feature = self.p5_reg_conv0(p5_stem_conv_feature, p5_reg_conv0_quant_weight, p5_reg_conv0_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            # =========== p5_reg_convs1============
            # s=1 p=1
            p5_reg_conv1_quant_weight = np2tensor('../para/head.reg_convs.2.1.conv.weight.int.npy')
            p5_reg_conv1_bias = np2tensor('../para/head.reg_convs.2.1.conv.bias.npy')
            p5_reg_conv1_feature = self.p5_reg_conv1(p5_reg_conv0_feature, p5_reg_conv1_quant_weight, p5_reg_conv1_bias,
                                                     out_hand=1,
                                                     write_coe_file=0, stride=1, padding=1)
            p5_reg_feat = p5_reg_conv1_feature
            # =========== p5_reg_preds============
            # s=1 p=0
            p5_reg_preds_quant_weight = np2tensor('../para/head.reg_preds.2.weight.int.npy')
            p5_reg_preds_bias = np2tensor('../para/head.reg_preds.2.bias.npy')
            p5_reg_output = self.p5_reg_preds(p5_reg_feat.numpy(), p5_reg_preds_quant_weight, p5_reg_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            # =========== p5_obj_preds============
            # s=1 p=0
            p5_obj_preds_quant_weight = np2tensor('../para/head.obj_preds.2.weight.int.npy')
            p5_obj_preds_bias = np2tensor('../para/head.obj_preds.2.bias.npy')
            p5_obj_output = self.p5_obj_preds(p5_reg_feat.numpy(), p5_obj_preds_quant_weight, p5_obj_preds_bias,
                                              out_hand=1,
                                              write_coe_file=0, stride=1, padding=0)

            p5_reg_output_act_scale = np2tensor('../para/head.reg_preds.2.scale.npy')
            p5_reg_output_act_zp = np2tensor('../para/head.reg_preds.2.zero_point.npy')
            p5_obj_output_act_scale = np2tensor('../para/head.obj_preds.2.scale.npy')
            p5_obj_output_act_zp = np2tensor('../para/head.obj_preds.2.zero_point.npy')
            head_cat_scale = np2tensor('../para/head.csp6.scale.npy')
            head_cat_zp = np2tensor('../para/head.csp6.zero_point.npy')

            p5_reg_obj_cat_feature = quant_cat(p5_reg_output, p5_obj_output, p5_reg_output_act_scale,
                                               p5_obj_output_act_scale, head_cat_scale, p5_reg_output_act_zp,
                                               p5_obj_output_act_zp, head_cat_zp)
            p5_cls_output_act_scale = np2tensor('../para/head.cls_preds.2.scale.npy')
            p5_cls_output_act_zp = np2tensor('../para/head.cls_preds.2.zero_point.npy')

            p5_output = quant_cat(p5_reg_obj_cat_feature, p5_cls_output, head_cat_scale, p5_cls_output_act_scale,
                                  head_cat_scale, head_cat_zp, p5_cls_output_act_zp, head_cat_zp)
            # gen_coe('../hand_coe/P5_output.coe', p5_output)
            if self.show_result:

                head_scale = torch.from_numpy(np.load('../para/head.csp6.scale.npy'))
                head_zp = torch.from_numpy(np.load('../para/head.csp6.zero_point.npy'))
                float_output = []
                output_list = [p3_output, p4_output, p5_output]
                for output in output_list:
                    out_shape = output.shape
                    tmp = torch.zeros((out_shape[0], 6, out_shape[2], out_shape[3]))
                    tmp[:, :4, :, :] = output[:, :4, :, :]
                    tmp[:, 4, :, :] = output[:, 8, :, :]
                    tmp[:, 5, :, :] = output[:, 16, :, :]
                    tmp_float = float(head_scale) * (tmp.to(torch.float) - int(head_zp))
                    float_output.append(tmp_float)

                final_out = decode_outputs(float_output, [640, 640])

                img_result = detect_img(self.img_path, outputs=final_out)
                img_result.show()


if __name__ == "__main__":
    img_path = '../img/001.jpg'
    model = QuantizableYolox(img_path, True)
    model.forward()

    # for i in out_list:
    #     print(i.shape)
    # print(p3)
    # exit()

    # print(type(p3_output))
