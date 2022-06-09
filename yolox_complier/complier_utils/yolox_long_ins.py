# -*- coding: UTF-8 -*-
# encoding=utf-8
import time
from tkinter.tix import CheckList
import torch
import torch.nn as nn
import numpy as np

from complier_utils.utils_base import ins64to32
from picture_load import *
from ins_conv_new_831 import *
import cv2
# np.set_printoptions(threshold=np.inf)
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys

sys.path.append("..")
import torch.quantization


def weight_4to8(weight_stem_conv):
    weight_shape = weight_stem_conv.shape
    channel_out_num = weight_shape[0]
    channel_in_num = weight_shape[1]
    if weight_shape[1] == 4:
        channel_in_num = weight_shape[1] * 2
    elif weight_shape[1] == 12:
        channel_in_num = weight_shape[1] + 4
    elif weight_shape[0] == 1:
        channel_out_num = channel_out_num * 8
    elif weight_shape[0] == 4:
        channel_out_num = channel_out_num * 2
    new_feature = np.zeros((channel_out_num, channel_in_num, weight_shape[2], weight_shape[3]), dtype=np.uint8)
    weight_stem_conv_0 = add_feature_channel(new_feature, weight_stem_conv, weight_shape)
    weight_stem_conv_0 = torch.from_numpy(weight_stem_conv_0)
    return weight_stem_conv_0


def add_feature_channel(new_weig, weig, shape):  # 补0 把weig存入new_weig；new_weig是全0，
    for kernel_num in range(shape[0]):
        for channel_in_num in range(shape[1]):
            for row in range(shape[2]):
                for col in range(shape[3]):
                    new_weig[kernel_num][channel_in_num][row][col] = weig[kernel_num][channel_in_num][row][col]
    return new_weig


def Focus(feature):
    # print(x)
    # x :1, 1, 640, 640  b c   h     w
    # …代替了切片操作中前面所有的:， 即a[:, :, None] 和a[…, None]等价 
    #  一  二
    #  三  四 
    # 第一组                    h      w  步幅都是2 都是取偶数
    patch_top_left = feature[..., ::2, ::2]  #::2 取偶数位置
    # 第三组                   h为奇  w为偶
    patch_bot_left = feature[..., 1::2, ::2]  # 1::2 取奇数位置
    # 第二组 h为偶 w为奇
    patch_top_right = feature[..., ::2, 1::2]
    # 第四组 h为奇 w为奇
    patch_bot_right = feature[..., 1::2, 1::2]
    x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )

    feature_shape = list(x.shape)
    if feature_shape[1] == 4:
        feature_shape[1] = feature_shape[1] * 2
    else:
        feature_shape[1] = feature_shape[1] + 4
    new_feature = np.zeros(feature_shape)
    focus_out = add_feature_channel(new_feature, x, x.shape)
    # compare_Focus(x,focus_out,feature_shape[0],feature_shape[1],feature_shape[2],feature_shape[3])
    focus_out_tensor = torch.from_numpy(focus_out)
    return focus_out_tensor


class Conv2d_Q(nn.Module):
    def __init__(
            self,

            quant_scale1=None,
            quant_zero_point1=None,
            quant_scale2=None,
            quant_zero_point2=None,
            quant_scale3=None,
            quant_zero_point3=None,
            first_convs=0,
            coe_name=None

    ):
        super(Conv2d_Q, self).__init__()

        self.quant_scale1 = quant_scale1
        self.quant_zero_point1 = quant_zero_point1
        self.quant_scale2 = quant_scale2
        self.quant_zero_point2 = quant_zero_point2
        self.quant_scale3 = quant_scale3
        self.quant_zero_point3 = quant_zero_point3
        self.first_convs = first_convs
        self.coe_name = coe_name

    def forward(self, weight_address, computer_address, feature, f_weight, q_weight, bias=0, path1=0, stride=2,
                padding=1, block=0, cat2_weight_address=1, isleakrelu=0, add_write_address=16777216,
                cat2=torch.tensor((1, 1, 1, 1)), operator='',
                index=1):  # weight_address权重地址 computer_address计算地址 add_write_address加上这个地址变成写地址
        print('weight_address=', '%08X' % int(weight_address))
        print('computer_address=', '%08X' % computer_address)
        print('computer_write_address=', '%08X' % (computer_address + add_write_address))
        if (computer_address + add_write_address) >= weight_address:
            print("error")
        if block != 0:  # 如果分块
            coe_name = self.coe_name
            # weight_address = weight_address+
            weight_address, computer_write_address = conv33_block(coe_name, weight_address, computer_address, feature,
                                                                  f_weight, q_weight, bias, path1, stride, padding,
                                                                  block, cat2_weight_address, isleakrelu,
                                                                  add_write_address, operator, self.quant_zero_point1,
                                                                  self.quant_zero_point3,
                                                                  self.quant_scale3)
            return weight_address, computer_write_address
        else:
            default_data = 1300005000000000
            file_name = '../ins/yolox_ins_64_682.dat'
            # reg表对应的地址 寄存器表
            ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                           'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                           'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                           'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                           'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                           'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40'}

            # if (operator == 'image'):
            #     with open(file_name, 'a+') as f:
            #         for index in range(19):
            #             f.write('1300005000000000')
            #             f.write('\n')
            #         f.write('1300000000000002')
            #         f.write('\n')
            if (operator == 'image_final'):  # image_final第一层
                shape = q_weight.shape  # 32 12 3 3 
                # 计算权重的数量
                # weight_size = int(1440)
                weight_B = int((shape[0] * shape[1] * shape[2] * shape[3]) / (72 / 8) * (256 / 8))
                bias_scale_shift_B = int((32 / 8) * 3 * 32)
                regB = 32
                weight_size = weight_B + bias_scale_shift_B + regB
                # weight_size为权重的数量单位B
                # ----------------conv33权重指令-------------------
                # 计算权重的reg4
                # dataSizeW是权重每行多少bit, dataSizeB是bias每行多少bit
                # dataSizeW = 8
                # dataSizeB = 8
                reg4 = default_data
                reg5 = default_data
                switch = default_data
                # 权重第一个指令:读地址
                with open(file_name, 'a+') as f:
                    # f.write('13')
                    # 第一个指令：读权重
                    f.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    f.write('%08X' % int(weight_address))
                    f.write('\n')
                    # 权重第二个指令:读数量
                    f.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    f.write('%08X' % weight_size)  # 读权重
                    f.write('\n')
                    # 权重的第三个指令reg4
                    # f.write('')
                    # f.write(str(reg4))
                    # f.write('\n')
                    # 权重的第四个指令reg5
                    # f.write('130000' + ins_address['TJPU_Reg5'])
                    # f.write('')
                    # f.write(str(reg5))
                    # f.write('\n')
                    # 计算的第五个指令,switch
                    # f.write('')
                    # f.write(str(switch))
                    # f.write('\n')
                    # 计算的第六个指令,control
                    f.write('')
                    # 10代表写  11代表读
                    f.write('1000000000000001')  # 01参数状态（权重）写寄存器
                    f.write('\n')
                    f.write('110000040000000F')  # 02结束    读寄存器
                    f.write('\n')
                # ----------------conv33计算指令-------------------
                # 计算图片的数量,单位是B
                # 12 320 320
                feature_shape = feature.shape
                feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]  # 图片大小
                # 计算写地址
                computer_write_address = computer_address + add_write_address  # 计算地址+add_write_address=写地址
                # 计算输出图片的大小
                out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1
                # 计算写地址的数量
                write_size = feature_shape[0] * shape[
                    0] * out_size * out_size  # bchw结果大小 单位B  结果的shape为[feature_shape[0],shape[0],out_size,out_size]
                # ------------------写入计算的指令----------------------
                with open(file_name, 'a+') as fp:
                    # 计算的第一个指令读地址
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    fp.write('%08X' % int(computer_address))
                    fp.write('\n')
                    # 计算的第二个指令读数量
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    fp.write('%08X' % int(feature_size))  # 读图片
                    fp.write('\n')
                    # 计算的第三个指令写地址
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                    fp.write('%08X' % int(computer_write_address))
                    fp.write('\n')
                    # 计算的第四个指令写数量
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                    fp.write('%08X' % int(write_size))  # 写卷积后的结果
                    fp.write('\n')
                    # # 计算的第五个指令reg4
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第六个指令reg5
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第七个指令reg6,33卷积reg6的8位不解析写啥都行
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # 计算的第八个指令reg7,33卷积reg7的后四位(共八位)没用
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第九个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第十个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第十一个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第十二个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第十三个指令,switch
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # 计算的第十四个指令,control
                    fp.write('100000' + '00')
                    fp.write('%08X' % int(2))  # 02表示计算（卷积）
                    fp.write('\n')
                    fp.write('110000040000000F')
                    fp.write('\n')

                weight_address = weight_address + weight_size  # 原本读地址+读了多少=下此开始读的地址
                return weight_address, computer_write_address  # 返回加完的计算写地址computer_write_address 当下层的computer_address
            elif (operator == "conv33"):  # 其他层是conv33
                shape = q_weight.shape
                print("q_weight_shape=", shape)
                # 计算权重的数量（计算weight+bias+scale+shift，单位是B）
                weight_size = (shape[0] * shape[1] * shape[2] * shape[3])  # m*c*k*k个卷积点 一个点是8bit=1B 单位B
                weight_size += ((shape[0]) * 3 * 4)  # shape[0]是m输出通道个数，bias+scale+shift是3，每个数是32bit=4B
                # print('%08X' % weight_size)
                # weight_size为权重的数量
                # ----------------conv33权重指令-------------------
                # 计算权重的reg4
                # dataSizeW是权重每行多少bit, dataSizeB是bias每行多少bit ；8入8出 8*8=64bit
                dataSizeW = 64
                dataSizeB = 64
                reg4 = conv33para(shape[0], shape[1], dataSizeW, dataSizeB)
                # 计算权重的reg5
                reg5 = '00000000'
                print("weight_size=", '%08x' % weight_size)
                # 权重第一个指令:读地址
                with open(file_name, 'a+') as f:

                    f.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    f.write('%08X' % int(weight_address))
                    f.write('\n')
                    # 权重第二个指令:读数量
                    f.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    f.write('%08X' % weight_size)
                    f.write('\n')
                    # 权重的第三个指令reg4
                    f.write('100000' + ins_address['TJPU_Reg4'])
                    f.write(str(reg4))
                    f.write('\n')
                    # 权重的第四个指令reg5
                    f.write('100000' + ins_address['TJPU_Reg5'])
                    f.write(str(reg5))
                    f.write('\n')
                    # 计算的第五个指令,switch
                    f.write('100000' + ins_address['TJPU_Switch'])
                    f.write('%08X' % int(1))
                    f.write('\n')
                    # 计算的第六个指令,control
                    f.write('100000' + ins_address['TJPU_Control'])
                    f.write('%08X' % int(1))
                    f.write('\n')
                    f.write('110000140000000F')
                    f.write('\n')

                # ----------------conv33计算指令-------------------
                # 计算图片的数量,单位是B
                feature_shape = feature.shape
                print("feature_shape=", feature_shape)
                feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
                # 计算写地址
                computer_write_address = computer_address + add_write_address * index
                print("computer_write_address22222", '%08x' % computer_write_address)
                # 计算输出图片的大小
                out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1
                # 计算写地址的数量
                write_size = feature_shape[0] * shape[0] * out_size * out_size
                print("write_size= ", '%08X' % write_size)
                if write_size >= add_write_address:
                    print("超过大小")
                print("==============conv33===============")
                # 计算的reg4 计算reg5
                computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv33compute(shape[0], shape[1],
                                                                                           dataSizeW,
                                                                                           dataSizeB, feature.shape[2],
                                                                                           stride, padding,
                                                                                           self.quant_zero_point1,
                                                                                           self.quant_zero_point3,
                                                                                           self.quant_scale3)
                # print("computer_reg5",bin(int(computer_reg5)))
                # ------------------写入权重的指令----------------------

                # ---p---------------写入计算的指令----------------------
                with open(file_name, 'a+') as fp:

                    # 计算的第一个指令读地址
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    fp.write('%08X' % int(computer_address))  # 读图片

                    fp.write('\n')
                    # 计算的第二个指令读数量
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    fp.write('%08X' % int(feature_size))
                    fp.write('\n')
                    # 计算的第三个指令写地址
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                    fp.write('%08X' % int(computer_write_address))
                    fp.write('\n')
                    # 计算的第四个指令写数量
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                    fp.write('%08X' % int(write_size))
                    fp.write('\n')
                    # 计算的第五个指令reg4
                    fp.write('100000' + ins_address['TJPU_Reg4'])
                    fp.write('%08X' % int(computer_reg4))
                    fp.write('\n')
                    # 计算的第六个指令reg5
                    fp.write('100000' + ins_address['TJPU_Reg5'])
                    fp.write('%08X' % int(computer_reg5))
                    fp.write('\n')
                    # 计算的第七个指令reg6,33卷积reg6的8位不解析写啥都行

                    fp.write('100000' + ins_address['TJPU_Reg6'])
                    fp.write('%08X' % int(computer_reg6))
                    fp.write('\n')
                    # 计算的第八个指令reg7,33卷积reg7的后四位(共八位)没用
                    fp.write('100000' + ins_address['TJPU_Reg7'])
                    fp.write('%08X' % int(computer_reg7))
                    fp.write('\n')
                    # # 计算的第九个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第十个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第十一个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # # 计算的第十二个指令,默认
                    # fp.write('' + str(default_data))
                    # fp.write('\n')
                    # 计算的第十三个指令,switch
                    fp.write('100000' + ins_address['TJPU_Switch'])
                    fp.write('%08X' % int(1))
                    fp.write('\n')
                    # 计算的第十四个指令,control
                    fp.write('100000' + ins_address['TJPU_Control'])
                    fp.write('%08X' % int(2))
                    fp.write('\n')
                    fp.write('110000140000000F')
                    fp.write('\n')

                weight_address = weight_address + weight_size
                return weight_address, computer_write_address  #
            elif (operator == "concat"):  # 结果存为cat1跟cat2（出入图片）拼接
                cat1_shape = feature.shape
                cat2_shape = cat2.shape
                print("cat1_shape", cat1_shape)
                print("cat2_shape", cat2_shape)
                cat1_address = computer_address
                cat2_address = cat2_weight_address
                # cat1_shape = int((cat1_shape[2] - 3 + 2 * padding) / stride) + 1
                cat1_size = cat1_shape[0] * cat1_shape[1] * cat1_shape[2] * cat1_shape[3]
                cat2_size = cat2_shape[0] * cat2_shape[1] * cat2_shape[2] * cat2_shape[3]
                # 计算写地址
                computer_write_address = cat1_address + add_write_address * index
                # 计算concat的reg4
                cat1_channel = '{:010b}'.format(
                    cat2_shape[1])  # cat2通道数b c h w  通过 {} 和 : 来代替以前的 % b、d、o、x 分别是二进制、十进制、八进制、十六进制。不足10位补0
                # 下面是二进制
                reg4 = cat1_channel + '0000000000000000000000'
                # 将二进制转成十进制
                reg4 = str(int(reg4, 2))
                # 计算concat的reg5()
                feature_h = '{:011b}'.format(cat2_shape[2])  # 11 位 输入图片 高  (行数)
                cat2_channel = '{:010b}'.format(cat1_shape[1])  # 10 位，concat 中 cat1 的通道数
                feature_w = '{:011b}'.format(cat2_shape[3])  # 11 位，输入的图片 宽  (列数)
                # 下面位二进制
                reg5 = feature_h + cat2_channel + feature_w
                # 将二进制转成10进制
                reg5 = str(int(reg5, 2))
                # 计算reg6,reg7,reg8,reg9
                reg6, reg7, reg8, reg9 = reg_cat(self.quant_scale1, self.quant_scale2, self.quant_scale3,
                                                 self.quant_zero_point1,
                                                 self.quant_zero_point2, self.quant_zero_point3)
                print("reg4", reg4, "reg5", reg5, "reg6", reg6, "reg7", reg7, "reg8", reg8, "reg9", reg9)
                default_data = 1300005000000000

                ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                               'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                               'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                               'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                               'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                               'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                               'Image_Reg1': '0C'}
                print("cat1_address=", '%08x' % cat1_address)
                print("cat1_size=", '%08x' % cat1_size)
                print("cat2_address=", '%08x' % cat2_address)
                print("cat2_size=", '%08x' % cat2_size)
                print("computer_write_address=", '%08x' % computer_write_address)
                print("cat2_size+cat1_size=", '%08x' % (cat2_size + cat1_size))
                print("=======================concat========================")
                # ----------------concat权重指令-------------------
                # 6个全都是默认的
                with open(file_name, 'a+') as fp:
                    # -------------concat的计算指令--------------
                    # 第一个指令:读第一个concat地址
                    # print('%08X' % cat2_address)
                    # exit()
                    fp.write('100000' + ins_address['Image_Reg0'])
                    fp.write('%08X' % cat1_address)
                    fp.write('\n')
                    # 第二个指令:读第一个concat数量
                    fp.write('100000' + ins_address['Image_Reg1'])
                    fp.write('%08X' % cat1_size)
                    fp.write('\n')
                    # 第三个指令:读第二个concat地址
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    fp.write('%08X' % cat2_address)
                    fp.write('\n')
                    # 第四个指令:读第二个concat数量
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    fp.write('%08X' % cat2_size)
                    fp.write('\n')
                    # 第五个指令:写地址
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                    fp.write('%08X' % computer_write_address)
                    fp.write('\n')
                    # 第六个指令:写数量
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                    fp.write('%08X' % (cat2_size + cat1_size))
                    fp.write('\n')
                    # 第七个指令:reg4
                    fp.write('100000' + ins_address['TJPU_Reg4'])
                    fp.write('%08X' % int(reg4))
                    fp.write('\n')
                    # 第八个指令:reg5
                    fp.write('100000' + ins_address['TJPU_Reg5'])
                    fp.write('%08X' % int(reg5))
                    fp.write('\n')
                    # 第九个指令:reg6
                    fp.write('100000' + ins_address['TJPU_Reg6'])
                    fp.write('%08X' % int(reg6))
                    fp.write('\n')
                    # 第十个指令:reg7
                    fp.write('100000' + ins_address['TJPU_Reg7'])
                    fp.write('%08X' % int(reg7))
                    fp.write('\n')
                    # 第十一个指令:reg8
                    fp.write('100000' + ins_address['TJPU_Reg8'])
                    fp.write('%08X' % int(reg8))
                    fp.write('\n')
                    # 第十二个指令:reg9
                    fp.write('100000' + ins_address['TJPU_Reg9'])
                    fp.write('%08X' % int(reg9))
                    fp.write('\n')
                    # 第十三个指令:switch
                    fp.write('100000' + ins_address['TJPU_Switch'])
                    fp.write('00000008')
                    fp.write('\n')
                    # 第十四个指令:control
                    fp.write('100000' + ins_address['TJPU_Control'])
                    ## 十六进制的0001 -> 二进制的0001 是concat的控制信号
                    fp.write('00000100')
                    fp.write('\n')
                    fp.write('1100001400000F00')
                    fp.write('\n')
                return weight_address, computer_write_address
            elif (operator == "add"):
                cat1_shape = feature.shape
                cat2_shape = cat2.shape
                cat1_address = computer_address
                cat2_address = cat2_weight_address
                # cat1_shape = int((cat1_shape[2] - 3 + 2 * padding) / stride) + 1
                cat1_size = cat1_shape[0] * cat1_shape[1] * cat1_shape[2] * cat1_shape[3]
                cat2_size = cat2_shape[0] * cat2_shape[1] * cat2_shape[2] * cat2_shape[3]
                # 计算写地址
                computer_write_address = cat1_address + add_write_address
                # 计算concat的reg4
                cat1_channel = '{:010b}'.format(
                    cat2_shape[1])  # cat2通道数b c h w  通过 {} 和 : 来代替以前的 % b、d、o、x 分别是二进制、十进制、八进制、十六进制。不足10位补0
                # 下面是二进制
                reg4 = cat1_channel + '0000000000000000000000'
                # 将二进制转成十进制
                reg4 = str(int(reg4, 2))
                # 计算concat的reg5()
                feature_h = '{:011b}'.format(cat2_shape[2])  # 11 位 输入图片 高  (行数)
                cat2_channel = '{:010b}'.format(cat1_shape[1])  # 10 位，concat 中 cat1 的通道数
                feature_w = '{:011b}'.format(cat2_shape[3])  # 11 位，输入的图片 宽  (列数)
                # 下面位二进制
                reg5 = feature_h + cat2_channel + feature_w
                # 将二进制转成10进制
                reg5 = str(int(reg5, 2))
                print("reg4", '%08X' % int(reg4))
                print("reg5", '%08X' % int(reg5))
                # 计算reg6,reg7,reg8,reg9
                reg6, reg7, reg8, reg9 = reg_add(self.quant_scale1, self.quant_scale2, self.quant_scale3,
                                                 self.quant_zero_point1,
                                                 self.quant_zero_point2, self.quant_zero_point3)
                print("reg6", '%08X' % reg6, "reg7", '%08X' % reg7, "reg8", '%08X' % reg8, "reg9", '%08X' % reg9)
                default_data = 1300005000000000

                ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                               'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                               'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                               'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                               'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                               'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                               'Image_Reg1': '0C'}
                print("add1_address=", '%08x' % cat1_address)
                print("add1_size=", '%08x' % cat1_size)
                print("add2_address=", '%08x' % cat2_address)
                print("add2_size=", '%08x' % cat2_size)
                print("computer_write_address=", '%08x' % computer_write_address)
                print("add2_size+add1_size=", '%08x' % (cat2_size + cat1_size))
                print("=======================add========================")
                # ----------------concat权重指令-------------------
                # 6个全都是默认的
                with open(file_name, 'a+') as fp:
                    # -------------concat的计算指令--------------
                    # 第一个指令:读第一个concat地址
                    # print('%08X' % cat2_address)
                    # exit()
                    fp.write('100000' + ins_address['Image_Reg0'])
                    fp.write('%08X' % cat1_address)
                    fp.write('\n')
                    # 第二个指令:读第一个concat数量
                    fp.write('100000' + ins_address['Image_Reg1'])
                    fp.write('%08X' % cat1_size)
                    fp.write('\n')
                    # 第三个指令:读第二个concat地址
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    fp.write('%08X' % cat2_address)
                    fp.write('\n')
                    # 第四个指令:读第二个concat数量
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    fp.write('%08X' % cat2_size)
                    fp.write('\n')
                    # 第五个指令:写地址
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                    fp.write('%08X' % computer_write_address)
                    fp.write('\n')
                    # 第六个指令:写数量
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                    fp.write('%08X' % (cat2_size))
                    fp.write('\n')
                    # 第七个指令:reg4
                    fp.write('100000' + ins_address['TJPU_Reg4'])
                    fp.write('%08X' % int(reg4))
                    fp.write('\n')
                    # 第八个指令:reg5
                    fp.write('100000' + ins_address['TJPU_Reg5'])
                    fp.write('%08X' % int(reg5))
                    fp.write('\n')
                    # 第九个指令:reg6
                    fp.write('100000' + ins_address['TJPU_Reg6'])
                    fp.write('%08X' % int(reg6))
                    fp.write('\n')
                    # 第十个指令:reg7
                    fp.write('100000' + ins_address['TJPU_Reg7'])
                    fp.write('%08X' % int(reg7))
                    fp.write('\n')
                    # 第十一个指令:reg8
                    fp.write('100000' + ins_address['TJPU_Reg8'])
                    fp.write('%08X' % int(reg8))
                    fp.write('\n')
                    # 第十二个指令:reg9
                    fp.write('100000' + ins_address['TJPU_Reg9'])
                    fp.write('%08X' % int(reg9))
                    fp.write('\n')
                    # 第十三个指令:switch
                    fp.write('100000' + ins_address['TJPU_Switch'])
                    fp.write('00000008')
                    fp.write('\n')
                    # 第十四个指令:control
                    fp.write('100000' + ins_address['TJPU_Control'])
                    # 十六进制的0009 -> 二进制的1001 是add的控制信号  
                    fp.write('00000900')
                    fp.write('\n')
                    fp.write('1100001400000F00')
                    fp.write('\n')
                return weight_address, computer_write_address
            elif (operator == "conv11"):
                shape = q_weight.shape
                print("q_weight_shape=", shape)
                dataSizeW = 64
                dataSizeB = 64
                # 计算权重的数量
                weight_size = (shape[0] * shape[1] * shape[2] * shape[3])
                weight_size += ((shape[0]) * 3 * 4)
                # 权重11conv的reg4
                reg4 = conv11para(shape[0], shape[1], dataSizeW, dataSizeB)
                print("weight_size=", '%08x' % weight_size)
                # 权重reg5的二进制32位全0
                # -------------conv11权重指令--------------
                # 权重第一个指令:读地址
                with open(file_name, 'a+') as f:
                    f.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    f.write('%08X' % int(weight_address))
                    f.write('\n')
                    # 权重第二个指令:读数量
                    f.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    f.write('%08X' % weight_size)
                    # print(weight_size)
                    # exit()
                    f.write('\n')
                    # 权重的第三个指令reg4
                    f.write('100000' + ins_address['TJPU_Reg4'])
                    f.write('%08X' % int(reg4))
                    f.write('\n')
                    # 权重的第四个指令reg5
                    f.write('100000' + ins_address['TJPU_Reg5'])
                    f.write('00000000')
                    f.write('\n')
                    # 计算的第五个指令,switch
                    f.write('100000' + ins_address['TJPU_Switch'])
                    f.write('%08X' % int(2))
                    f.write('\n')
                    # 计算的第六个指令,control
                    f.write('100000' + ins_address['TJPU_Control'])
                    f.write('%08X' % int(16))
                    f.write('\n')
                    f.write('11000014000000F0')
                    f.write('\n')
                weight_address = weight_address + weight_size
                # 计算读地址数量
                feature_shape = feature.shape
                feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
                # 计算写地址
                computer_write_address = computer_address + add_write_address * index
                print('computer_write_address_conv2=', '%08X' % computer_write_address)
                # 计算输出图片的大小
                out_size = int((feature_shape[2] - 1 + 2 * padding) / stride) + 1
                # 计算写地址的数量
                write_size = feature_shape[0] * shape[0] * out_size * out_size
                print("write_size= ", '%08X' % write_size)
                if write_size >= add_write_address:
                    print("超过大小")
                print("===============conv11================")
                # print(out_size)
                # exit()
                # 计算11conv的reg4,reg5,reg6,reg7
                # isleakrelu:是否需要 leakyrelu 。 1 为不用，0 为用
                computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv11compute(shape[0], shape[1],
                                                                                           dataSizeW,
                                                                                           dataSizeB, feature.shape[2],
                                                                                           stride, padding,
                                                                                           self.quant_zero_point1,
                                                                                           self.quant_zero_point3,
                                                                                           isleakrelu,
                                                                                           self.quant_scale3)
                # print(self.coe_name)
                # head的1x1 没做leakyrelu
                if self.coe_name == '../data1_coe/out_hand_cls_preds0.coe':
                    computer_reg6 = 0
                # print(self.coe_name)
                if self.coe_name == '../data1_coe/out_hand_reg_preds0.coe':
                    computer_reg6 = 0
                if self.coe_name == '../data1_coe/out_hand_obj_preds0.coe':
                    computer_reg6 = 0

                if self.coe_name == '../data1_coe/out_hand_cls_preds1.coe':
                    computer_reg6 = 0
                # print(self.coe_name)
                if self.coe_name == '../data1_coe/out_hand_reg_preds1.coe':
                    computer_reg6 = 0
                if self.coe_name == '../data1_coe/out_hand_obj_preds1.coe':
                    computer_reg6 = 0

                if self.coe_name == '../data1_coe/out_hand_cls_preds2.coe':
                    computer_reg6 = 0
                # print(self.coe_name)
                if self.coe_name == '../data1_coe/out_hand_reg_preds2.coe':
                    computer_reg6 = 0
                if self.coe_name == '../data1_coe/out_hand_obj_preds2.coe':
                    computer_reg6 = 0
                # -------------conv11计算指令--------------
                with open(file_name, 'a+') as fp:
                    # 计算的第一个指令读地址
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    fp.write('%08X' % int(computer_address))
                    fp.write('\n')
                    # 计算的第二个指令读数量
                    fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    fp.write('%08X' % int(feature_size))
                    fp.write('\n')
                    # 计算的第三个指令写地址
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                    fp.write('%08X' % int(computer_write_address))
                    fp.write('\n')
                    # 计算的第四个指令写数量
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                    fp.write('%08X' % int(write_size))
                    fp.write('\n')
                    # 计算的第五个指令reg4
                    fp.write('100000' + ins_address['TJPU_Reg4'])
                    fp.write('%08X' % int(computer_reg4))
                    fp.write('\n')
                    # 计算的第六个指令reg5
                    fp.write('100000' + ins_address['TJPU_Reg5'])
                    fp.write('%08X' % int(computer_reg5))
                    fp.write('\n')
                    # 计算的第七个指令reg6,reg6的8位不解析写啥都行(所有都没用)
                    fp.write('100000' + ins_address['TJPU_Reg6'])
                    fp.write('%08X' % int(computer_reg6))
                    fp.write('\n')
                    # 计算的第八个指令reg7,reg7的第七位没用
                    fp.write('100000' + ins_address['TJPU_Reg7'])
                    fp.write('%08X' % int(computer_reg7))
                    fp.write('\n')
                    # 计算的第十三个指令,switch
                    fp.write('100000' + ins_address['TJPU_Switch'])
                    fp.write('%08X' % int(2))
                    fp.write('\n')
                    # 计算的第十四个指令,control
                    fp.write('100000' + ins_address['TJPU_Control'])
                    fp.write('%08X' % int(32))
                    fp.write('\n')
                    fp.write('11000014000000F0')
                    fp.write('\n')
                # 返回下一次要读的权重地址，结果写入的开始地址
                return weight_address, computer_write_address


def conv33_block(coe_name, weight_address, computer_address, feature, f_weight, q_weight, bias, path1, stride, padding,
                 block, cat2_weight_address, isleakrelu, add_write_address, operator, quant_zero_point1,
                 quant_zero_point3,
                 quant_scale3):  # 分成4块，然后做concat
    default_data = 1300005000000000
    file_name = '../ins/yolox_ins_64_682.dat'  # 写入的文件名
    # reg表对应的地址
    ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                   'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                   'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                   'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                   'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                   'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                   'Image_Reg1': '0C'}
    # exit()
    if block != 0:  # 如果分块
        #  512 256 3 3
        shape = q_weight.shape  # m输出通道数 c输入通道数 k卷积核大小 k
        # 计算权重的数量
        channel_num = shape[0] / block  # 分块后的m
        weight_size = (channel_num * shape[1] * shape[2] * shape[3])  # m*c*k*k 每块总的权重的大小 单位是B
        weight_size += (channel_num * 3 * 4)  # 加上bias scale shift 每个数32bit-->4B
        # weight_size为权重的数量

        # ----------------conv33权重指令-------------------
        # 计算权重的reg4
        # dataSizeW是权重每行多少bit, dataSizeB是bias每行多少bit
        dataSizeW = 64  # 64/8=8  每行8个数
        dataSizeB = 64
        reg4 = conv33para(channel_num, shape[1], dataSizeW, dataSizeB)
        # 计算权重的reg5
        reg5 = '00000000'
        # 权重第一个指令:读地址
        # ----------------conv33计算指令-------------------
        # 计算图片的数量,单位是B
        feature_shape = feature.shape
        feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
        # 计算写地址
        computer_write_address = computer_address + add_write_address
        # 计算输出图片的大小
        out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1  # (n+2p-f)/s+1 feature_shape[2]输入通道大小
        # 计算写地址的数量
        write_size = feature_shape[0] * channel_num * out_size * out_size  # channel_numf分块后每块输出通道大小
        # 计算的reg4 计算reg5
        computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv33compute(channel_num, shape[1],
                                                                                   dataSizeW,
                                                                                   dataSizeB, feature.shape[2],
                                                                                   stride, padding,
                                                                                   quant_zero_point1,
                                                                                   quant_zero_point3,
                                                                                   quant_scale3)
        if coe_name == '../yolo_head_p5_13_coe/hand_P5_13_conv33.coe':  # P5往后移了6个add_write_address，因为被其他占了位置
            add_new = 6

        else:
            add_new = 0
        for index in range(block):
            # ------------------写入计算的指令----------------------
            with open(file_name, 'a+') as fp:
                # 权重第二个指令:读地址
                fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                fp.write('%08X' % int(weight_address))
                print("=====================block=========================")
                print('weight_address=', '%08X' % int(weight_address))
                fp.write('\n')
                # 权重第二个指令:读数量
                fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                fp.write('%08X' % int(weight_size))
                fp.write('\n')
                # 权重的第三个指令reg4
                fp.write('100000' + ins_address['TJPU_Reg4'])
                fp.write(str(reg4))
                fp.write('\n')
                # 权重的第四个指令reg5
                fp.write('100000' + ins_address['TJPU_Reg5'])
                fp.write(str(reg5))
                fp.write('\n')
                # 计算的第五个指令,switch
                fp.write('100000' + ins_address['TJPU_Switch'])
                fp.write('%08X' % int(1))
                fp.write('\n')
                # 计算的第六个指令,control
                fp.write('100000' + ins_address['TJPU_Control'])
                fp.write('%08X' % int(1))
                fp.write('\n')
                fp.write('110000140000000F')
                fp.write('\n')

                # ------------------计算的第一个指令读地址------------------
                # 计算的第一个指令读地址
                fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                fp.write('%08X' % computer_address)
                print('computer_address=', '%08X' % computer_address)
                fp.write('\n')
                # 计算的第二个指令读数量
                fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                fp.write('%08X' % int(feature_size))
                fp.write('\n')
                # 计算的第三个指令写地址
                fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                fp.write('%08X' % int(computer_write_address + write_size * index + 16777216 * add_new))
                print('computer_write_address_conv2=',
                      '%08X' % int(computer_write_address + write_size * index + 16777216 * add_new))
                print("=====================================================")
                fp.write('\n')
                # 计算的第四个指令写数量
                fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                fp.write('%08X' % int(write_size))
                fp.write('\n')
                # 计算的第五个指令reg4
                fp.write('100000' + ins_address['TJPU_Reg4'])
                fp.write('%08X' % int(computer_reg4))
                fp.write('\n')
                # 计算的第六个指令reg5
                fp.write('100000' + ins_address['TJPU_Reg5'])
                fp.write('%08X' % int(computer_reg5))
                fp.write('\n')
                # 计算的第七个指令reg6,33卷积reg6的8位不解析写啥都行

                fp.write('100000' + ins_address['TJPU_Reg6'])
                fp.write('%08X' % int(computer_reg6))
                fp.write('\n')
                # 计算的第八个指令reg7,33卷积reg7的后四位(共八位)没用
                fp.write('100000' + ins_address['TJPU_Reg7'])
                fp.write('%08X' % int(computer_reg7))
                fp.write('\n')

                # 计算的第十三个指令,switch
                fp.write('100000' + ins_address['TJPU_Switch'])
                fp.write('%08X' % int(1))
                fp.write('\n')
                # 计算的第十四个指令,control
                fp.write('100000' + ins_address['TJPU_Control'])
                fp.write('%08X' % int(2))
                fp.write('\n')
                fp.write('110000140000000F')
                fp.write('\n')
            weight_address = weight_address + weight_size
        for cat_index in range(int(block / 2)):  # 先做两次两个的concat
            # if coe_name == '../yolo_head_p5_13_coe/hand_P5_13_conv33.coe' and cat_index!=0 and int(block/4)!=0:
            #     add_new = 8
            #
            # else:
            #     add_new = 0
            # 计算concat的reg4
            cat1_channel = '{:010b}'.format(int(channel_num))  # cat2通道数
            # 下面是二进制
            reg4 = cat1_channel + '0000000000000000000000'
            # 将二进制转成十进制
            reg4 = str(int(reg4, 2))
            # 计算concat的reg5()
            feature_h = '{:011b}'.format(out_size)  # 11 位 输入图片 高  (行数)
            cat2_channel = '{:010b}'.format(int(channel_num))  # 10 位，concat 中 cat1 的通道数
            feature_w = '{:011b}'.format(out_size)  # 11 位，输入的图片 宽  (列数)
            # 下面位二进制
            reg5 = feature_h + cat2_channel + feature_w
            # 将二进制转成10进制
            reg5 = str(int(reg5, 2))
            # 计算reg6,reg7,reg8,reg9
            reg6 = 65536
            reg7 = 65536
            reg8 = 0
            reg9 = 0
            default_data = 1300005000000000

            # ----------------concat权重指令-------------------
            # 6个全都是默认的
            with open(file_name, 'a+') as fp:
                # -------------concat的计算指令--------------
                # 第一个指令:读第一个concat地址
                fp.write('100000' + ins_address['Image_Reg0'])
                # if cat_index==0:
                #     add_new=0

                fp.write('%08X' % int(computer_write_address + write_size * cat_index * 2 + 16777216 * add_new))
                print("===========================cat=============================")
                print('computer_write_address_cat1_block=',
                      '%08X' % int(computer_write_address + write_size * cat_index * 2 + 16777216 * add_new))
                fp.write('\n')
                # 第二个指令:读第一个concat数量
                fp.write('100000' + ins_address['Image_Reg1'])
                fp.write('%08X' % int(write_size))
                fp.write('\n')
                # 第三个指令:读第二个concat地址
                fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                fp.write(
                    '%08X' % int(computer_write_address + write_size * cat_index * 2 + write_size + 16777216 * add_new))
                print('computer_write_address_cat2_block=',
                      '%08X' % int(
                          computer_write_address + write_size * cat_index * 2 + write_size + 16777216 * add_new))
                fp.write('\n')
                # 第四个指令:读第二个concat数量
                fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                fp.write('%08X' % int(write_size))
                fp.write('\n')
                # 第五个指令:写地址
                fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                fp.write(
                    '%08X' % int(computer_write_address + 16777216 + cat_index * write_size * 2 + 16777216 * add_new))
                fp.write('\n')
                print('computer_write_address_cat3_block=',
                      '%08X' % int(computer_write_address + 16777216 + cat_index * write_size * 2 + 16777216 * add_new))
                print("============================================================================================")
                # 第六个指令:写数量
                fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                fp.write('%08X' % int(write_size * 2))  # concat里有两个
                fp.write('\n')
                # 第七个指令:reg4
                fp.write('100000' + ins_address['TJPU_Reg4'])
                fp.write('%08X' % int(reg4))
                fp.write('\n')
                # 第八个指令:reg5
                fp.write('100000' + ins_address['TJPU_Reg5'])
                fp.write('%08X' % int(reg5))
                fp.write('\n')
                # 第九个指令:reg6
                fp.write('100000' + ins_address['TJPU_Reg6'])
                fp.write('%08X' % int(reg6))
                fp.write('\n')
                # 第十个指令:reg7
                fp.write('100000' + ins_address['TJPU_Reg7'])
                fp.write('%08X' % int(reg7))
                fp.write('\n')
                # 第十一个指令:reg8
                fp.write('100000' + ins_address['TJPU_Reg8'])
                fp.write('%08X' % int(reg8))
                fp.write('\n')
                # 第十二个指令:reg9
                fp.write('100000' + ins_address['TJPU_Reg9'])
                fp.write('%08X' % int(reg9))
                fp.write('\n')
                # 第十三个指令:switch
                fp.write('100000' + ins_address['TJPU_Switch'])
                fp.write('00000008')
                fp.write('\n')
                # 第十四个指令:control
                fp.write('100000' + ins_address['TJPU_Control'])
                fp.write('00000100')
                fp.write('\n')
                fp.write('1100001400000F00')
                fp.write('\n')
        # if block==2:
        if int(block / 4) == 0:  # 再做一次两个的concat
            return weight_address, computer_write_address + 16777216 + 16777216 * add_new
        for cat_index in range(int(block / 4)):
            # print('***********************************')
            # 计算concat的reg4
            cat1_channel = '{:010b}'.format(int(channel_num * 2))  # cat2通道数
            # 下面是二进制
            reg4 = cat1_channel + '0000000000000000000000'
            # 将二进制转成十进制
            reg4 = str(int(reg4, 2))
            # 计算concat的reg5()
            feature_h = '{:011b}'.format(out_size)  # 11 位 输入图片 高  (行数)
            cat2_channel = '{:010b}'.format(int(channel_num * 2))  # 10 位，concat 中 cat1 的通道数
            feature_w = '{:011b}'.format(out_size)  # 11 位，输入的图片 宽  (列数)
            # 下面位二进制
            reg5 = feature_h + cat2_channel + feature_w
            # 将二进制转成10进制
            reg5 = str(int(reg5, 2))
            # 计算reg6,reg7,reg8,reg9
            reg6 = 65536
            reg7 = 65536
            reg8 = 0
            reg9 = 0
            default_data = 1300005000000000

            # ----------------concat权重指令-------------------
            # 6个全都是默认的
            with open(file_name, 'a+') as fp:
                # -------------concat的计算指令--------------
                # 第一个指令:读第一个concat地址
                fp.write('100000' + ins_address['Image_Reg0'])
                fp.write('%08X' % int(computer_write_address + 16777216 + 16777216 * add_new))
                fp.write('\n')
                # 第二个指令:读第一个concat数量
                fp.write('100000' + ins_address['Image_Reg1'])
                fp.write('%08X' % int(write_size * 2))  # 第一个concat有两个
                fp.write('\n')
                # 第三个指令:读第二个concat地址
                fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                fp.write('%08X' % int(computer_write_address + 16777216 + write_size * 2 + 16777216 * add_new))
                fp.write('\n')
                # 第四个指令:读第二个concat数量
                fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                fp.write('%08X' % int(write_size * 2))
                fp.write('\n')
                # 第五个指令:写地址
                fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                fp.write('%08X' % int(computer_write_address + 16777216 * 2 + 16777216 * add_new))
                fp.write('\n')
                computer_write_address = computer_write_address + 16777216 * 2 + 16777216 * add_new
                # print(computer_write_address)
                # print('-----------------------------------************************************')
                # 第六个指令:写数量
                fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                fp.write('%08X' % int(write_size * 4))  # 一共四个
                fp.write('\n')
                # 第七个指令:reg4
                fp.write('100000' + ins_address['TJPU_Reg4'])
                fp.write('%08X' % int(reg4))
                fp.write('\n')
                # 第八个指令:reg5
                fp.write('100000' + ins_address['TJPU_Reg5'])
                fp.write('%08X' % int(reg5))
                fp.write('\n')
                # 第九个指令:reg6
                fp.write('100000' + ins_address['TJPU_Reg6'])
                fp.write('%08X' % int(reg6))
                fp.write('\n')
                # 第十个指令:reg7
                fp.write('100000' + ins_address['TJPU_Reg7'])
                fp.write('%08X' % int(reg7))
                fp.write('\n')
                # 第十一个指令:reg8
                fp.write('100000' + ins_address['TJPU_Reg8'])
                fp.write('%08X' % int(reg8))
                fp.write('\n')
                # 第十二个指令:reg9
                fp.write('100000' + ins_address['TJPU_Reg9'])
                fp.write('%08X' % int(reg9))
                fp.write('\n')
                # 第十三个指令:switch
                fp.write('100000' + ins_address['TJPU_Switch'])
                fp.write('00000008')
                fp.write('\n')
                # 第十四个指令:control
                fp.write('100000' + ins_address['TJPU_Control'])
                fp.write('00000100')
                fp.write('\n')
                fp.write('1100001400000F00')
                fp.write('\n')

        return weight_address, computer_write_address

    # elif block == 2:
    #
    #    exit()


def reg_cat(cat1_scale, cat2_scale, cat3_scale, cat1_zero_point, cat2_zero_point, cat3_zero_point):  # 做量化 两组r3=r1和r3=r2
    zero_point_one = (cat3_scale / cat1_scale) * cat3_zero_point - cat1_zero_point  # 定义Z
    zero_point_one = (torch.round(zero_point_one * (2 ** 16)))  # 乘2^16
    zero_point_one = zero_point_one.numpy().astype(np.uint32)
    M1 = (torch.round((cat1_scale / cat3_scale) * (2 ** 16)))  # 定义M
    M1 = M1.numpy().astype(np.uint32)
    # print('%08x' % zero_point_one)
    # print('---------------------')
    # print('%08x' % M1)
    zero_point_two = (cat3_scale / cat2_scale) * cat3_zero_point - cat2_zero_point
    zero_point_two = (torch.round(zero_point_two * (2 ** 16)))
    zero_point_two = zero_point_two.numpy().astype(np.uint32)
    M2 = (torch.round((cat2_scale / cat3_scale) * (2 ** 16)))
    M2 = M2.numpy().astype(np.uint32)
    # print('%08x' % zero_point_two)
    # print('%08x' % M2)
    return M1, M2, zero_point_one, zero_point_two


def reg_add(cat1_scale, cat2_scale, cat3_scale, cat1_zero_point, cat2_zero_point, cat3_zero_point):
    # ===============s3/s1*z3-z1======================
    zero_point_one = (cat3_scale / cat1_scale) * cat3_zero_point - cat1_zero_point
    zero_point_one = (torch.round(zero_point_one * (2 ** 16)))
    zero_point_one = zero_point_one.numpy().astype(np.uint32)
    M1 = (torch.round((cat1_scale / cat3_scale) * (2 ** 16)))
    M1 = M1.numpy().astype(np.uint32)
    # print('%08x' % zero_point_one)
    # print('---------------------')
    # print('%08x' % M1)
    # =================z2================
    zero_point_two = 0 - cat2_zero_point
    zero_point_two = torch.as_tensor(zero_point_two, dtype=torch.float32)
    zero_point_two = (torch.round(zero_point_two * (2 ** 16)))
    zero_point_two = zero_point_two.numpy().astype(np.uint32)
    M2 = (torch.round((cat2_scale / cat3_scale) * (2 ** 16)))
    M2 = M2.numpy().astype(np.uint32)
    # print('%08x' % zero_point_two)
    # print('%08x' % M2)
    #     reg6  reg7  reg8           reg9
    return M1, M2, zero_point_one, zero_point_two


def reshape(weight_address, computer_address, feature, operator="", add_write_address=16777216, filename=' '):
    default_data = 1300005000000000
    # 计算读的数量
    shape = feature.shape  # b c  h w

    channel_in = shape[1]
    feature_in = shape[2]  # 图片大小
    # if operator == "upsample":
    #     feature_in = int(feature_in / 2)  # 1593行reshape传的是P5_Upsample，是p5做了1*1和Upsample之后的结果扩大了2倍
    feature_size = shape[0] * channel_in * feature_in * feature_in  # shape[0]是1
    # 计算写地址
    computer_write_address = computer_address + add_write_address
    # 计算写数量
    write_size = 0
    if (operator == "maxpool"):
        write_size = int(feature_size / 4)  # 长宽缩小一半
    elif (operator == "upsample"):
        write_size = int(feature_size * 4)  # 一个写四遍
    print(shape)
    print(write_size)
    ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                   'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                   'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                   'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                   'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                   'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                   'Image_Reg1': '0C'}
    # ----------------reshape权重指令-------------------
    # 6个全都是默认的
    with open(filename, 'a+') as fp:
        # -------------reshape的计算指令--------------
        # reshape的reg4,reg5,reg6都全是0
        #    reg7:11位   split,maxpool,upsample 没用到,10位输入图片通道, 11 位，输入的图片 宽(高)
        channel_in = '{:010b}'.format(channel_in)
        feature_in = '{:011b}'.format(feature_in)
        reg7 = '00000000000' + str(channel_in) + str(feature_in)
        reg7 = str(int(reg7, 2))
        # 第一个指令:读第一个reshape地址
        # print('%08X' % computer_address)
        fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
        fp.write('%08X' % computer_address)
        print('================reshape================')
        print("computer_address=", '%08X' % computer_address)
        fp.write('\n')
        # 第二个指令:读第一个reshape数量
        fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
        fp.write('%08X' % feature_size)
        fp.write('\n')
        # 第三个指令:写地址
        fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
        fp.write('%08X' % computer_write_address)
        print("computer_write_address", '%08X' % computer_write_address)
        print('======================================')
        fp.write('\n')
        # 第六个指令:写数量
        fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
        fp.write('%08X' % (write_size))
        fp.write('\n')
        # 第七个指令:reg4
        fp.write('100000' + ins_address['TJPU_Reg4'])
        fp.write('00000000')
        fp.write('\n')
        # 第八个指令:reg5
        fp.write('100000' + ins_address['TJPU_Reg5'])
        fp.write('00000000')
        fp.write('\n')
        # 第九个指令:reg6
        fp.write('100000' + ins_address['TJPU_Reg6'])
        fp.write('00000000')
        fp.write('\n')
        # 第十个指令:reg7
        fp.write('100000' + ins_address['TJPU_Reg7'])
        fp.write('%08X' % int(reg7))
        fp.write('\n')
        # 第十三个指令:switch
        fp.write('100000' + ins_address['TJPU_Switch'])
        fp.write('00000008')
        fp.write('\n')
        # 第十四个指令:control
        if (operator == "maxpool"):
            fp.write('100000' + ins_address['TJPU_Control'])
            # 十六进制0004  ->  二进制0100  是maxpool的控制信号
            fp.write('00000400')
            fp.write('\n')
            fp.write('1100001400000F00')
            fp.write('\n')
        elif (operator == "upsample"):
            fp.write('100000' + ins_address['TJPU_Control'])
            ## 十六进制0008  ->  二进制1000  是upsample的控制信号
            fp.write('00000800')
            fp.write('\n')
            fp.write('1100001400000F00')
            fp.write('\n')

    return weight_address, computer_write_address


def reshape_block(weight_address, computer_address, feature, operator="", add_write_address=16777216, filename=' ',
                  block=2):
    default_data = 1300005000000000
    # 计算读的数量
    # 1 256 20 20
    shape = feature.shape  # b c  h w
    shape = shape[1] / block
    channel_in = shape[1]
    feature_in = shape[2]  # 图片大小
    # if operator == "upsample":
    #     feature_in = int(feature_in / 2)  # 1593行reshape传的是P5_Upsample，是p5做了1*1和Upsample之后的结果扩大了2倍
    feature_size = shape[0] * channel_in * feature_in * feature_in  # shape[0]是1
    # 计算写地址
    computer_write_address = computer_address + add_write_address
    # 计算写数量
    write_size = 0
    if (operator == "maxpool"):
        write_size = int(feature_size / 4)  # 长宽缩小一半
    elif (operator == "upsample"):
        write_size = int(feature_size * 4)  # 一个写四遍
    ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                   'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                   'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                   'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                   'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                   'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                   'Image_Reg1': '0C'}
    # ----------------reshape权重指令-------------------
    # 6个全都是默认的
    for index in range(block):
        with open(filename, 'a+') as fp:
            # -------------reshape的计算指令--------------
            # reshape的reg4,reg5,reg6都全是0
            #    reg7:11位   split,maxpool,upsample 没用到,10位输入图片通道, 11 位，输入的图片 宽(高)
            channel_in = '{:010b}'.format(channel_in)
            feature_in = '{:011b}'.format(feature_in)
            reg7 = '00000000000' + str(channel_in) + str(feature_in)
            reg7 = str(int(reg7, 2))
            # 第一个指令:读第一个reshape地址
            # print('%08X' % computer_address)
            fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
            fp.write('%08X' % computer_address)
            print('================reshape================')
            print("computer_address=", '%08X' % computer_address)
            fp.write('\n')
            # 第二个指令:读第一个reshape数量
            fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
            fp.write('%08X' % feature_size)
            fp.write('\n')
            # 第三个指令:写地址
            fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
            # fp.write('%08X' % computer_write_address)
            fp.write('%08x' % int(computer_write_address + write_size * index + add_write_address))
            print("computer_write_address", '%08X' % computer_write_address)
            print('======================================')
            fp.write('\n')
            # 第六个指令:写数量
            fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
            fp.write('%08X' % (write_size))
            fp.write('\n')
            # 第七个指令:reg4
            fp.write('100000' + ins_address['TJPU_Reg4'])
            fp.write('00000000')
            fp.write('\n')
            # 第八个指令:reg5
            fp.write('100000' + ins_address['TJPU_Reg5'])
            fp.write('00000000')
            fp.write('\n')
            # 第九个指令:reg6
            fp.write('100000' + ins_address['TJPU_Reg6'])
            fp.write('00000000')
            fp.write('\n')
            # 第十个指令:reg7
            fp.write('100000' + ins_address['TJPU_Reg7'])
            fp.write('%08X' % int(reg7))
            fp.write('\n')
            # 第十三个指令:switch
            fp.write('100000' + ins_address['TJPU_Switch'])
            fp.write('00000008')
            fp.write('\n')
            # 第十四个指令:control
            if (operator == "maxpool"):
                fp.write('100000' + ins_address['TJPU_Control'])
                # 十六进制0004  ->  二进制0100  是maxpool的控制信号
                fp.write('00000400')
                fp.write('\n')
                fp.write('1100001400000F00')
                fp.write('\n')
            elif (operator == "upsample"):
                fp.write('100000' + ins_address['TJPU_Control'])
                ## 十六进制0008  ->  二进制1000  是upsample的控制信号
                fp.write('00000800')
                fp.write('\n')
                fp.write('1100001400000F00')
                fp.write('\n')

    return weight_address, computer_write_address


def tensorr(x):
    tensor_py = torch.from_numpy(np.load(x))
    # print("tensor_py.shape======",tensor_py.shape)
    return tensor_py


class QuantizableYolo_tiny(nn.Module):

    def __init__(self, img_size=416):
        super(QuantizableYolo_tiny, self).__init__()
        ###############################
        # print(self)  QuantizableYolo_tiny()
        # exit()
        # Focus之后作为输入
        s1_stem_conv = tensorr('../para_682/backbone.backbone.stem.csp0.scale.npy')
        z1_stem_conv = tensorr('../para_682/backbone.backbone.stem.csp0.zero_point.npy')
        # stem_conv 权重
        s2_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.weight.scale.npy')
        z2_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.weight.zero_point.npy')
        # stem的输出
        s3_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.scale.npy')
        z3_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.zero_point.npy')
        bias_f_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.bias.npy')
        self.bias_int_stem_conv = bias_f_stem_conv
        coe_name = '../data1_coe/out_hand_stem_conv_leak.coe'
        self.stem_conv = Conv2d_Q(quant_scale1=s1_stem_conv, quant_zero_point1=z1_stem_conv, quant_scale2=s2_stem_conv,
                                  quant_zero_point2=z2_stem_conv, quant_scale3=s3_stem_conv,
                                  quant_zero_point3=z3_stem_conv, coe_name=coe_name)
        # *==========================dark2=====================================
        s1_dark2_0_conv = tensorr("../para_682/backbone.backbone.stem.conv.conv.scale.npy")
        z1_dark2_0_conv = tensorr("../para_682/backbone.backbone.stem.conv.conv.zero_point.npy")
        s2_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.weight.scale.npy")
        z2_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.weight.zero_point.npy")
        s3_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.scale.npy")
        z3_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.zero_point.npy")
        bias_f_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.bias.npy")
        self.bias_int_dark2_0_conv = bias_f_dark2_0_conv
        coe_name = '../data1_coe/out_hand_dark2_0_conv_leak.coe'
        self.dark2_0_conv = Conv2d_Q(quant_scale1=s1_dark2_0_conv, quant_zero_point1=z1_dark2_0_conv,
                                     quant_scale2=s2_dark2_0_conv,
                                     quant_zero_point2=z2_dark2_0_conv, quant_scale3=s3_dark2_0_conv,
                                     quant_zero_point3=z3_dark2_0_conv, coe_name=coe_name)

        s1_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.0.conv.scale.npy")
        z1_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.0.conv.zero_point.npy")
        s2_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.weight.scale.npy")
        z2_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.weight.zero_point.npy")
        s3_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.scale.npy")
        z3_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.zero_point.npy")
        bias_f_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.bias.npy")
        self.bias_int_dark2_1_conv1 = bias_f_dark2_1_conv1
        coe_name = '../data1_coe/out_hand_dark2_1_conv1_leak.coe'
        self.dark2_1_conv1 = Conv2d_Q(quant_scale1=s1_dark2_1_conv1, quant_zero_point1=z1_dark2_1_conv1,
                                      quant_scale2=s2_dark2_1_conv1,
                                      quant_zero_point2=z2_dark2_1_conv1, quant_scale3=s3_dark2_1_conv1,
                                      quant_zero_point3=z3_dark2_1_conv1, coe_name=coe_name)

        s1_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.0.conv.scale.npy")
        z1_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.0.conv.zero_point.npy")
        s2_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.weight.scale.npy")
        z2_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.weight.zero_point.npy")
        s3_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.scale.npy")
        z3_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.zero_point.npy")
        bias_f_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.bias.npy")
        self.bias_int_dark2_1_conv2 = bias_f_dark2_1_conv2
        coe_name = '../data1_coe/out_hand_dark2_1_conv2_leak.coe'
        self.dark2_1_conv2 = Conv2d_Q(quant_scale1=s1_dark2_1_conv2, quant_zero_point1=z1_dark2_1_conv2,
                                      quant_scale2=s2_dark2_1_conv2,
                                      quant_zero_point2=z2_dark2_1_conv2, quant_scale3=s3_dark2_1_conv2,
                                      quant_zero_point3=z3_dark2_1_conv2, coe_name=coe_name)

        s1_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.scale.npy")
        z1_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.zero_point.npy")
        s2_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.weight.scale.npy")
        z2_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.weight.zero_point.npy")
        s3_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.scale.npy")
        z3_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.zero_point.npy")
        bias_f_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.bias.npy")
        self.bias_int_dark2_1_m0_conv1 = bias_f_dark2_1_m0_conv1
        coe_name = '../data1_coe/out_hand_dark2_1_m0_conv1_leak.coe'
        self.dark2_1_m0_conv1 = Conv2d_Q(quant_scale1=s1_dark2_1_m0_conv1, quant_zero_point1=z1_dark2_1_m0_conv1,
                                         quant_scale2=s2_dark2_1_m0_conv1,
                                         quant_zero_point2=z2_dark2_1_m0_conv1, quant_scale3=s3_dark2_1_m0_conv1,
                                         quant_zero_point3=z3_dark2_1_m0_conv1, coe_name=coe_name)

        s1_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.scale.npy")
        z1_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.zero_point.npy")
        s2_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.weight.scale.npy")
        z2_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.weight.zero_point.npy")
        s3_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.scale.npy")
        z3_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.zero_point.npy")
        bias_f_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.bias.npy")
        self.bias_int_dark2_1_m0_conv2 = bias_f_dark2_1_m0_conv2
        coe_name = '../data1_coe/out_hand_dark2_1_m0_conv2_leak.coe'
        self.dark2_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark2_1_m0_conv2, quant_zero_point1=z1_dark2_1_m0_conv2,
                                         quant_scale2=s2_dark2_1_m0_conv2,
                                         quant_zero_point2=z2_dark2_1_m0_conv2, quant_scale3=s3_dark2_1_m0_conv2,
                                         quant_zero_point3=z3_dark2_1_m0_conv2, coe_name=coe_name)

        s1_dark2_1_m0_add = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.scale.npy")
        z1_dark2_1_m0_add = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.zero_point.npy")
        s2_dark2_1_m0_add = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.scale.npy")
        z2_dark2_1_m0_add = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.zero_point.npy")
        s3_dark2_1_m0_add = tensorr("../para_682/backbone.backbone.dark2.1.m.0.csp.scale.npy")
        z3_dark2_1_m0_add = tensorr("../para_682/backbone.backbone.dark2.1.m.0.csp.zero_point.npy")
        coe_name = '../data1_coe/out_hand_dark2_1_m0_csp_leak.coe'
        self.dark2_1_m0_add = Conv2d_Q(quant_scale1=s1_dark2_1_m0_add, quant_zero_point1=z1_dark2_1_m0_add,
                                       quant_scale2=s2_dark2_1_m0_add,
                                       quant_zero_point2=z2_dark2_1_m0_add, quant_scale3=s3_dark2_1_m0_add,
                                       quant_zero_point3=z3_dark2_1_m0_add, coe_name=coe_name)

        s1_dark2_1_cat = tensorr("../para_682/backbone.backbone.dark2.1.m.0.csp.scale.npy")
        z1_dark2_1_cat = tensorr("../para_682/backbone.backbone.dark2.1.m.0.csp.zero_point.npy")
        s2_dark2_1_cat = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.scale.npy")
        z2_dark2_1_cat = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.zero_point.npy")
        s3_dark2_1_cat = tensorr("../para_682/backbone.backbone.dark2.1.csp1.scale.npy")
        z3_dark2_1_cat = tensorr("../para_682/backbone.backbone.dark2.1.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_s1_dark2_1_csp1_leak.coe'
        self.dark2_1_cat = Conv2d_Q(quant_scale1=s1_dark2_1_cat, quant_zero_point1=z1_dark2_1_cat,
                                    quant_scale2=s2_dark2_1_cat,
                                    quant_zero_point2=z2_dark2_1_cat, quant_scale3=s3_dark2_1_cat,
                                    quant_zero_point3=z3_dark2_1_cat, coe_name=coe_name)

        s1_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.csp1.scale.npy")
        z1_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.csp1.zero_point.npy")
        s2_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.weight.scale.npy")
        z2_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.weight.zero_point.npy")
        s3_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.scale.npy")
        z3_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.zero_point.npy")
        bias_f_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.bias.npy")
        self.bias_int_dark2_1_conv3 = bias_f_dark2_1_conv3
        coe_name = '../data1_coe/out_hand_dark2_1_conv2_leak.coe'
        self.dark2_1_conv3 = Conv2d_Q(quant_scale1=s1_dark2_1_conv3, quant_zero_point1=z1_dark2_1_conv3,
                                      quant_scale2=s2_dark2_1_conv3,
                                      quant_zero_point2=z2_dark2_1_conv3, quant_scale3=s3_dark2_1_conv3,
                                      quant_zero_point3=z3_dark2_1_conv3, coe_name=coe_name)
        # *================================dark3===============================================
        s1_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.scale.npy")
        z1_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.zero_point.npy")
        s2_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.weight.scale.npy")
        z2_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.weight.zero_point.npy")
        s3_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.scale.npy")
        z3_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.zero_point.npy")
        bias_f_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.bias.npy")
        self.bias_int_dark3_0_conv = bias_f_dark3_0_conv
        coe_name = '../data1_coe/out_hand_dark3_0_conv_leak.coe'
        self.dark3_0_conv = Conv2d_Q(quant_scale1=s1_dark3_0_conv, quant_zero_point1=z1_dark3_0_conv,
                                     quant_scale2=s2_dark3_0_conv,
                                     quant_zero_point2=z2_dark3_0_conv, quant_scale3=s3_dark3_0_conv,
                                     quant_zero_point3=z3_dark3_0_conv, coe_name=coe_name)

        s1_dark3_1_conv1 = tensorr("../para_682/backbone.backbone.dark3.0.conv.scale.npy")
        z1_dark3_1_conv1 = tensorr("../para_682/backbone.backbone.dark3.0.conv.zero_point.npy")
        s2_dark3_1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.weight.scale.npy")
        z2_dark3_1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.weight.zero_point.npy")
        s3_dark3_1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.scale.npy")
        z3_dark3_1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.zero_point.npy")
        bias_f_dark3_1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.bias.npy")
        self.bias_int_dark3_1_conv1 = bias_f_dark3_1_conv1
        coe_name = '../data1_coe/out_hand_dark3_1_conv1_leak.coe'
        self.dark3_1_conv1 = Conv2d_Q(quant_scale1=s1_dark3_1_conv1, quant_zero_point1=z1_dark3_1_conv1,
                                      quant_scale2=s2_dark3_1_conv1,
                                      quant_zero_point2=z2_dark3_1_conv1, quant_scale3=s3_dark3_1_conv1,
                                      quant_zero_point3=z3_dark3_1_conv1, coe_name=coe_name)

        s1_dark3_1_conv2 = tensorr("../para_682/backbone.backbone.dark3.0.conv.scale.npy")
        z1_dark3_1_conv2 = tensorr("../para_682/backbone.backbone.dark3.0.conv.zero_point.npy")
        s2_dark3_1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.conv2.conv.weight.scale.npy")
        z2_dark3_1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.conv2.conv.weight.zero_point.npy")
        s3_dark3_1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.conv2.conv.scale.npy")
        z3_dark3_1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.conv2.conv.zero_point.npy")
        bias_f_dark3_1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.conv2.conv.bias.npy")
        self.bias_int_dark3_1_conv2 = bias_f_dark3_1_conv2
        coe_name = '../data1_coe/out_hand_dark3_1_conv2_leak.coe'
        self.dark3_1_conv2 = Conv2d_Q(quant_scale1=s1_dark3_1_conv2, quant_zero_point1=z1_dark3_1_conv2,
                                      quant_scale2=s2_dark3_1_conv2,
                                      quant_zero_point2=z2_dark3_1_conv2, quant_scale3=s3_dark3_1_conv2,
                                      quant_zero_point3=z3_dark3_1_conv2, coe_name=coe_name)

        s1_dark3_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.scale.npy")
        z1_dark3_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.zero_point.npy")
        s2_dark3_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.weight.scale.npy")
        z2_dark3_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.weight.zero_point.npy")
        s3_dark3_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.scale.npy")
        z3_dark3_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.zero_point.npy")
        bias_f_dark3_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.bias.npy")
        self.bias_int_dark3_1_m0_conv1 = bias_f_dark3_1_m0_conv1
        coe_name = '../data1_coe/out_hand_dark3_1_m0_conv1_leak.coe'
        self.dark3_1_m0_conv1 = Conv2d_Q(quant_scale1=s1_dark3_1_m0_conv1, quant_zero_point1=z1_dark3_1_m0_conv1,
                                         quant_scale2=s2_dark3_1_m0_conv1,
                                         quant_zero_point2=z2_dark3_1_m0_conv1, quant_scale3=s3_dark3_1_m0_conv1,
                                         quant_zero_point3=z3_dark3_1_m0_conv1, coe_name=coe_name)

        s1_dark3_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.scale.npy")
        z1_dark3_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.zero_point.npy")
        s2_dark3_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.weight.scale.npy")
        z2_dark3_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.weight.zero_point.npy")
        s3_dark3_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.scale.npy")
        z3_dark3_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.zero_point.npy")
        bias_f_dark3_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.bias.npy")
        self.bias_int_dark3_1_m0_conv2 = bias_f_dark3_1_m0_conv2
        coe_name = '../data1_coe/out_hand_dark3_1_m0_conv2_leak.coe'
        self.dark3_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark3_1_m0_conv2, quant_zero_point1=z1_dark3_1_m0_conv2,
                                         quant_scale2=s2_dark3_1_m0_conv2,
                                         quant_zero_point2=z2_dark3_1_m0_conv2, quant_scale3=s3_dark3_1_m0_conv2,
                                         quant_zero_point3=z3_dark3_1_m0_conv2, coe_name=coe_name)

        s1_dark3_1_m0_add = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.scale.npy")
        z1_dark3_1_m0_add = tensorr("../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.zero_point.npy")
        s2_dark3_1_m0_add = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.scale.npy")
        z2_dark3_1_m0_add = tensorr("../para_682/backbone.backbone.dark3.1.conv1.conv.zero_point.npy")
        s3_dark3_1_m0_add = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.scale.npy")
        z3_dark3_1_m0_add = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.zero_point.npy")
        coe_name = '../data1_coe/out_hand_dark3_1_m0_csp_leak.coe'
        self.dark3_1_m0_add = Conv2d_Q(quant_scale1=s1_dark3_1_m0_add, quant_zero_point1=z1_dark3_1_m0_add,
                                       quant_scale2=s2_dark3_1_m0_add,
                                       quant_zero_point2=z2_dark3_1_m0_add, quant_scale3=s3_dark3_1_m0_add,
                                       quant_zero_point3=z3_dark3_1_m0_add, coe_name=coe_name)

        s1_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.scale.npy")
        z1_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.zero_point.npy")
        s2_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.weight.scale.npy")
        z2_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.weight.zero_point.npy")
        s3_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.scale.npy")
        z3_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.zero_point.npy")
        bias_f_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.bias.npy")
        self.bias_int_dark3_1_m1_conv1 = bias_f_dark3_1_m1_conv1
        coe_name = '../data1_coe/out_hand_dark3_1_m1_conv1_leak.coe'
        self.dark3_1_m1_conv1 = Conv2d_Q(quant_scale1=s1_dark3_1_m1_conv1, quant_zero_point1=z1_dark3_1_m1_conv1,
                                         quant_scale2=s2_dark3_1_m1_conv1,
                                         quant_zero_point2=z2_dark3_1_m1_conv1, quant_scale3=s3_dark3_1_m1_conv1,
                                         quant_zero_point3=z3_dark3_1_m1_conv1, coe_name=coe_name)

        s1_dark3_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.scale.npy")
        z1_dark3_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.zero_point.npy")
        s2_dark3_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.weight.scale.npy")
        z2_dark3_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.weight.zero_point.npy")
        s3_dark3_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.scale.npy")
        z3_dark3_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.zero_point.npy")
        bias_f_dark3_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.bias.npy")
        self.bias_int_dark3_1_m1_conv2 = bias_f_dark3_1_m1_conv2
        coe_name = '../data1_coe/out_hand_dark3_1_m1_conv2_leak.coe'
        self.dark3_1_m1_conv2 = Conv2d_Q(quant_scale1=s1_dark3_1_m1_conv2, quant_zero_point1=z1_dark3_1_m1_conv2,
                                         quant_scale2=s2_dark3_1_m1_conv2,
                                         quant_zero_point2=z2_dark3_1_m1_conv2, quant_scale3=s3_dark3_1_m1_conv2,
                                         quant_zero_point3=z3_dark3_1_m1_conv2, coe_name=coe_name)

        s1_dark3_1_m1_add = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.scale.npy")
        z1_dark3_1_m1_add = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.zero_point.npy")
        s2_dark3_1_m1_add = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.scale.npy")
        z2_dark3_1_m1_add = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.zero_point.npy")
        s3_dark3_1_m1_add = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.scale.npy")
        z3_dark3_1_m1_add = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.zero_point.npy")
        coe_name = '../data1_coe/out_hand_dark3_1_m1_csp_leak.coe'
        self.dark3_1_m1_add = Conv2d_Q(quant_scale1=s1_dark3_1_m1_add, quant_zero_point1=z1_dark3_1_m1_add,
                                       quant_scale2=s2_dark3_1_m1_add,
                                       quant_zero_point2=z2_dark3_1_m1_add, quant_scale3=s3_dark3_1_m1_add,
                                       quant_zero_point3=z3_dark3_1_m1_add, coe_name=coe_name)

        s1_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.scale.npy")
        z1_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.zero_point.npy")
        s2_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.weight.scale.npy")
        z2_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.weight.zero_point.npy")
        s3_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.scale.npy")
        z3_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.zero_point.npy")
        bias_f_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.bias.npy")
        self.bias_int_dark3_1_m2_conv1 = bias_f_dark3_1_m2_conv1
        coe_name = '../data1_coe/out_hand_dark3_1_m2_conv1_leak.coe'
        self.dark3_1_m2_conv1 = Conv2d_Q(quant_scale1=s1_dark3_1_m2_conv1, quant_zero_point1=z1_dark3_1_m2_conv1,
                                         quant_scale2=s2_dark3_1_m2_conv1,
                                         quant_zero_point2=z2_dark3_1_m2_conv1, quant_scale3=s3_dark3_1_m2_conv1,
                                         quant_zero_point3=z3_dark3_1_m2_conv1, coe_name=coe_name)

        s1_dark3_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.scale.npy")
        z1_dark3_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.zero_point.npy")
        s2_dark3_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.weight.scale.npy")
        z2_dark3_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.weight.zero_point.npy")
        s3_dark3_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.scale.npy")
        z3_dark3_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.zero_point.npy")
        bias_f_dark3_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.bias.npy")
        self.bias_int_dark3_1_m2_conv2 = bias_f_dark3_1_m2_conv2
        coe_name = '../data1_coe/out_hand_dark3_1_m2_conv2_leak.coe'
        self.dark3_1_m2_conv2 = Conv2d_Q(quant_scale1=s1_dark3_1_m2_conv2, quant_zero_point1=z1_dark3_1_m2_conv2,
                                         quant_scale2=s2_dark3_1_m2_conv2,
                                         quant_zero_point2=z2_dark3_1_m2_conv2, quant_scale3=s3_dark3_1_m2_conv2,
                                         quant_zero_point3=z3_dark3_1_m2_conv2, coe_name=coe_name)

        s1_dark3_1_m2_add = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.scale.npy")
        z1_dark3_1_m2_add = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.zero_point.npy")
        s2_dark3_1_m2_add = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.scale.npy")
        z2_dark3_1_m2_add = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.zero_point.npy")
        s3_dark3_1_m2_add = tensorr("../para_682/backbone.backbone.dark3.1.m.2.csp.scale.npy")
        z3_dark3_1_m2_add = tensorr("../para_682/backbone.backbone.dark3.1.m.2.csp.zero_point.npy")
        coe_name = '../data1_coe/out_hand_dark3_1_m2_csp_leak.coe'
        self.dark3_1_m2_add = Conv2d_Q(quant_scale1=s1_dark3_1_m2_add, quant_zero_point1=z1_dark3_1_m2_add,
                                       quant_scale2=s2_dark3_1_m2_add,
                                       quant_zero_point2=z2_dark3_1_m2_add, quant_scale3=s3_dark3_1_m2_add,
                                       quant_zero_point3=z3_dark3_1_m2_add, coe_name=coe_name)

        s1_dark3_1_cat = tensorr("../para_682/backbone.backbone.dark3.1.m.2.csp.scale.npy")
        z1_dark3_1_cat = tensorr("../para_682/backbone.backbone.dark3.1.m.2.csp.zero_point.npy")
        s2_dark3_1_cat = tensorr("../para_682/backbone.backbone.dark3.1.conv2.conv.scale.npy")
        z2_dark3_1_cat = tensorr("../para_682/backbone.backbone.dark3.1.conv2.conv.zero_point.npy")
        s3_dark3_1_cat = tensorr("../para_682/backbone.backbone.dark3.1.csp1.scale.npy")
        z3_dark3_1_cat = tensorr("../para_682/backbone.backbone.dark3.1.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_s1_dark3_1_csp1_leak.coe'
        self.dark3_1_cat = Conv2d_Q(quant_scale1=s1_dark3_1_cat, quant_zero_point1=z1_dark3_1_cat,
                                    quant_scale2=s2_dark3_1_cat,
                                    quant_zero_point2=z2_dark3_1_cat, quant_scale3=s3_dark3_1_cat,
                                    quant_zero_point3=z3_dark3_1_cat, coe_name=coe_name)

        s1_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.csp1.scale.npy")
        z1_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.csp1.zero_point.npy")
        s2_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.weight.scale.npy")
        z2_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.weight.zero_point.npy")
        s3_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.scale.npy")
        z3_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.zero_point.npy")
        bias_f_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.bias.npy")
        self.bias_int_dark3_1_conv3 = bias_f_dark3_1_conv3
        coe_name = '../data1_coe/out_hand_dark3_1_conv2_leak.coe'
        self.dark3_1_conv3 = Conv2d_Q(quant_scale1=s1_dark3_1_conv3, quant_zero_point1=z1_dark3_1_conv3,
                                      quant_scale2=s2_dark3_1_conv3,
                                      quant_zero_point2=z2_dark3_1_conv3, quant_scale3=s3_dark3_1_conv3,
                                      quant_zero_point3=z3_dark3_1_conv3, coe_name=coe_name)
        # *=====================================dark4===================================================
        s1_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.scale.npy")
        z1_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.zero_point.npy")
        s2_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.weight.scale.npy")
        z2_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.weight.zero_point.npy")
        s3_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.scale.npy")
        z3_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.zero_point.npy")
        bias_f_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.bias.npy")
        self.bias_int_dark4_0_conv = bias_f_dark4_0_conv
        coe_name = '../data1_coe/out_hand_dark4_0_conv_leak.coe'
        self.dark4_0_conv = Conv2d_Q(quant_scale1=s1_dark4_0_conv, quant_zero_point1=z1_dark4_0_conv,
                                     quant_scale2=s2_dark4_0_conv,
                                     quant_zero_point2=z2_dark4_0_conv, quant_scale3=s3_dark4_0_conv,
                                     quant_zero_point3=z3_dark4_0_conv, coe_name=coe_name)

        s1_dark4_1_conv1 = tensorr("../para_682/backbone.backbone.dark4.0.conv.scale.npy")
        z1_dark4_1_conv1 = tensorr("../para_682/backbone.backbone.dark4.0.conv.zero_point.npy")
        s2_dark4_1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.weight.scale.npy")
        z2_dark4_1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.weight.zero_point.npy")
        s3_dark4_1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.scale.npy")
        z3_dark4_1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.zero_point.npy")
        bias_f_dark4_1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.bias.npy")
        self.bias_int_dark4_1_conv1 = bias_f_dark4_1_conv1
        coe_name = '../data1_coe/out_hand_dark4_1_conv1_leak.coe'
        self.dark4_1_conv1 = Conv2d_Q(quant_scale1=s1_dark4_1_conv1, quant_zero_point1=z1_dark4_1_conv1,
                                      quant_scale2=s2_dark4_1_conv1,
                                      quant_zero_point2=z2_dark4_1_conv1, quant_scale3=s3_dark4_1_conv1,
                                      quant_zero_point3=z3_dark4_1_conv1, coe_name=coe_name)

        s1_dark4_1_conv2 = tensorr("../para_682/backbone.backbone.dark4.0.conv.scale.npy")
        z1_dark4_1_conv2 = tensorr("../para_682/backbone.backbone.dark4.0.conv.zero_point.npy")
        s2_dark4_1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.conv2.conv.weight.scale.npy")
        z2_dark4_1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.conv2.conv.weight.zero_point.npy")
        s3_dark4_1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.conv2.conv.scale.npy")
        z3_dark4_1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.conv2.conv.zero_point.npy")
        bias_f_dark4_1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.conv2.conv.bias.npy")
        self.bias_int_dark4_1_conv2 = bias_f_dark4_1_conv2
        coe_name = '../data1_coe/out_hand_dark4_1_conv2_leak.coe'
        self.dark4_1_conv2 = Conv2d_Q(quant_scale1=s1_dark4_1_conv2, quant_zero_point1=z1_dark4_1_conv2,
                                      quant_scale2=s2_dark4_1_conv2,
                                      quant_zero_point2=z2_dark4_1_conv2, quant_scale3=s3_dark4_1_conv2,
                                      quant_zero_point3=z3_dark4_1_conv2, coe_name=coe_name)

        s1_dark4_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.scale.npy")
        z1_dark4_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.zero_point.npy")
        s2_dark4_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.weight.scale.npy")
        z2_dark4_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.weight.zero_point.npy")
        s3_dark4_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.scale.npy")
        z3_dark4_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.zero_point.npy")
        bias_f_dark4_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.bias.npy")
        self.bias_int_dark4_1_m0_conv1 = bias_f_dark4_1_m0_conv1
        coe_name = '../data1_coe/out_hand_dark4_1_m0_conv1_leak.coe'
        self.dark4_1_m0_conv1 = Conv2d_Q(quant_scale1=s1_dark4_1_m0_conv1, quant_zero_point1=z1_dark4_1_m0_conv1,
                                         quant_scale2=s2_dark4_1_m0_conv1,
                                         quant_zero_point2=z2_dark4_1_m0_conv1, quant_scale3=s3_dark4_1_m0_conv1,
                                         quant_zero_point3=z3_dark4_1_m0_conv1, coe_name=coe_name)

        s1_dark4_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.scale.npy")
        z1_dark4_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.zero_point.npy")
        s2_dark4_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.weight.scale.npy")
        z2_dark4_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.weight.zero_point.npy")
        s3_dark4_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.scale.npy")
        z3_dark4_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.zero_point.npy")
        bias_f_dark4_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.bias.npy")
        self.bias_int_dark4_1_m0_conv2 = bias_f_dark4_1_m0_conv2
        coe_name = '../data1_coe/out_hand_dark4_1_m0_conv2_leak.coe'
        self.dark4_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark4_1_m0_conv2, quant_zero_point1=z1_dark4_1_m0_conv2,
                                         quant_scale2=s2_dark4_1_m0_conv2,
                                         quant_zero_point2=z2_dark4_1_m0_conv2, quant_scale3=s3_dark4_1_m0_conv2,
                                         quant_zero_point3=z3_dark4_1_m0_conv2, coe_name=coe_name)

        s1_dark4_1_m0_add = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.scale.npy")
        z1_dark4_1_m0_add = tensorr("../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.zero_point.npy")
        s2_dark4_1_m0_add = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.scale.npy")
        z2_dark4_1_m0_add = tensorr("../para_682/backbone.backbone.dark4.1.conv1.conv.zero_point.npy")
        s3_dark4_1_m0_add = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.scale.npy")
        z3_dark4_1_m0_add = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.zero_point.npy")
        coe_name = '../data1_coe/out_hand_dark4_1_m0_csp_leak.coe'
        self.dark4_1_m0_add = Conv2d_Q(quant_scale1=s1_dark4_1_m0_add, quant_zero_point1=z1_dark4_1_m0_add,
                                       quant_scale2=s2_dark4_1_m0_add,
                                       quant_zero_point2=z2_dark4_1_m0_add, quant_scale3=s3_dark4_1_m0_add,
                                       quant_zero_point3=z3_dark4_1_m0_add, coe_name=coe_name)

        s1_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.scale.npy")
        z1_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.zero_point.npy")
        s2_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.weight.scale.npy")
        z2_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.weight.zero_point.npy")
        s3_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.scale.npy")
        z3_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.zero_point.npy")
        bias_f_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.bias.npy")
        self.bias_int_dark4_1_m1_conv1 = bias_f_dark4_1_m1_conv1
        coe_name = '../data1_coe/out_hand_dark4_1_m1_conv1_leak.coe'
        self.dark4_1_m1_conv1 = Conv2d_Q(quant_scale1=s1_dark4_1_m1_conv1, quant_zero_point1=z1_dark4_1_m1_conv1,
                                         quant_scale2=s2_dark4_1_m1_conv1,
                                         quant_zero_point2=z2_dark4_1_m1_conv1, quant_scale3=s3_dark4_1_m1_conv1,
                                         quant_zero_point3=z3_dark4_1_m1_conv1, coe_name=coe_name)

        s1_dark4_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.scale.npy")
        z1_dark4_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.zero_point.npy")
        s2_dark4_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.weight.scale.npy")
        z2_dark4_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.weight.zero_point.npy")
        s3_dark4_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.scale.npy")
        z3_dark4_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.zero_point.npy")
        bias_f_dark4_1_m1_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.bias.npy")
        self.bias_int_dark4_1_m1_conv2 = bias_f_dark4_1_m1_conv2
        coe_name = '../data1_coe/out_hand_dark4_1_m1_conv2_leak.coe'
        self.dark4_1_m1_conv2 = Conv2d_Q(quant_scale1=s1_dark4_1_m1_conv2, quant_zero_point1=z1_dark4_1_m1_conv2,
                                         quant_scale2=s2_dark4_1_m1_conv2,
                                         quant_zero_point2=z2_dark4_1_m1_conv2, quant_scale3=s3_dark4_1_m1_conv2,
                                         quant_zero_point3=z3_dark4_1_m1_conv2, coe_name=coe_name)

        s1_dark4_1_m1_add = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.scale.npy")
        z1_dark4_1_m1_add = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.zero_point.npy")
        s2_dark4_1_m1_add = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.scale.npy")
        z2_dark4_1_m1_add = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.zero_point.npy")
        s3_dark4_1_m1_add = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.scale.npy")
        z3_dark4_1_m1_add = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.zero_point.npy")
        coe_name = '../data1_coe/out_hand_dark4_1_m1_csp_leak.coe'
        self.dark4_1_m1_add = Conv2d_Q(quant_scale1=s1_dark4_1_m1_add, quant_zero_point1=z1_dark4_1_m1_add,
                                       quant_scale2=s2_dark4_1_m1_add,
                                       quant_zero_point2=z2_dark4_1_m1_add, quant_scale3=s3_dark4_1_m1_add,
                                       quant_zero_point3=z3_dark4_1_m1_add, coe_name=coe_name)

        s1_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.scale.npy")
        z1_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.zero_point.npy")
        s2_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.weight.scale.npy")
        z2_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.weight.zero_point.npy")
        s3_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.scale.npy")
        z3_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.zero_point.npy")
        bias_f_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.bias.npy")
        self.bias_int_dark4_1_m2_conv1 = bias_f_dark4_1_m2_conv1
        coe_name = '../data1_coe/out_hand_dark4_1_m2_conv1_leak.coe'
        self.dark4_1_m2_conv1 = Conv2d_Q(quant_scale1=s1_dark4_1_m2_conv1, quant_zero_point1=z1_dark4_1_m2_conv1,
                                         quant_scale2=s2_dark4_1_m2_conv1,
                                         quant_zero_point2=z2_dark4_1_m2_conv1, quant_scale3=s3_dark4_1_m2_conv1,
                                         quant_zero_point3=z3_dark4_1_m2_conv1, coe_name=coe_name)

        s1_dark4_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.scale.npy")
        z1_dark4_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.zero_point.npy")
        s2_dark4_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.weight.scale.npy")
        z2_dark4_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.weight.zero_point.npy")
        s3_dark4_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.scale.npy")
        z3_dark4_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.zero_point.npy")
        bias_f_dark4_1_m2_conv2 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.bias.npy")
        self.bias_int_dark4_1_m2_conv2 = bias_f_dark4_1_m2_conv2
        coe_name = '../data1_coe/out_hand_dark4_1_m2_conv2_leak.coe'
        self.dark4_1_m2_conv2 = Conv2d_Q(quant_scale1=s1_dark4_1_m2_conv2, quant_zero_point1=z1_dark4_1_m2_conv2,
                                         quant_scale2=s2_dark4_1_m2_conv2,
                                         quant_zero_point2=z2_dark4_1_m2_conv2, quant_scale3=s3_dark4_1_m2_conv2,
                                         quant_zero_point3=z3_dark4_1_m2_conv2, coe_name=coe_name)

        s1_dark4_1_m2_add = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.scale.npy")
        z1_dark4_1_m2_add = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.zero_point.npy")
        s2_dark4_1_m2_add = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.scale.npy")
        z2_dark4_1_m2_add = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.zero_point.npy")
        s3_dark4_1_m2_add = tensorr("../para_682/backbone.backbone.dark4.1.m.2.csp.scale.npy")
        z3_dark4_1_m2_add = tensorr("../para_682/backbone.backbone.dark4.1.m.2.csp.zero_point.npy")
        coe_name = '../data1_coe/out_hand_dark4_1_m2_csp_leak.coe'
        self.dark4_1_m2_add = Conv2d_Q(quant_scale1=s1_dark4_1_m2_add, quant_zero_point1=z1_dark4_1_m2_add,
                                       quant_scale2=s2_dark4_1_m2_add,
                                       quant_zero_point2=z2_dark4_1_m2_add, quant_scale3=s3_dark4_1_m2_add,
                                       quant_zero_point3=z3_dark4_1_m2_add, coe_name=coe_name)

        s1_dark4_1_cat = tensorr("../para_682/backbone.backbone.dark4.1.m.2.csp.scale.npy")
        z1_dark4_1_cat = tensorr("../para_682/backbone.backbone.dark4.1.m.2.csp.zero_point.npy")
        s2_dark4_1_cat = tensorr("../para_682/backbone.backbone.dark4.1.conv2.conv.scale.npy")
        z2_dark4_1_cat = tensorr("../para_682/backbone.backbone.dark4.1.conv2.conv.zero_point.npy")
        s3_dark4_1_cat = tensorr("../para_682/backbone.backbone.dark4.1.csp1.scale.npy")
        z3_dark4_1_cat = tensorr("../para_682/backbone.backbone.dark4.1.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_s1_dark4_1_csp1_leak.coe'
        self.dark4_1_cat = Conv2d_Q(quant_scale1=s1_dark4_1_cat, quant_zero_point1=z1_dark4_1_cat,
                                    quant_scale2=s2_dark4_1_cat,
                                    quant_zero_point2=z2_dark4_1_cat, quant_scale3=s3_dark4_1_cat,
                                    quant_zero_point3=z3_dark4_1_cat, coe_name=coe_name)

        s1_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.csp1.scale.npy")
        z1_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.csp1.zero_point.npy")
        s2_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.weight.scale.npy")
        z2_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.weight.zero_point.npy")
        s3_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.scale.npy")
        z3_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.zero_point.npy")
        bias_f_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.bias.npy")
        self.bias_int_dark4_1_conv3 = bias_f_dark4_1_conv3
        coe_name = '../data1_coe/out_hand_dark4_1_conv2_leak.coe'
        self.dark4_1_conv3 = Conv2d_Q(quant_scale1=s1_dark4_1_conv3, quant_zero_point1=z1_dark4_1_conv3,
                                      quant_scale2=s2_dark4_1_conv3,
                                      quant_zero_point2=z2_dark4_1_conv3, quant_scale3=s3_dark4_1_conv3,
                                      quant_zero_point3=z3_dark4_1_conv3, coe_name=coe_name)
        # *=====================================dark5====================================================
        s1_dark5_0_conv = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.scale.npy")
        z1_dark5_0_conv = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.zero_point.npy")
        s2_dark5_0_conv = tensorr("../para_682/backbone.backbone.dark5.0.conv.weight.scale.npy")
        z2_dark5_0_conv = tensorr("../para_682/backbone.backbone.dark5.0.conv.weight.zero_point.npy")
        s3_dark5_0_conv = tensorr("../para_682/backbone.backbone.dark5.0.conv.scale.npy")
        z3_dark5_0_conv = tensorr("../para_682/backbone.backbone.dark5.0.conv.zero_point.npy")
        bias_f_dark5_0_conv = tensorr("../para_682/backbone.backbone.dark5.0.conv.bias.npy")
        self.bias_int_dark5_0_conv = bias_f_dark5_0_conv
        coe_name = '../data1_coe/out_hand_dark5_0_conv_leak.coe'
        self.dark5_0_conv = Conv2d_Q(quant_scale1=s1_dark5_0_conv, quant_zero_point1=z1_dark5_0_conv,
                                     quant_scale2=s2_dark5_0_conv,
                                     quant_zero_point2=z2_dark5_0_conv, quant_scale3=s3_dark5_0_conv,
                                     quant_zero_point3=z3_dark5_0_conv, coe_name=coe_name)

        s1_dark5_1_conv1 = tensorr("../para_682/backbone.backbone.dark5.0.conv.scale.npy")
        z1_dark5_1_conv1 = tensorr("../para_682/backbone.backbone.dark5.0.conv.zero_point.npy")
        s2_dark5_1_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.weight.scale.npy")
        z2_dark5_1_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.weight.zero_point.npy")
        s3_dark5_1_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.scale.npy")
        z3_dark5_1_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.zero_point.npy")
        bias_f_dark5_1_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.bias.npy")
        self.bias_int_dark5_1_conv1 = bias_f_dark5_1_conv1
        coe_name = '../data1_coe/out_hand_dark5_1_conv1_leak.coe'
        self.dark5_1_conv1 = Conv2d_Q(quant_scale1=s1_dark5_1_conv1, quant_zero_point1=z1_dark5_1_conv1,
                                      quant_scale2=s2_dark5_1_conv1,
                                      quant_zero_point2=z2_dark5_1_conv1, quant_scale3=s3_dark5_1_conv1,
                                      quant_zero_point3=z3_dark5_1_conv1, coe_name=coe_name)

        s1_dark5_1_conv2 = tensorr("../para_682/backbone.backbone.dark5.0.conv.scale.npy")
        z1_dark5_1_conv2 = tensorr("../para_682/backbone.backbone.dark5.0.conv.zero_point.npy")
        s2_dark5_1_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.conv2.conv.weight.scale.npy")
        z2_dark5_1_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.conv2.conv.weight.zero_point.npy")
        s3_dark5_1_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.conv2.conv.scale.npy")
        z3_dark5_1_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.conv2.conv.zero_point.npy")
        bias_f_dark5_1_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.conv2.conv.bias.npy")
        self.bias_int_dark5_1_conv2 = bias_f_dark5_1_conv2
        coe_name = '../data1_coe/out_hand_dark5_1_conv2_leak.coe'
        self.dark5_1_conv2 = Conv2d_Q(quant_scale1=s1_dark5_1_conv2, quant_zero_point1=z1_dark5_1_conv2,
                                      quant_scale2=s2_dark5_1_conv2,
                                      quant_zero_point2=z2_dark5_1_conv2, quant_scale3=s3_dark5_1_conv2,
                                      quant_zero_point3=z3_dark5_1_conv2, coe_name=coe_name)

        s1_dark5_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.scale.npy")
        z1_dark5_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.zero_point.npy")
        s2_dark5_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.weight.scale.npy")
        z2_dark5_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.weight.zero_point.npy")
        s3_dark5_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.scale.npy")
        z3_dark5_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.zero_point.npy")
        bias_f_dark5_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.bias.npy")
        self.bias_int_dark5_1_m0_conv1 = bias_f_dark5_1_m0_conv1
        coe_name = '../data1_coe/out_hand_dark5_1_m0_conv1_leak.coe'
        self.dark5_1_m0_conv1 = Conv2d_Q(quant_scale1=s1_dark5_1_m0_conv1, quant_zero_point1=z1_dark5_1_m0_conv1,
                                         quant_scale2=s2_dark5_1_m0_conv1,
                                         quant_zero_point2=z2_dark5_1_m0_conv1, quant_scale3=s3_dark5_1_m0_conv1,
                                         quant_zero_point3=z3_dark5_1_m0_conv1, coe_name=coe_name)

        s1_dark5_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.scale.npy")
        z1_dark5_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.zero_point.npy")
        s2_dark5_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.weight.scale.npy")
        z2_dark5_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.weight.zero_point.npy")
        s3_dark5_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.scale.npy")
        z3_dark5_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.zero_point.npy")
        bias_f_dark5_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.bias.npy")
        self.bias_int_dark5_1_m0_conv2 = bias_f_dark5_1_m0_conv2
        coe_name = '../data1_coe/out_hand_dark5_1_m0_conv2_leak.coe'
        self.dark5_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark5_1_m0_conv2, quant_zero_point1=z1_dark5_1_m0_conv2,
                                         quant_scale2=s2_dark5_1_m0_conv2,
                                         quant_zero_point2=z2_dark5_1_m0_conv2, quant_scale3=s3_dark5_1_m0_conv2,
                                         quant_zero_point3=z3_dark5_1_m0_conv2, coe_name=coe_name)

        # s1_dark5_1_m0_add = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.scale.npy")
        # z1_dark5_1_m0_add = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.zero_point.npy")
        # s2_dark5_1_m0_add = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.scale.npy")
        # z2_dark5_1_m0_add = tensorr("../para_682/backbone.backbone.dark5.1.conv1.conv.zero_point.npy")
        # s3_dark5_1_m0_add = tensorr("../para_682/backbone.backbone.dark5.1.m.0.csp.scale.npy")
        # z3_dark5_1_m0_add = tensorr("../para_682/backbone.backbone.dark5.1.m.0.csp.zero_point.npy")
        # coe_name = '../data1_coe/out_hand_dark5_1_m0_csp_leak.coe'
        # self.dark5_1_m0_add = Conv2d_Q(quant_scale1=s1_dark5_1_m0_add, quant_zero_point1=z1_dark5_1_m0_add,
        #                                quant_scale2=s2_dark5_1_m0_add,
        #                                quant_zero_point2=z2_dark5_1_m0_add, quant_scale3=s3_dark5_1_m0_add,
        #                                quant_zero_point3=z3_dark5_1_m0_add, coe_name=coe_name)

        s1_dark5_1_cat = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.scale.npy")
        z1_dark5_1_cat = tensorr("../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.zero_point.npy")
        s2_dark5_1_cat = tensorr("../para_682/backbone.backbone.dark5.1.conv2.conv.scale.npy")
        z2_dark5_1_cat = tensorr("../para_682/backbone.backbone.dark5.1.conv2.conv.zero_point.npy")
        s3_dark5_1_cat = tensorr("../para_682/backbone.backbone.dark5.1.csp1.scale.npy")
        z3_dark5_1_cat = tensorr("../para_682/backbone.backbone.dark5.1.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_s1_dark5_1_csp1_leak.coe'
        self.dark5_1_cat = Conv2d_Q(quant_scale1=s1_dark5_1_cat, quant_zero_point1=z1_dark5_1_cat,
                                    quant_scale2=s2_dark5_1_cat,
                                    quant_zero_point2=z2_dark5_1_cat, quant_scale3=s3_dark5_1_cat,
                                    quant_zero_point3=z3_dark5_1_cat, coe_name=coe_name)

        s1_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.csp1.scale.npy")
        z1_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.csp1.zero_point.npy")
        s2_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.weight.scale.npy")
        z2_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.weight.zero_point.npy")
        s3_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.scale.npy")
        z3_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.zero_point.npy")
        bias_f_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.bias.npy")
        self.bias_int_dark5_1_conv3 = bias_f_dark5_1_conv3
        coe_name = '../data1_coe/out_hand_dark5_1_conv2_leak.coe'
        self.dark5_1_conv3 = Conv2d_Q(quant_scale1=s1_dark5_1_conv3, quant_zero_point1=z1_dark5_1_conv3,
                                      quant_scale2=s2_dark5_1_conv3,
                                      quant_zero_point2=z2_dark5_1_conv3, quant_scale3=s3_dark5_1_conv3,
                                      quant_zero_point3=z3_dark5_1_conv3, coe_name=coe_name)
        # *=====================FPN===========================================
        s1_lateral_conv0 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.scale.npy")
        z1_lateral_conv0 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.zero_point.npy")
        s2_lateral_conv0 = tensorr("../para_682/backbone.lateral_conv0.conv.weight.scale.npy")
        z2_lateral_conv0 = tensorr("../para_682/backbone.lateral_conv0.conv.weight.zero_point.npy")
        s3_lateral_conv0 = tensorr("../para_682/backbone.lateral_conv0.conv.scale.npy")
        z3_lateral_conv0 = tensorr("../para_682/backbone.lateral_conv0.conv.zero_point.npy")
        bias_f_lateral_conv0 = tensorr("../para_682/backbone.lateral_conv0.conv.bias.npy")
        self.bias_int_lateral_conv0 = bias_f_lateral_conv0
        coe_name = '../data1_coe/out_hand_lateral_conv0_leak.coe'
        self.lateral_conv0 = Conv2d_Q(quant_scale1=s1_lateral_conv0, quant_zero_point1=z1_lateral_conv0,
                                      quant_scale2=s2_lateral_conv0,
                                      quant_zero_point2=z2_lateral_conv0, quant_scale3=s3_lateral_conv0,
                                      quant_zero_point3=z3_lateral_conv0, coe_name=coe_name)

        s1_csp2_cat = tensorr("../para_682/backbone.lateral_conv0.conv.scale.npy")
        z1_csp2_cat = tensorr("../para_682/backbone.lateral_conv0.conv.zero_point.npy")
        s2_csp2_cat = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.scale.npy")
        z2_csp2_cat = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.zero_point.npy")
        s3_csp2_cat = tensorr("../para_682/backbone.csp2.scale.npy")
        z3_csp2_cat = tensorr("../para_682/backbone.csp2.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp2_cat_leak.coe'
        self.csp2_cat = Conv2d_Q(quant_scale1=s1_csp2_cat, quant_zero_point1=z1_csp2_cat, quant_scale2=s2_csp2_cat,
                                 quant_zero_point2=z2_csp2_cat, quant_scale3=s3_csp2_cat,
                                 quant_zero_point3=z3_csp2_cat, coe_name=coe_name)
        # *==================C3_p4===========================
        s1_C3_p4_conv1 = tensorr("../para_682/backbone.csp2.scale.npy")
        z1_C3_p4_conv1 = tensorr("../para_682/backbone.csp2.zero_point.npy")
        s2_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.weight.scale.npy")
        z2_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.weight.zero_point.npy")
        s3_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.scale.npy")
        z3_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.zero_point.npy")
        bias_f_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.bias.npy")
        self.bias_int_C3_p4_conv1 = bias_f_C3_p4_conv1
        coe_name = '../data1_coe/out_hand_C3_p4_conv1_leak.coe'
        self.C3_p4_conv1 = Conv2d_Q(quant_scale1=s1_C3_p4_conv1, quant_zero_point1=z1_C3_p4_conv1,
                                    quant_scale2=s2_C3_p4_conv1,
                                    quant_zero_point2=z2_C3_p4_conv1, quant_scale3=s3_C3_p4_conv1,
                                    quant_zero_point3=z3_C3_p4_conv1, coe_name=coe_name)

        s1_C3_p4_conv2 = tensorr("../para_682/backbone.csp2.scale.npy")
        z1_C3_p4_conv2 = tensorr("../para_682/backbone.csp2.zero_point.npy")
        s2_C3_p4_conv2 = tensorr("../para_682/backbone.C3_p4.conv2.conv.weight.scale.npy")
        z2_C3_p4_conv2 = tensorr("../para_682/backbone.C3_p4.conv2.conv.weight.zero_point.npy")
        s3_C3_p4_conv2 = tensorr("../para_682/backbone.C3_p4.conv2.conv.scale.npy")
        z3_C3_p4_conv2 = tensorr("../para_682/backbone.C3_p4.conv2.conv.zero_point.npy")
        bias_f_C3_p4_conv2 = tensorr("../para_682/backbone.C3_p4.conv2.conv.bias.npy")
        self.bias_int_C3_p4_conv2 = bias_f_C3_p4_conv2
        coe_name = '../data1_coe/out_hand_C3_p4_conv2_leak.coe'
        self.C3_p4_conv2 = Conv2d_Q(quant_scale1=s1_C3_p4_conv2, quant_zero_point1=z1_C3_p4_conv2,
                                    quant_scale2=s2_C3_p4_conv2,
                                    quant_zero_point2=z2_C3_p4_conv2, quant_scale3=s3_C3_p4_conv2,
                                    quant_zero_point3=z3_C3_p4_conv2, coe_name=coe_name)

        s1_C3_p4_m0_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.scale.npy")
        z1_C3_p4_m0_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.zero_point.npy")
        s2_C3_p4_m0_conv1 = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.weight.scale.npy")
        z2_C3_p4_m0_conv1 = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.weight.zero_point.npy")
        s3_C3_p4_m0_conv1 = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.scale.npy")
        z3_C3_p4_m0_conv1 = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.zero_point.npy")
        bias_f_C3_p4_m0_conv1 = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.bias.npy")
        self.bias_int_C3_p4_m0_conv1 = bias_f_C3_p4_m0_conv1
        coe_name = '../data1_coe/out_hand_C3_p4_m0_conv1_leak.coe'
        self.C3_p4_m0_conv1 = Conv2d_Q(quant_scale1=s1_C3_p4_m0_conv1, quant_zero_point1=z1_C3_p4_m0_conv1,
                                       quant_scale2=s2_C3_p4_m0_conv1,
                                       quant_zero_point2=z2_C3_p4_m0_conv1, quant_scale3=s3_C3_p4_m0_conv1,
                                       quant_zero_point3=z3_C3_p4_m0_conv1, coe_name=coe_name)

        s1_C3_p4_m0_conv2 = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.scale.npy")
        z1_C3_p4_m0_conv2 = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.zero_point.npy")
        s2_C3_p4_m0_conv2 = tensorr("../para_682/backbone.C3_p4.m.0.conv2.conv.weight.scale.npy")
        z2_C3_p4_m0_conv2 = tensorr("../para_682/backbone.C3_p4.m.0.conv2.conv.weight.zero_point.npy")
        s3_C3_p4_m0_conv2 = tensorr("../para_682/backbone.C3_p4.m.0.conv2.conv.scale.npy")
        z3_C3_p4_m0_conv2 = tensorr("../para_682/backbone.C3_p4.m.0.conv2.conv.zero_point.npy")
        bias_f_C3_p4_m0_conv2 = tensorr("../para_682/backbone.C3_p4.m.0.conv2.conv.bias.npy")
        self.bias_int_C3_p4_m0_conv2 = bias_f_C3_p4_m0_conv2
        coe_name = '../data1_coe/out_hand_C3_p4_m0_conv2_leak.coe'
        self.C3_p4_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_p4_m0_conv2, quant_zero_point1=z1_C3_p4_m0_conv2,
                                       quant_scale2=s2_C3_p4_m0_conv2,
                                       quant_zero_point2=z2_C3_p4_m0_conv2, quant_scale3=s3_C3_p4_m0_conv2,
                                       quant_zero_point3=z3_C3_p4_m0_conv2, coe_name=coe_name)

        # s1_C3_p4_m0_add = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.scale.npy")
        # z1_C3_p4_m0_add = tensorr("../para_682/backbone.C3_p4.m.0.conv1.conv.zero_point.npy")
        # s2_C3_p4_m0_add = tensorr("../para_682/backbone.C3_p4.conv1.conv.scale.npy")
        # z2_C3_p4_m0_add = tensorr("../para_682/backbone.C3_p4.conv1.conv.zero_point.npy")
        # s3_C3_p4_m0_add = tensorr("../para_682/backbone.C3_p4.m.0.csp.scale.npy")
        # z3_C3_p4_m0_add = tensorr("../para_682/backbone.C3_p4.m.0.csp.zero_point.npy")
        # coe_name = '../data1_coe/out_hand_C3_p4_m0_add.coe'
        # self.C3_p4_m0_add = Conv2d_Q(quant_scale1=s1_C3_p4_m0_add, quant_zero_point1=z1_C3_p4_m0_add,
        #                              quant_scale2=s2_C3_p4_m0_add,
        #                              quant_zero_point2=z2_C3_p4_m0_add, quant_scale3=s3_C3_p4_m0_add,
        #                              quant_zero_point3=z3_C3_p4_m0_add, coe_name=coe_name)

        s1_C3_p4_csp1_cat = tensorr("../para_682/backbone.C3_p4.m.0.conv2.conv.scale.npy")
        z1_C3_p4_csp1_cat = tensorr("../para_682/backbone.C3_p4.m.0.conv2.conv.zero_point.npy")
        s2_C3_p4_csp1_cat = tensorr("../para_682/backbone.C3_p4.conv2.conv.scale.npy")
        z2_C3_p4_csp1_cat = tensorr("../para_682/backbone.C3_p4.conv2.conv.zero_point.npy")
        s3_C3_p4_csp1_cat = tensorr("../para_682/backbone.C3_p4.csp1.scale.npy")
        z3_C3_p4_csp1_cat = tensorr("../para_682/backbone.C3_p4.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_C3_p4_csp1_cat.coe'
        self.C3_p4_csp1_cat = Conv2d_Q(quant_scale1=s1_C3_p4_csp1_cat, quant_zero_point1=z1_C3_p4_csp1_cat,
                                       quant_scale2=s2_C3_p4_csp1_cat,
                                       quant_zero_point2=z2_C3_p4_csp1_cat, quant_scale3=s3_C3_p4_csp1_cat,
                                       quant_zero_point3=z3_C3_p4_csp1_cat, coe_name=coe_name)

        s1_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.csp1.scale.npy")
        z1_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.csp1.zero_point.npy")
        s2_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.weight.scale.npy")
        z2_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.weight.zero_point.npy")
        s3_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.scale.npy")
        z3_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.zero_point.npy")
        bias_f_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.bias.npy")
        self.bias_int_C3_p4_conv3 = bias_f_C3_p4_conv3
        coe_name = '../data1_coe/out_hand_C3_p4_conv3.coe'
        self.C3_p4_conv3 = Conv2d_Q(quant_scale1=s1_C3_p4_conv3, quant_zero_point1=z1_C3_p4_conv3,
                                    quant_scale2=s2_C3_p4_conv3,
                                    quant_zero_point2=z2_C3_p4_conv3, quant_scale3=s3_C3_p4_conv3,
                                    quant_zero_point3=z3_C3_p4_conv3, coe_name=coe_name)
        # *==================================================
        s1_reduce_conv1 = tensorr("../para_682/backbone.C3_p4.conv3.conv.scale.npy")
        z1_reduce_conv1 = tensorr("../para_682/backbone.C3_p4.conv3.conv.zero_point.npy")
        s2_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.weight.scale.npy")
        z2_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.weight.zero_point.npy")
        s3_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.scale.npy")
        z3_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.zero_point.npy")
        bias_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.bias.npy")
        self.bias_int_reduce_conv1 = bias_reduce_conv1
        coe_name = '../data1_coe/out_hand_reduce_conv1.coe'
        self.reduce_conv1 = Conv2d_Q(quant_scale1=s1_reduce_conv1, quant_zero_point1=z1_reduce_conv1,
                                     quant_scale2=s2_reduce_conv1,
                                     quant_zero_point2=z2_reduce_conv1, quant_scale3=s3_reduce_conv1,
                                     quant_zero_point3=z3_reduce_conv1, coe_name=coe_name)

        s1_csp3_cat = tensorr("../para_682/backbone.reduce_conv1.conv.scale.npy")
        z1_csp3_cat = tensorr("../para_682/backbone.reduce_conv1.conv.zero_point.npy")
        s2_csp3_cat = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.scale.npy")
        z2_csp3_cat = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.zero_point.npy")
        s3_csp3_cat = tensorr("../para_682/backbone.csp3.scale.npy")
        z3_csp3_cat = tensorr("../para_682/backbone.csp3.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp3_cat_leak.coe'
        self.csp3_cat = Conv2d_Q(quant_scale1=s1_csp3_cat, quant_zero_point1=z1_csp3_cat, quant_scale2=s2_csp3_cat,
                                 quant_zero_point2=z2_csp3_cat, quant_scale3=s3_csp3_cat,
                                 quant_zero_point3=z3_csp3_cat, coe_name=coe_name)
        # *==========================C3_p3=======================================
        s1_C3_p3_conv1 = tensorr("../para_682/backbone.csp3.scale.npy")
        z1_C3_p3_conv1 = tensorr("../para_682/backbone.csp3.zero_point.npy")
        s2_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.weight.scale.npy")
        z2_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.weight.zero_point.npy")
        s3_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.scale.npy")
        z3_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.zero_point.npy")
        bias_f_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.bias.npy")
        self.bias_int_C3_p3_conv1 = bias_f_C3_p3_conv1
        coe_name = '../data1_coe/out_hand_C3_p3_conv1_leak.coe'
        self.C3_p3_conv1 = Conv2d_Q(quant_scale1=s1_C3_p3_conv1, quant_zero_point1=z1_C3_p3_conv1,
                                    quant_scale2=s2_C3_p3_conv1,
                                    quant_zero_point2=z2_C3_p3_conv1, quant_scale3=s3_C3_p3_conv1,
                                    quant_zero_point3=z3_C3_p3_conv1, coe_name=coe_name)

        s1_C3_p3_conv2 = tensorr("../para_682/backbone.csp3.scale.npy")
        z1_C3_p3_conv2 = tensorr("../para_682/backbone.csp3.zero_point.npy")
        s2_C3_p3_conv2 = tensorr("../para_682/backbone.C3_p3.conv2.conv.weight.scale.npy")
        z2_C3_p3_conv2 = tensorr("../para_682/backbone.C3_p3.conv2.conv.weight.zero_point.npy")
        s3_C3_p3_conv2 = tensorr("../para_682/backbone.C3_p3.conv2.conv.scale.npy")
        z3_C3_p3_conv2 = tensorr("../para_682/backbone.C3_p3.conv2.conv.zero_point.npy")
        bias_f_C3_p3_conv2 = tensorr("../para_682/backbone.C3_p3.conv2.conv.bias.npy")
        self.bias_int_C3_p3_conv2 = bias_f_C3_p3_conv2
        coe_name = '../data1_coe/out_hand_C3_p3_conv2_leak.coe'
        self.C3_p3_conv2 = Conv2d_Q(quant_scale1=s1_C3_p3_conv2, quant_zero_point1=z1_C3_p3_conv2,
                                    quant_scale2=s2_C3_p3_conv2,
                                    quant_zero_point2=z2_C3_p3_conv2, quant_scale3=s3_C3_p3_conv2,
                                    quant_zero_point3=z3_C3_p3_conv2, coe_name=coe_name)

        s1_C3_p3_m0_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.scale.npy")
        z1_C3_p3_m0_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.zero_point.npy")
        s2_C3_p3_m0_conv1 = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.weight.scale.npy")
        z2_C3_p3_m0_conv1 = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.weight.zero_point.npy")
        s3_C3_p3_m0_conv1 = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.scale.npy")
        z3_C3_p3_m0_conv1 = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.zero_point.npy")
        bias_f_C3_p3_m0_conv1 = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.bias.npy")
        self.bias_int_C3_p3_m0_conv1 = bias_f_C3_p3_m0_conv1
        coe_name = '../data1_coe/out_hand_C3_p3_m0_conv1_leak.coe'
        self.C3_p3_m0_conv1 = Conv2d_Q(quant_scale1=s1_C3_p3_m0_conv1, quant_zero_point1=z1_C3_p3_m0_conv1,
                                       quant_scale2=s2_C3_p3_m0_conv1,
                                       quant_zero_point2=z2_C3_p3_m0_conv1, quant_scale3=s3_C3_p3_m0_conv1,
                                       quant_zero_point3=z3_C3_p3_m0_conv1, coe_name=coe_name)

        s1_C3_p3_m0_conv2 = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.scale.npy")
        z1_C3_p3_m0_conv2 = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.zero_point.npy")
        s2_C3_p3_m0_conv2 = tensorr("../para_682/backbone.C3_p3.m.0.conv2.conv.weight.scale.npy")
        z2_C3_p3_m0_conv2 = tensorr("../para_682/backbone.C3_p3.m.0.conv2.conv.weight.zero_point.npy")
        s3_C3_p3_m0_conv2 = tensorr("../para_682/backbone.C3_p3.m.0.conv2.conv.scale.npy")
        z3_C3_p3_m0_conv2 = tensorr("../para_682/backbone.C3_p3.m.0.conv2.conv.zero_point.npy")
        bias_f_C3_p3_m0_conv2 = tensorr("../para_682/backbone.C3_p3.m.0.conv2.conv.bias.npy")
        self.bias_int_C3_p3_m0_conv2 = bias_f_C3_p3_m0_conv2
        coe_name = '../data1_coe/out_hand_C3_p3_m0_conv2_leak.coe'
        self.C3_p3_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_p3_m0_conv2, quant_zero_point1=z1_C3_p3_m0_conv2,
                                       quant_scale2=s2_C3_p3_m0_conv2,
                                       quant_zero_point2=z2_C3_p3_m0_conv2, quant_scale3=s3_C3_p3_m0_conv2,
                                       quant_zero_point3=z3_C3_p3_m0_conv2, coe_name=coe_name)

        # s1_C3_p3_m0_add = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.scale.npy")
        # z1_C3_p3_m0_add = tensorr("../para_682/backbone.C3_p3.m.0.conv1.conv.zero_point.npy")
        # s2_C3_p3_m0_add = tensorr("../para_682/backbone.C3_p3.conv1.conv.scale.npy")
        # z2_C3_p3_m0_add = tensorr("../para_682/backbone.C3_p3.conv1.conv.zero_point.npy")
        # s3_C3_p3_m0_add = tensorr("../para_682/backbone.C3_p3.m.0.csp.scale.npy")
        # z3_C3_p3_m0_add = tensorr("../para_682/backbone.C3_p3.m.0.csp.zero_point.npy")
        # coe_name = '../data1_coe/out_hand_C3_p3_m0_add.coe'
        # self.C3_p3_m0_add = Conv2d_Q(quant_scale1=s1_C3_p3_m0_add, quant_zero_point1=z1_C3_p3_m0_add,
        #                              quant_scale2=s2_C3_p3_m0_add,
        #                              quant_zero_point2=z2_C3_p3_m0_add, quant_scale3=s3_C3_p3_m0_add,
        #                              quant_zero_point3=z3_C3_p3_m0_add, coe_name=coe_name)

        s1_C3_p3_csp1_cat = tensorr("../para_682/backbone.C3_p3.m.0.conv2.conv.scale.npy")
        z1_C3_p3_csp1_cat = tensorr("../para_682/backbone.C3_p3.m.0.conv2.conv.zero_point.npy")
        s2_C3_p3_csp1_cat = tensorr("../para_682/backbone.C3_p3.conv2.conv.scale.npy")
        z2_C3_p3_csp1_cat = tensorr("../para_682/backbone.C3_p3.conv2.conv.zero_point.npy")
        s3_C3_p3_csp1_cat = tensorr("../para_682/backbone.C3_p3.csp1.scale.npy")
        z3_C3_p3_csp1_cat = tensorr("../para_682/backbone.C3_p3.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_C3_p3_csp1_cat.coe'
        self.C3_p3_csp1_cat = Conv2d_Q(quant_scale1=s1_C3_p3_csp1_cat, quant_zero_point1=z1_C3_p3_csp1_cat,
                                       quant_scale2=s2_C3_p3_csp1_cat,
                                       quant_zero_point2=z2_C3_p3_csp1_cat, quant_scale3=s3_C3_p3_csp1_cat,
                                       quant_zero_point3=z3_C3_p3_csp1_cat, coe_name=coe_name)

        s1_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.csp1.scale.npy")
        z1_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.csp1.zero_point.npy")
        s2_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.weight.scale.npy")
        z2_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.weight.zero_point.npy")
        s3_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.scale.npy")
        z3_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.zero_point.npy")
        bias_f_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.bias.npy")
        self.bias_int_C3_p3_conv3 = bias_f_C3_p3_conv3
        coe_name = '../data1_coe/out_hand_C3_p3_conv3.coe'
        self.C3_p3_conv3 = Conv2d_Q(quant_scale1=s1_C3_p3_conv3, quant_zero_point1=z1_C3_p3_conv3,
                                    quant_scale2=s2_C3_p3_conv3,
                                    quant_zero_point2=z2_C3_p3_conv3, quant_scale3=s3_C3_p3_conv3,
                                    quant_zero_point3=z3_C3_p3_conv3, coe_name=coe_name)
        # *==========================================================================
        s1_bu_conv2 = tensorr("../para_682/backbone.C3_p3.conv3.conv.scale.npy")
        z1_bu_conv2 = tensorr("../para_682/backbone.C3_p3.conv3.conv.zero_point.npy")
        s2_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.weight.scale.npy")
        z2_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.weight.zero_point.npy")
        s3_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.scale.npy")
        z3_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.zero_point.npy")
        bias_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.bias.npy")
        self.bias_int_bu_conv2 = bias_bu_conv2
        coe_name = '../data1_coe/out_hand_bu_conv2.coe'
        self.bu_conv2 = Conv2d_Q(quant_scale1=s1_bu_conv2, quant_zero_point1=z1_bu_conv2, quant_scale2=s2_bu_conv2,
                                 quant_zero_point2=z2_bu_conv2, quant_scale3=s3_bu_conv2,
                                 quant_zero_point3=z3_bu_conv2, coe_name=coe_name)

        s1_csp4_cat = tensorr("../para_682/backbone.bu_conv2.conv.scale.npy")
        z1_csp4_cat = tensorr("../para_682/backbone.bu_conv2.conv.zero_point.npy")
        s2_csp4_cat = tensorr("../para_682/backbone.reduce_conv1.conv.scale.npy")
        z2_csp4_cat = tensorr("../para_682/backbone.reduce_conv1.conv.zero_point.npy")
        s3_csp4_cat = tensorr("../para_682/backbone.csp4.scale.npy")
        z3_csp4_cat = tensorr("../para_682/backbone.csp4.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp4_cat_leak.coe'
        self.csp4_cat = Conv2d_Q(quant_scale1=s1_csp4_cat, quant_zero_point1=z1_csp4_cat, quant_scale2=s2_csp4_cat,
                                 quant_zero_point2=z2_csp4_cat, quant_scale3=s3_csp4_cat,
                                 quant_zero_point3=z3_csp4_cat, coe_name=coe_name)
        # *====================C3_n3==========================
        s1_C3_n3_conv1 = tensorr("../para_682/backbone.csp4.scale.npy")
        z1_C3_n3_conv1 = tensorr("../para_682/backbone.csp4.zero_point.npy")
        s2_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.weight.scale.npy")
        z2_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.weight.zero_point.npy")
        s3_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.scale.npy")
        z3_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.zero_point.npy")
        bias_f_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.bias.npy")
        self.bias_int_C3_n3_conv1 = bias_f_C3_n3_conv1
        coe_name = '../data1_coe/out_hand_C3_n3_conv1_leak.coe'
        self.C3_n3_conv1 = Conv2d_Q(quant_scale1=s1_C3_n3_conv1, quant_zero_point1=z1_C3_n3_conv1,
                                    quant_scale2=s2_C3_n3_conv1,
                                    quant_zero_point2=z2_C3_n3_conv1, quant_scale3=s3_C3_n3_conv1,
                                    quant_zero_point3=z3_C3_n3_conv1, coe_name=coe_name)

        s1_C3_n3_conv2 = tensorr("../para_682/backbone.csp4.scale.npy")
        z1_C3_n3_conv2 = tensorr("../para_682/backbone.csp4.zero_point.npy")
        s2_C3_n3_conv2 = tensorr("../para_682/backbone.C3_n3.conv2.conv.weight.scale.npy")
        z2_C3_n3_conv2 = tensorr("../para_682/backbone.C3_n3.conv2.conv.weight.zero_point.npy")
        s3_C3_n3_conv2 = tensorr("../para_682/backbone.C3_n3.conv2.conv.scale.npy")
        z3_C3_n3_conv2 = tensorr("../para_682/backbone.C3_n3.conv2.conv.zero_point.npy")
        bias_f_C3_n3_conv2 = tensorr("../para_682/backbone.C3_n3.conv2.conv.bias.npy")
        self.bias_int_C3_n3_conv2 = bias_f_C3_n3_conv2
        coe_name = '../data1_coe/out_hand_C3_n3_conv2_leak.coe'
        self.C3_n3_conv2 = Conv2d_Q(quant_scale1=s1_C3_n3_conv2, quant_zero_point1=z1_C3_n3_conv2,
                                    quant_scale2=s2_C3_n3_conv2,
                                    quant_zero_point2=z2_C3_n3_conv2, quant_scale3=s3_C3_n3_conv2,
                                    quant_zero_point3=z3_C3_n3_conv2, coe_name=coe_name)

        s1_C3_n3_m0_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.scale.npy")
        z1_C3_n3_m0_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.zero_point.npy")
        s2_C3_n3_m0_conv1 = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.weight.scale.npy")
        z2_C3_n3_m0_conv1 = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.weight.zero_point.npy")
        s3_C3_n3_m0_conv1 = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.scale.npy")
        z3_C3_n3_m0_conv1 = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.zero_point.npy")
        bias_f_C3_n3_m0_conv1 = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.bias.npy")
        self.bias_C3_n3_m0_conv1 = bias_f_C3_n3_m0_conv1
        coe_name = '../data1_coe/out_hand_C3_n3_m0_conv1_leak.coe'
        self.C3_n3_m0_conv1 = Conv2d_Q(quant_scale1=s1_C3_n3_m0_conv1, quant_zero_point1=z1_C3_n3_m0_conv1,
                                       quant_scale2=s2_C3_n3_m0_conv1,
                                       quant_zero_point2=z2_C3_n3_m0_conv1, quant_scale3=s3_C3_n3_m0_conv1,
                                       quant_zero_point3=z3_C3_n3_m0_conv1, coe_name=coe_name)

        s1_C3_n3_m0_conv2 = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.scale.npy")
        z1_C3_n3_m0_conv2 = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.zero_point.npy")
        s2_C3_n3_m0_conv2 = tensorr("../para_682/backbone.C3_n3.m.0.conv2.conv.weight.scale.npy")
        z2_C3_n3_m0_conv2 = tensorr("../para_682/backbone.C3_n3.m.0.conv2.conv.weight.zero_point.npy")
        s3_C3_n3_m0_conv2 = tensorr("../para_682/backbone.C3_n3.m.0.conv2.conv.scale.npy")
        z3_C3_n3_m0_conv2 = tensorr("../para_682/backbone.C3_n3.m.0.conv2.conv.zero_point.npy")
        bias_f_C3_n3_m0_conv2 = tensorr("../para_682/backbone.C3_n3.m.0.conv2.conv.bias.npy")
        self.bias_int_C3_n3_m0_conv2 = bias_f_C3_n3_m0_conv2
        coe_name = '../data1_coe/out_hand_C3_n3_m0_conv2_leak.coe'
        self.C3_n3_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_n3_m0_conv2, quant_zero_point1=z1_C3_n3_m0_conv2,
                                       quant_scale2=s2_C3_n3_m0_conv2,
                                       quant_zero_point2=z2_C3_n3_m0_conv2, quant_scale3=s3_C3_n3_m0_conv2,
                                       quant_zero_point3=z3_C3_n3_m0_conv2, coe_name=coe_name)

        # s1_C3_n3_m0_add = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.scale.npy")
        # z1_C3_n3_m0_add = tensorr("../para_682/backbone.C3_n3.m.0.conv1.conv.zero_point.npy")
        # s2_C3_n3_m0_add = tensorr("../para_682/backbone.C3_n3.conv1.conv.scale.npy")
        # z2_C3_n3_m0_add = tensorr("../para_682/backbone.C3_n3.conv1.conv.zero_point.npy")
        # s3_C3_n3_m0_add = tensorr("../para_682/backbone.C3_n3.m.0.csp.scale.npy")
        # z3_C3_n3_m0_add = tensorr("../para_682/backbone.C3_n3.m.0.csp.zero_point.npy")
        # coe_name = '../data1_coe/out_hand_C3_n3_m0_add.coe'
        # self.C3_n3_m0_add = Conv2d_Q(quant_scale1=s1_C3_n3_m0_add, quant_zero_point1=z1_C3_n3_m0_add,
        #                              quant_scale2=s2_C3_n3_m0_add,
        #                              quant_zero_point2=z2_C3_n3_m0_add, quant_scale3=s3_C3_n3_m0_add,
        #                              quant_zero_point3=z3_C3_n3_m0_add, coe_name=coe_name)

        s1_C3_n3_csp1_cat = tensorr("../para_682/backbone.C3_n3.m.0.conv2.conv.scale.npy")
        z1_C3_n3_csp1_cat = tensorr("../para_682/backbone.C3_n3.m.0.conv2.conv.zero_point.npy")
        s2_C3_n3_csp1_cat = tensorr("../para_682/backbone.C3_n3.conv2.conv.scale.npy")
        z2_C3_n3_csp1_cat = tensorr("../para_682/backbone.C3_n3.conv2.conv.zero_point.npy")
        s3_C3_n3_csp1_cat = tensorr("../para_682/backbone.C3_n3.csp1.scale.npy")
        z3_C3_n3_csp1_cat = tensorr("../para_682/backbone.C3_n3.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_C3_n3_csp1_cat.coe'
        self.C3_n3_csp1_cat = Conv2d_Q(quant_scale1=s1_C3_n3_csp1_cat, quant_zero_point1=z1_C3_n3_csp1_cat,
                                       quant_scale2=s2_C3_n3_csp1_cat,
                                       quant_zero_point2=z2_C3_n3_csp1_cat, quant_scale3=s3_C3_n3_csp1_cat,
                                       quant_zero_point3=z3_C3_n3_csp1_cat, coe_name=coe_name)

        s1_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.csp1.scale.npy")
        z1_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.csp1.zero_point.npy")
        s2_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.weight.scale.npy")
        z2_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.weight.zero_point.npy")
        s3_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.scale.npy")
        z3_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.zero_point.npy")
        bias_f_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.bias.npy")
        self.bias_int_C3_n3_conv3 = bias_f_C3_n3_conv3
        coe_name = '../data1_coe/out_hand_C3_n3_conv3.coe'
        self.C3_n3_conv3 = Conv2d_Q(quant_scale1=s1_C3_n3_conv3, quant_zero_point1=z1_C3_n3_conv3,
                                    quant_scale2=s2_C3_n3_conv3,
                                    quant_zero_point2=z2_C3_n3_conv3, quant_scale3=s3_C3_n3_conv3,
                                    quant_zero_point3=z3_C3_n3_conv3, coe_name=coe_name)
        # *==================================================================================
        s1_bu_conv1 = tensorr("../para_682/backbone.C3_n3.conv3.conv.scale.npy")
        z1_bu_conv1 = tensorr("../para_682/backbone.C3_n3.conv3.conv.zero_point.npy")
        s2_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.weight.scale.npy")
        z2_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.weight.zero_point.npy")
        s3_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.scale.npy")
        z3_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.zero_point.npy")
        bias_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.bias.npy")
        self.bias_int_bu_conv1 = bias_bu_conv1
        coe_name = '../data1_coe/out_hand_bu_conv1.coe'
        self.bu_conv1 = Conv2d_Q(quant_scale1=s1_bu_conv1, quant_zero_point1=z1_bu_conv1, quant_scale2=s2_bu_conv1,
                                 quant_zero_point2=z2_bu_conv1, quant_scale3=s3_bu_conv1,
                                 quant_zero_point3=z3_bu_conv1, coe_name=coe_name)

        s1_csp5_cat = tensorr("../para_682/backbone.bu_conv1.conv.scale.npy")
        z1_csp5_cat = tensorr("../para_682/backbone.bu_conv1.conv.zero_point.npy")
        s2_csp5_cat = tensorr("../para_682/backbone.lateral_conv0.conv.scale.npy")
        z2_csp5_cat = tensorr("../para_682/backbone.lateral_conv0.conv.zero_point.npy")
        s3_csp5_cat = tensorr("../para_682/backbone.csp5.scale.npy")
        z3_csp5_cat = tensorr("../para_682/backbone.csp5.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp5_cat_leak.coe'
        self.csp5_cat = Conv2d_Q(quant_scale1=s1_csp5_cat, quant_zero_point1=z1_csp5_cat, quant_scale2=s2_csp5_cat,
                                 quant_zero_point2=z2_csp5_cat, quant_scale3=s3_csp5_cat,
                                 quant_zero_point3=z3_csp5_cat, coe_name=coe_name)
        # *=======================C3_n4============================
        s1_C3_n4_conv1 = tensorr("../para_682/backbone.csp5.scale.npy")
        z1_C3_n4_conv1 = tensorr("../para_682/backbone.csp5.zero_point.npy")
        s2_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.weight.scale.npy")
        z2_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.weight.zero_point.npy")
        s3_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.scale.npy")
        z3_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.zero_point.npy")
        bias_f_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.bias.npy")
        self.bias_int_C3_n4_conv1 = bias_f_C3_n4_conv1
        coe_name = '../data1_coe/out_hand_C3_n4_conv1_leak.coe'
        self.C3_n4_conv1 = Conv2d_Q(quant_scale1=s1_C3_n4_conv1, quant_zero_point1=z1_C3_n4_conv1,
                                    quant_scale2=s2_C3_n4_conv1,
                                    quant_zero_point2=z2_C3_n4_conv1, quant_scale3=s3_C3_n4_conv1,
                                    quant_zero_point3=z3_C3_n4_conv1, coe_name=coe_name)

        s1_C3_n4_conv2 = tensorr("../para_682/backbone.csp5.scale.npy")
        z1_C3_n4_conv2 = tensorr("../para_682/backbone.csp5.zero_point.npy")
        s2_C3_n4_conv2 = tensorr("../para_682/backbone.C3_n4.conv2.conv.weight.scale.npy")
        z2_C3_n4_conv2 = tensorr("../para_682/backbone.C3_n4.conv2.conv.weight.zero_point.npy")
        s3_C3_n4_conv2 = tensorr("../para_682/backbone.C3_n4.conv2.conv.scale.npy")
        z3_C3_n4_conv2 = tensorr("../para_682/backbone.C3_n4.conv2.conv.zero_point.npy")
        bias_f_C3_n4_conv2 = tensorr("../para_682/backbone.C3_n4.conv2.conv.bias.npy")
        self.bias_int_C3_n4_conv2 = bias_f_C3_n4_conv2
        coe_name = '../data1_coe/out_hand_C3_n4_conv2_leak.coe'
        self.C3_n4_conv2 = Conv2d_Q(quant_scale1=s1_C3_n4_conv2, quant_zero_point1=z1_C3_n4_conv2,
                                    quant_scale2=s2_C3_n4_conv2,
                                    quant_zero_point2=z2_C3_n4_conv2, quant_scale3=s3_C3_n4_conv2,
                                    quant_zero_point3=z3_C3_n4_conv2, coe_name=coe_name)

        s1_C3_n4_m0_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.scale.npy")
        z1_C3_n4_m0_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.zero_point.npy")
        s2_C3_n4_m0_conv1 = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.weight.scale.npy")
        z2_C3_n4_m0_conv1 = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.weight.zero_point.npy")
        s3_C3_n4_m0_conv1 = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.scale.npy")
        z3_C3_n4_m0_conv1 = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.zero_point.npy")
        bias_f_C3_n4_m0_conv1 = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.bias.npy")
        self.bias_C3_n4_m0_conv1 = bias_f_C3_n4_m0_conv1
        coe_name = '../data1_coe/out_hand_C3_n4_m0_conv1_leak.coe'
        self.C3_n4_m0_conv1 = Conv2d_Q(quant_scale1=s1_C3_n4_m0_conv1, quant_zero_point1=z1_C3_n4_m0_conv1,
                                       quant_scale2=s2_C3_n4_m0_conv1,
                                       quant_zero_point2=z2_C3_n4_m0_conv1, quant_scale3=s3_C3_n4_m0_conv1,
                                       quant_zero_point3=z3_C3_n4_m0_conv1, coe_name=coe_name)

        s1_C3_n4_m0_conv2 = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.scale.npy")
        z1_C3_n4_m0_conv2 = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.zero_point.npy")
        s2_C3_n4_m0_conv2 = tensorr("../para_682/backbone.C3_n4.m.0.conv2.conv.weight.scale.npy")
        z2_C3_n4_m0_conv2 = tensorr("../para_682/backbone.C3_n4.m.0.conv2.conv.weight.zero_point.npy")
        s3_C3_n4_m0_conv2 = tensorr("../para_682/backbone.C3_n4.m.0.conv2.conv.scale.npy")
        z3_C3_n4_m0_conv2 = tensorr("../para_682/backbone.C3_n4.m.0.conv2.conv.zero_point.npy")
        bias_f_C3_n4_m0_conv2 = tensorr("../para_682/backbone.C3_n4.m.0.conv2.conv.bias.npy")
        self.bias_int_C3_n4_m0_conv2 = bias_f_C3_n4_m0_conv2
        coe_name = '../data1_coe/out_hand_C3_n4_m0_conv2_leak.coe'
        self.C3_n4_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_n4_m0_conv2, quant_zero_point1=z1_C3_n4_m0_conv2,
                                       quant_scale2=s2_C3_n4_m0_conv2,
                                       quant_zero_point2=z2_C3_n4_m0_conv2, quant_scale3=s3_C3_n4_m0_conv2,
                                       quant_zero_point3=z3_C3_n4_m0_conv2, coe_name=coe_name)

        # s1_C3_n4_m0_add = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.scale.npy")
        # z1_C3_n4_m0_add = tensorr("../para_682/backbone.C3_n4.m.0.conv1.conv.zero_point.npy")
        # s2_C3_n4_m0_add = tensorr("../para_682/backbone.C3_n4.conv1.conv.scale.npy")
        # z2_C3_n4_m0_add = tensorr("../para_682/backbone.C3_n4.conv1.conv.zero_point.npy")
        # s3_C3_n4_m0_add = tensorr("../para_682/backbone.C3_n4.m.0.csp.scale.npy")
        # z3_C3_n4_m0_add = tensorr("../para_682/backbone.C3_n4.m.0.csp.zero_point.npy")
        # coe_name = '../data1_coe/out_hand_C3_n4_m0_add.coe'
        # self.C3_n4_m0_add = Conv2d_Q(quant_scale1=s1_C3_n4_m0_add, quant_zero_point1=z1_C3_n4_m0_add,
        #                              quant_scale2=s2_C3_n4_m0_add,
        #                              quant_zero_point2=z2_C3_n4_m0_add, quant_scale3=s3_C3_n4_m0_add,
        #                              quant_zero_point3=z3_C3_n4_m0_add, coe_name=coe_name)

        s1_C3_n4_csp1_cat = tensorr("../para_682/backbone.C3_n4.m.0.conv2.conv.scale.npy")
        z1_C3_n4_csp1_cat = tensorr("../para_682/backbone.C3_n4.m.0.conv2.conv.zero_point.npy")
        s2_C3_n4_csp1_cat = tensorr("../para_682/backbone.C3_n4.conv2.conv.scale.npy")
        z2_C3_n4_csp1_cat = tensorr("../para_682/backbone.C3_n4.conv2.conv.zero_point.npy")
        s3_C3_n4_csp1_cat = tensorr("../para_682/backbone.C3_n4.csp1.scale.npy")
        z3_C3_n4_csp1_cat = tensorr("../para_682/backbone.C3_n4.csp1.zero_point.npy")
        coe_name = '../data1_coe/out_hand_C3_n4_csp1_cat.coe'
        self.C3_n4_csp1_cat = Conv2d_Q(quant_scale1=s1_C3_n4_csp1_cat, quant_zero_point1=z1_C3_n4_csp1_cat,
                                       quant_scale2=s2_C3_n4_csp1_cat,
                                       quant_zero_point2=z2_C3_n4_csp1_cat, quant_scale3=s3_C3_n4_csp1_cat,
                                       quant_zero_point3=z3_C3_n4_csp1_cat, coe_name=coe_name)

        s1_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.csp1.scale.npy")
        z1_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.csp1.zero_point.npy")
        s2_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.weight.scale.npy")
        z2_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.weight.zero_point.npy")
        s3_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.scale.npy")
        z3_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.zero_point.npy")
        bias_f_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.bias.npy")
        self.bias_int_C3_n4_conv3 = bias_f_C3_n4_conv3
        coe_name = '../data1_coe/out_hand_C3_n4_conv3.coe'
        self.C3_n4_conv3 = Conv2d_Q(quant_scale1=s1_C3_n4_conv3, quant_zero_point1=z1_C3_n4_conv3,
                                    quant_scale2=s2_C3_n4_conv3,
                                    quant_zero_point2=z2_C3_n4_conv3, quant_scale3=s3_C3_n4_conv3,
                                    quant_zero_point3=z3_C3_n4_conv3, coe_name=coe_name)
        # *================================output_p3===================================================
        s1_stems0_conv = tensorr("../para_682/backbone.C3_p3.conv3.conv.scale.npy")
        z1_stems0_conv = tensorr("../para_682/backbone.C3_p3.conv3.conv.zero_point.npy")
        s2_stems0_conv = tensorr("../para_682/head.stems.0.conv.weight.scale.npy")
        z2_stems0_conv = tensorr("../para_682/head.stems.0.conv.weight.zero_point.npy")
        s3_stems0_conv = tensorr("../para_682/head.stems.0.conv.scale.npy")
        z3_stems0_conv = tensorr("../para_682/head.stems.0.conv.zero_point.npy")
        bias_f_stems0_conv = tensorr("../para_682/head.stems.0.conv.bias.npy")
        self.bias_int_stems0_conv = bias_f_stems0_conv
        coe_name = '../data1_coe/out_hand_stems0_conv.coe'
        self.stems0_conv = Conv2d_Q(quant_scale1=s1_stems0_conv, quant_zero_point1=z1_stems0_conv,
                                    quant_scale2=s2_stems0_conv,
                                    quant_zero_point2=z2_stems0_conv, quant_scale3=s3_stems0_conv,
                                    quant_zero_point3=z3_stems0_conv, coe_name=coe_name)

        s1_cls_convs0_0 = tensorr("../para_682/head.stems.0.conv.scale.npy")
        z1_cls_convs0_0 = tensorr("../para_682/head.stems.0.conv.zero_point.npy")
        s2_cls_convs0_0 = tensorr("../para_682/head.cls_convs.0.0.conv.weight.scale.npy")
        z2_cls_convs0_0 = tensorr("../para_682/head.cls_convs.0.0.conv.weight.zero_point.npy")
        s3_cls_convs0_0 = tensorr("../para_682/head.cls_convs.0.0.conv.scale.npy")
        z3_cls_convs0_0 = tensorr("../para_682/head.cls_convs.0.0.conv.zero_point.npy")
        bias_f_cls_convs0_0 = tensorr("../para_682/head.cls_convs.0.0.conv.bias.npy")
        self.bias_int_cls_convs0_0 = bias_f_cls_convs0_0
        coe_name = '../data1_coe/out_hand_cls_convs0_0.coe'
        self.cls_convs0_0 = Conv2d_Q(quant_scale1=s1_cls_convs0_0, quant_zero_point1=z1_cls_convs0_0,
                                     quant_scale2=s2_cls_convs0_0,
                                     quant_zero_point2=z2_cls_convs0_0, quant_scale3=s3_cls_convs0_0,
                                     quant_zero_point3=z3_cls_convs0_0, coe_name=coe_name)

        s1_cls_convs0_1 = tensorr("../para_682/head.cls_convs.0.0.conv.scale.npy")
        z1_cls_convs0_1 = tensorr("../para_682/head.cls_convs.0.0.conv.zero_point.npy")
        s2_cls_convs0_1 = tensorr("../para_682/head.cls_convs.0.1.conv.weight.scale.npy")
        z2_cls_convs0_1 = tensorr("../para_682/head.cls_convs.0.1.conv.weight.zero_point.npy")
        s3_cls_convs0_1 = tensorr("../para_682/head.cls_convs.0.1.conv.scale.npy")
        z3_cls_convs0_1 = tensorr("../para_682/head.cls_convs.0.1.conv.zero_point.npy")
        bias_f_cls_convs0_1 = tensorr("../para_682/head.cls_convs.0.1.conv.bias.npy")
        self.bias_int_cls_convs0_1 = bias_f_cls_convs0_1
        coe_name = '../data1_coe/out_hand_cls_convs0_1.coe'
        self.cls_convs0_1 = Conv2d_Q(quant_scale1=s1_cls_convs0_1, quant_zero_point1=z1_cls_convs0_1,
                                     quant_scale2=s2_cls_convs0_1,
                                     quant_zero_point2=z2_cls_convs0_1, quant_scale3=s3_cls_convs0_1,
                                     quant_zero_point3=z3_cls_convs0_1, coe_name=coe_name)

        s1_cls_preds0 = tensorr("../para_682/head.cls_convs.0.1.conv.scale.npy")
        z1_cls_preds0 = tensorr("../para_682/head.cls_convs.0.1.conv.zero_point.npy")
        s2_cls_preds0 = tensorr("../para_682/head.cls_preds.0.weight.scale.npy")
        z2_cls_preds0 = tensorr("../para_682/head.cls_preds.0.weight.zero_point.npy")
        s3_cls_preds0 = tensorr("../para_682/head.cls_preds.0.scale.npy")
        z3_cls_preds0 = tensorr("../para_682/head.cls_preds.0.zero_point.npy")
        bias_f_cls_preds0 = tensorr("../para_682/head.cls_preds.0.bias.npy")
        self.bias_int_cls_preds0 = bias_f_cls_preds0
        coe_name = '../data1_coe/out_hand_cls_preds0.coe'
        self.cls_preds0 = Conv2d_Q(quant_scale1=s1_cls_preds0, quant_zero_point1=z1_cls_preds0,
                                   quant_scale2=s2_cls_preds0,
                                   quant_zero_point2=z2_cls_preds0, quant_scale3=s3_cls_preds0,
                                   quant_zero_point3=z3_cls_preds0, coe_name=coe_name)

        s1_reg_convs0_0 = tensorr("../para_682/head.stems.0.conv.scale.npy")
        z1_reg_convs0_0 = tensorr("../para_682/head.stems.0.conv.zero_point.npy")
        s2_reg_convs0_0 = tensorr("../para_682/head.reg_convs.0.0.conv.weight.scale.npy")
        z2_reg_convs0_0 = tensorr("../para_682/head.reg_convs.0.0.conv.weight.zero_point.npy")
        s3_reg_convs0_0 = tensorr("../para_682/head.reg_convs.0.0.conv.scale.npy")
        z3_reg_convs0_0 = tensorr("../para_682/head.reg_convs.0.0.conv.zero_point.npy")
        bias_f_reg_convs0_0 = tensorr("../para_682/head.reg_convs.0.0.conv.bias.npy")
        self.bias_int_reg_convs0_0 = bias_f_reg_convs0_0
        coe_name = '../data1_coe/out_hand_reg_convs0_0.coe'
        self.reg_convs0_0 = Conv2d_Q(quant_scale1=s1_reg_convs0_0, quant_zero_point1=z1_reg_convs0_0,
                                     quant_scale2=s2_reg_convs0_0,
                                     quant_zero_point2=z2_reg_convs0_0, quant_scale3=s3_reg_convs0_0,
                                     quant_zero_point3=z3_reg_convs0_0, coe_name=coe_name)

        s1_reg_convs0_1 = tensorr("../para_682/head.reg_convs.0.0.conv.scale.npy")
        z1_reg_convs0_1 = tensorr("../para_682/head.reg_convs.0.0.conv.zero_point.npy")
        s2_reg_convs0_1 = tensorr("../para_682/head.reg_convs.0.1.conv.weight.scale.npy")
        z2_reg_convs0_1 = tensorr("../para_682/head.reg_convs.0.1.conv.weight.zero_point.npy")
        s3_reg_convs0_1 = tensorr("../para_682/head.reg_convs.0.1.conv.scale.npy")
        z3_reg_convs0_1 = tensorr("../para_682/head.reg_convs.0.1.conv.zero_point.npy")
        bias_f_reg_convs0_1 = tensorr("../para_682/head.reg_convs.0.1.conv.bias.npy")
        self.bias_int_reg_convs0_1 = bias_f_reg_convs0_1
        coe_name = '../data1_coe/out_hand_reg_convs0_1.coe'
        self.reg_convs0_1 = Conv2d_Q(quant_scale1=s1_reg_convs0_1, quant_zero_point1=z1_reg_convs0_1,
                                     quant_scale2=s2_reg_convs0_1,
                                     quant_zero_point2=z2_reg_convs0_1, quant_scale3=s3_reg_convs0_1,
                                     quant_zero_point3=z3_reg_convs0_1, coe_name=coe_name)

        s1_reg_preds0 = tensorr("../para_682/head.reg_convs.0.1.conv.scale.npy")
        z1_reg_preds0 = tensorr("../para_682/head.reg_convs.0.1.conv.zero_point.npy")
        s2_reg_preds0 = tensorr("../para_682/head.reg_preds.0.weight.scale.npy")
        z2_reg_preds0 = tensorr("../para_682/head.reg_preds.0.weight.zero_point.npy")
        s3_reg_preds0 = tensorr("../para_682/head.reg_preds.0.scale.npy")
        z3_reg_preds0 = tensorr("../para_682/head.reg_preds.0.zero_point.npy")
        bias_f_reg_preds0 = tensorr("../para_682/head.reg_preds.0.bias.npy")
        self.bias_int_reg_preds0 = bias_f_reg_preds0
        coe_name = '../data1_coe/out_hand_reg_preds0.coe'
        self.reg_preds0 = Conv2d_Q(quant_scale1=s1_reg_preds0, quant_zero_point1=z1_reg_preds0,
                                   quant_scale2=s2_reg_preds0,
                                   quant_zero_point2=z2_reg_preds0, quant_scale3=s3_reg_preds0,
                                   quant_zero_point3=z3_reg_preds0, coe_name=coe_name)

        s1_obj_preds0 = tensorr("../para_682/head.reg_convs.0.1.conv.scale.npy")
        z1_obj_preds0 = tensorr("../para_682/head.reg_convs.0.1.conv.zero_point.npy")
        s2_obj_preds0 = tensorr("../para_682/head.obj_preds.0.weight.scale.npy")
        z2_obj_preds0 = tensorr("../para_682/head.obj_preds.0.weight.zero_point.npy")
        s3_obj_preds0 = tensorr("../para_682/head.obj_preds.0.scale.npy")
        z3_obj_preds0 = tensorr("../para_682/head.obj_preds.0.zero_point.npy")
        bias_f_obj_preds0 = tensorr("../para_682/head.obj_preds.0.bias.npy")
        self.bias_int_obj_preds0 = bias_f_obj_preds0
        coe_name = '../data1_coe/out_hand_obj_preds0.coe'
        self.obj_preds0 = Conv2d_Q(quant_scale1=s1_obj_preds0, quant_zero_point1=z1_obj_preds0,
                                   quant_scale2=s2_obj_preds0,
                                   quant_zero_point2=z2_obj_preds0, quant_scale3=s3_obj_preds0,
                                   quant_zero_point3=z3_obj_preds0, coe_name=coe_name)
        # ==================csp6========================               
        s1_csp6_cat0_0 = tensorr("../para_682/head.reg_preds.0.scale.npy")
        z1_csp6_cat0_0 = tensorr("../para_682/head.reg_preds.0.zero_point.npy")
        s2_csp6_cat0_0 = tensorr("../para_682/head.obj_preds.0.scale.npy")
        z2_csp6_cat0_0 = tensorr("../para_682/head.obj_preds.0.zero_point.npy")
        s3_csp6_cat0_0 = tensorr("../para_682/head.csp6.scale.npy")
        z3_csp6_cat0_0 = tensorr("../para_682/head.csp6.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp6_cat0_0.coe'
        self.csp6_cat0_0 = Conv2d_Q(quant_scale1=s1_csp6_cat0_0, quant_zero_point1=z1_csp6_cat0_0,
                                    quant_scale2=s2_csp6_cat0_0,
                                    quant_zero_point2=z2_csp6_cat0_0, quant_scale3=s3_csp6_cat0_0,
                                    quant_zero_point3=z3_csp6_cat0_0, coe_name=coe_name)

        s1_csp6_cat0_1 = tensorr("../para_682/head.csp6.scale.npy")
        z1_csp6_cat0_1 = tensorr("../para_682/head.csp6.zero_point.npy")
        s2_csp6_cat0_1 = tensorr("../para_682/head.cls_preds.0.scale.npy")
        z2_csp6_cat0_1 = tensorr("../para_682/head.cls_preds.0.zero_point.npy")
        s3_csp6_cat0_1 = tensorr("../para_682/head.csp6.scale.npy")
        z3_csp6_cat0_1 = tensorr("../para_682/head.csp6.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp6_cat0_1.coe'
        self.csp6_cat0_1 = Conv2d_Q(quant_scale1=s1_csp6_cat0_1, quant_zero_point1=z1_csp6_cat0_1,
                                    quant_scale2=s2_csp6_cat0_1,
                                    quant_zero_point2=z2_csp6_cat0_1, quant_scale3=s3_csp6_cat0_1,
                                    quant_zero_point3=z3_csp6_cat0_1, coe_name=coe_name)
        # *================================output_p4===================================================
        s1_stems1_conv = tensorr("../para_682/backbone.C3_n3.conv3.conv.scale.npy")
        z1_stems1_conv = tensorr("../para_682/backbone.C3_n3.conv3.conv.zero_point.npy")
        s2_stems1_conv = tensorr("../para_682/head.stems.1.conv.weight.scale.npy")
        z2_stems1_conv = tensorr("../para_682/head.stems.1.conv.weight.zero_point.npy")
        s3_stems1_conv = tensorr("../para_682/head.stems.1.conv.scale.npy")
        z3_stems1_conv = tensorr("../para_682/head.stems.1.conv.zero_point.npy")
        bias_f_stems1_conv = tensorr("../para_682/head.stems.1.conv.bias.npy")
        self.bias_int_stems1_conv = bias_f_stems1_conv
        coe_name = '../data1_coe/out_hand_stems1_conv.coe'
        self.stems1_conv = Conv2d_Q(quant_scale1=s1_stems1_conv, quant_zero_point1=z1_stems1_conv,
                                    quant_scale2=s2_stems1_conv,
                                    quant_zero_point2=z2_stems1_conv, quant_scale3=s3_stems1_conv,
                                    quant_zero_point3=z3_stems1_conv, coe_name=coe_name)

        s1_cls_convs1_0 = tensorr("../para_682/head.stems.1.conv.scale.npy")
        z1_cls_convs1_0 = tensorr("../para_682/head.stems.1.conv.zero_point.npy")
        s2_cls_convs1_0 = tensorr("../para_682/head.cls_convs.1.0.conv.weight.scale.npy")
        z2_cls_convs1_0 = tensorr("../para_682/head.cls_convs.1.0.conv.weight.zero_point.npy")
        s3_cls_convs1_0 = tensorr("../para_682/head.cls_convs.1.0.conv.scale.npy")
        z3_cls_convs1_0 = tensorr("../para_682/head.cls_convs.1.0.conv.zero_point.npy")
        bias_f_cls_convs1_0 = tensorr("../para_682/head.cls_convs.1.0.conv.bias.npy")
        self.bias_int_cls_convs1_0 = bias_f_cls_convs1_0
        coe_name = '../data1_coe/out_hand_cls_convs1_0.coe'
        self.cls_convs1_0 = Conv2d_Q(quant_scale1=s1_cls_convs1_0, quant_zero_point1=z1_cls_convs1_0,
                                     quant_scale2=s2_cls_convs1_0,
                                     quant_zero_point2=z2_cls_convs1_0, quant_scale3=s3_cls_convs1_0,
                                     quant_zero_point3=z3_cls_convs1_0, coe_name=coe_name)

        s1_cls_convs1_1 = tensorr("../para_682/head.cls_convs.1.0.conv.scale.npy")
        z1_cls_convs1_1 = tensorr("../para_682/head.cls_convs.1.0.conv.zero_point.npy")
        s2_cls_convs1_1 = tensorr("../para_682/head.cls_convs.1.1.conv.weight.scale.npy")
        z2_cls_convs1_1 = tensorr("../para_682/head.cls_convs.1.1.conv.weight.zero_point.npy")
        s3_cls_convs1_1 = tensorr("../para_682/head.cls_convs.1.1.conv.scale.npy")
        z3_cls_convs1_1 = tensorr("../para_682/head.cls_convs.1.1.conv.zero_point.npy")
        bias_f_cls_convs1_1 = tensorr("../para_682/head.cls_convs.1.1.conv.bias.npy")
        self.bias_int_cls_convs1_1 = bias_f_cls_convs1_1
        coe_name = '../data1_coe/out_hand_cls_convs1_1.coe'
        self.cls_convs1_1 = Conv2d_Q(quant_scale1=s1_cls_convs1_1, quant_zero_point1=z1_cls_convs1_1,
                                     quant_scale2=s2_cls_convs1_1,
                                     quant_zero_point2=z2_cls_convs1_1, quant_scale3=s3_cls_convs1_1,
                                     quant_zero_point3=z3_cls_convs1_1, coe_name=coe_name)

        s1_cls_preds1 = tensorr("../para_682/head.cls_convs.1.1.conv.scale.npy")
        z1_cls_preds1 = tensorr("../para_682/head.cls_convs.1.1.conv.zero_point.npy")
        s2_cls_preds1 = tensorr("../para_682/head.cls_preds.1.weight.scale.npy")
        z2_cls_preds1 = tensorr("../para_682/head.cls_preds.1.weight.zero_point.npy")
        s3_cls_preds1 = tensorr("../para_682/head.cls_preds.1.scale.npy")
        z3_cls_preds1 = tensorr("../para_682/head.cls_preds.1.zero_point.npy")
        bias_f_cls_preds1 = tensorr("../para_682/head.cls_preds.1.bias.npy")
        self.bias_int_cls_preds1 = bias_f_cls_preds1
        coe_name = '../data1_coe/out_hand_cls_preds1.coe'
        self.cls_preds1 = Conv2d_Q(quant_scale1=s1_cls_preds1, quant_zero_point1=z1_cls_preds1,
                                   quant_scale2=s2_cls_preds1,
                                   quant_zero_point2=z2_cls_preds1, quant_scale3=s3_cls_preds1,
                                   quant_zero_point3=z3_cls_preds1, coe_name=coe_name)

        s1_reg_convs1_0 = tensorr("../para_682/head.stems.1.conv.scale.npy")
        z1_reg_convs1_0 = tensorr("../para_682/head.stems.1.conv.zero_point.npy")
        s2_reg_convs1_0 = tensorr("../para_682/head.reg_convs.1.0.conv.weight.scale.npy")
        z2_reg_convs1_0 = tensorr("../para_682/head.reg_convs.1.0.conv.weight.zero_point.npy")
        s3_reg_convs1_0 = tensorr("../para_682/head.reg_convs.1.0.conv.scale.npy")
        z3_reg_convs1_0 = tensorr("../para_682/head.reg_convs.1.0.conv.zero_point.npy")
        bias_f_reg_convs1_0 = tensorr("../para_682/head.reg_convs.1.0.conv.bias.npy")
        self.bias_int_reg_convs1_0 = bias_f_reg_convs1_0
        coe_name = '../data1_coe/out_hand_reg_convs1_0.coe'
        self.reg_convs1_0 = Conv2d_Q(quant_scale1=s1_reg_convs1_0, quant_zero_point1=z1_reg_convs1_0,
                                     quant_scale2=s2_reg_convs1_0,
                                     quant_zero_point2=z2_reg_convs1_0, quant_scale3=s3_reg_convs1_0,
                                     quant_zero_point3=z3_reg_convs1_0, coe_name=coe_name)

        s1_reg_convs1_1 = tensorr("../para_682/head.reg_convs.1.0.conv.scale.npy")
        z1_reg_convs1_1 = tensorr("../para_682/head.reg_convs.1.0.conv.zero_point.npy")
        s2_reg_convs1_1 = tensorr("../para_682/head.reg_convs.1.1.conv.weight.scale.npy")
        z2_reg_convs1_1 = tensorr("../para_682/head.reg_convs.1.1.conv.weight.zero_point.npy")
        s3_reg_convs1_1 = tensorr("../para_682/head.reg_convs.1.1.conv.scale.npy")
        z3_reg_convs1_1 = tensorr("../para_682/head.reg_convs.1.1.conv.zero_point.npy")
        bias_f_reg_convs1_1 = tensorr("../para_682/head.reg_convs.1.1.conv.bias.npy")
        self.bias_int_reg_convs1_1 = bias_f_reg_convs1_1
        coe_name = '../data1_coe/out_hand_reg_convs1_1.coe'
        self.reg_convs1_1 = Conv2d_Q(quant_scale1=s1_reg_convs1_1, quant_zero_point1=z1_reg_convs1_1,
                                     quant_scale2=s2_reg_convs1_1,
                                     quant_zero_point2=z2_reg_convs1_1, quant_scale3=s3_reg_convs1_1,
                                     quant_zero_point3=z3_reg_convs1_1, coe_name=coe_name)

        s1_reg_preds1 = tensorr("../para_682/head.reg_convs.1.1.conv.scale.npy")
        z1_reg_preds1 = tensorr("../para_682/head.reg_convs.1.1.conv.zero_point.npy")
        s2_reg_preds1 = tensorr("../para_682/head.reg_preds.1.weight.scale.npy")
        z2_reg_preds1 = tensorr("../para_682/head.reg_preds.1.weight.zero_point.npy")
        s3_reg_preds1 = tensorr("../para_682/head.reg_preds.1.scale.npy")
        z3_reg_preds1 = tensorr("../para_682/head.reg_preds.1.zero_point.npy")
        bias_f_reg_preds1 = tensorr("../para_682/head.reg_preds.1.bias.npy")
        self.bias_int_reg_preds1 = bias_f_reg_preds1
        coe_name = '../data1_coe/out_hand_reg_preds1.coe'
        self.reg_preds1 = Conv2d_Q(quant_scale1=s1_reg_preds1, quant_zero_point1=z1_reg_preds1,
                                   quant_scale2=s2_reg_preds1,
                                   quant_zero_point2=z2_reg_preds1, quant_scale3=s3_reg_preds1,
                                   quant_zero_point3=z3_reg_preds1, coe_name=coe_name)

        s1_obj_preds1 = tensorr("../para_682/head.reg_convs.1.1.conv.scale.npy")
        z1_obj_preds1 = tensorr("../para_682/head.reg_convs.1.1.conv.zero_point.npy")
        s2_obj_preds1 = tensorr("../para_682/head.obj_preds.1.weight.scale.npy")
        z2_obj_preds1 = tensorr("../para_682/head.obj_preds.1.weight.zero_point.npy")
        s3_obj_preds1 = tensorr("../para_682/head.obj_preds.1.scale.npy")
        z3_obj_preds1 = tensorr("../para_682/head.obj_preds.1.zero_point.npy")
        bias_f_obj_preds1 = tensorr("../para_682/head.obj_preds.1.bias.npy")
        self.bias_int_obj_preds1 = bias_f_obj_preds1
        coe_name = '../data1_coe/out_hand_obj_preds1.coe'
        self.obj_preds1 = Conv2d_Q(quant_scale1=s1_obj_preds1, quant_zero_point1=z1_obj_preds1,
                                   quant_scale2=s2_obj_preds1,
                                   quant_zero_point2=z2_obj_preds1, quant_scale3=s3_obj_preds1,
                                   quant_zero_point3=z3_obj_preds1, coe_name=coe_name)
        # ==================csp6========================               
        s1_csp6_cat1_0 = tensorr("../para_682/head.reg_preds.1.scale.npy")
        z1_csp6_cat1_0 = tensorr("../para_682/head.reg_preds.1.zero_point.npy")
        s2_csp6_cat1_0 = tensorr("../para_682/head.obj_preds.1.scale.npy")
        z2_csp6_cat1_0 = tensorr("../para_682/head.obj_preds.1.zero_point.npy")
        s3_csp6_cat1_0 = tensorr("../para_682/head.csp6.scale.npy")
        z3_csp6_cat1_0 = tensorr("../para_682/head.csp6.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp6_cat1_1.coe'
        self.csp6_cat1_0 = Conv2d_Q(quant_scale1=s1_csp6_cat1_0, quant_zero_point1=z1_csp6_cat1_0,
                                    quant_scale2=s2_csp6_cat1_0,
                                    quant_zero_point2=z2_csp6_cat1_0, quant_scale3=s3_csp6_cat1_0,
                                    quant_zero_point3=z3_csp6_cat1_0, coe_name=coe_name)

        s1_csp6_cat1_1 = tensorr("../para_682/head.csp6.scale.npy")
        z1_csp6_cat1_1 = tensorr("../para_682/head.csp6.zero_point.npy")
        s2_csp6_cat1_1 = tensorr("../para_682/head.cls_preds.1.scale.npy")
        z2_csp6_cat1_1 = tensorr("../para_682/head.cls_preds.1.zero_point.npy")
        s3_csp6_cat1_1 = tensorr("../para_682/head.csp6.scale.npy")
        z3_csp6_cat1_1 = tensorr("../para_682/head.csp6.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp6_cat1_1.coe'
        self.csp6_cat1_1 = Conv2d_Q(quant_scale1=s1_csp6_cat1_1, quant_zero_point1=z1_csp6_cat1_1,
                                    quant_scale2=s2_csp6_cat1_1,
                                    quant_zero_point2=z2_csp6_cat1_1, quant_scale3=s3_csp6_cat1_1,
                                    quant_zero_point3=z3_csp6_cat1_1, coe_name=coe_name)
        # *================================output_p5===================================================
        s1_stems2_conv = tensorr("../para_682/backbone.C3_n4.conv3.conv.scale.npy")
        z1_stems2_conv = tensorr("../para_682/backbone.C3_n4.conv3.conv.zero_point.npy")
        s2_stems2_conv = tensorr("../para_682/head.stems.2.conv.weight.scale.npy")
        z2_stems2_conv = tensorr("../para_682/head.stems.2.conv.weight.zero_point.npy")
        s3_stems2_conv = tensorr("../para_682/head.stems.2.conv.scale.npy")
        z3_stems2_conv = tensorr("../para_682/head.stems.2.conv.zero_point.npy")
        bias_f_stems2_conv = tensorr("../para_682/head.stems.2.conv.bias.npy")
        self.bias_int_stems2_conv = bias_f_stems2_conv
        coe_name = '../data1_coe/out_hand_stems2_conv.coe'
        self.stems2_conv = Conv2d_Q(quant_scale1=s1_stems2_conv, quant_zero_point1=z1_stems2_conv,
                                    quant_scale2=s2_stems2_conv,
                                    quant_zero_point2=z2_stems2_conv, quant_scale3=s3_stems2_conv,
                                    quant_zero_point3=z3_stems2_conv, coe_name=coe_name)

        s1_cls_convs2_0 = tensorr("../para_682/head.stems.2.conv.scale.npy")
        z1_cls_convs2_0 = tensorr("../para_682/head.stems.2.conv.zero_point.npy")
        s2_cls_convs2_0 = tensorr("../para_682/head.cls_convs.2.0.conv.weight.scale.npy")
        z2_cls_convs2_0 = tensorr("../para_682/head.cls_convs.2.0.conv.weight.zero_point.npy")
        s3_cls_convs2_0 = tensorr("../para_682/head.cls_convs.2.0.conv.scale.npy")
        z3_cls_convs2_0 = tensorr("../para_682/head.cls_convs.2.0.conv.zero_point.npy")
        bias_f_cls_convs2_0 = tensorr("../para_682/head.cls_convs.2.0.conv.bias.npy")
        self.bias_int_cls_convs2_0 = bias_f_cls_convs2_0
        coe_name = '../data1_coe/out_hand_cls_convs2_0.coe'
        self.cls_convs2_0 = Conv2d_Q(quant_scale1=s1_cls_convs2_0, quant_zero_point1=z1_cls_convs2_0,
                                     quant_scale2=s2_cls_convs2_0,
                                     quant_zero_point2=z2_cls_convs2_0, quant_scale3=s3_cls_convs2_0,
                                     quant_zero_point3=z3_cls_convs2_0, coe_name=coe_name)

        s1_cls_convs2_1 = tensorr("../para_682/head.cls_convs.2.0.conv.scale.npy")
        z1_cls_convs2_1 = tensorr("../para_682/head.cls_convs.2.0.conv.zero_point.npy")
        s2_cls_convs2_1 = tensorr("../para_682/head.cls_convs.2.1.conv.weight.scale.npy")
        z2_cls_convs2_1 = tensorr("../para_682/head.cls_convs.2.1.conv.weight.zero_point.npy")
        s3_cls_convs2_1 = tensorr("../para_682/head.cls_convs.2.1.conv.scale.npy")
        z3_cls_convs2_1 = tensorr("../para_682/head.cls_convs.2.1.conv.zero_point.npy")
        bias_f_cls_convs2_1 = tensorr("../para_682/head.cls_convs.2.1.conv.bias.npy")
        self.bias_int_cls_convs2_1 = bias_f_cls_convs2_1
        coe_name = '../data1_coe/out_hand_cls_convs2_1.coe'
        self.cls_convs2_1 = Conv2d_Q(quant_scale1=s1_cls_convs2_1, quant_zero_point1=z1_cls_convs2_1,
                                     quant_scale2=s2_cls_convs2_1,
                                     quant_zero_point2=z2_cls_convs2_1, quant_scale3=s3_cls_convs2_1,
                                     quant_zero_point3=z3_cls_convs2_1, coe_name=coe_name)

        s1_cls_preds2 = tensorr("../para_682/head.cls_convs.2.1.conv.scale.npy")
        z1_cls_preds2 = tensorr("../para_682/head.cls_convs.2.1.conv.zero_point.npy")
        s2_cls_preds2 = tensorr("../para_682/head.cls_preds.2.weight.scale.npy")
        z2_cls_preds2 = tensorr("../para_682/head.cls_preds.2.weight.zero_point.npy")
        s3_cls_preds2 = tensorr("../para_682/head.cls_preds.2.scale.npy")
        z3_cls_preds2 = tensorr("../para_682/head.cls_preds.2.zero_point.npy")
        bias_f_cls_preds2 = tensorr("../para_682/head.cls_preds.2.bias.npy")
        self.bias_int_cls_preds2 = bias_f_cls_preds2
        coe_name = '../data1_coe/out_hand_cls_preds2.coe'
        self.cls_preds2 = Conv2d_Q(quant_scale1=s1_cls_preds2, quant_zero_point1=z1_cls_preds2,
                                   quant_scale2=s2_cls_preds2,
                                   quant_zero_point2=z2_cls_preds2, quant_scale3=s3_cls_preds2,
                                   quant_zero_point3=z3_cls_preds2, coe_name=coe_name)

        s1_reg_convs2_0 = tensorr("../para_682/head.stems.2.conv.scale.npy")
        z1_reg_convs2_0 = tensorr("../para_682/head.stems.2.conv.zero_point.npy")
        s2_reg_convs2_0 = tensorr("../para_682/head.reg_convs.2.0.conv.weight.scale.npy")
        z2_reg_convs2_0 = tensorr("../para_682/head.reg_convs.2.0.conv.weight.zero_point.npy")
        s3_reg_convs2_0 = tensorr("../para_682/head.reg_convs.2.0.conv.scale.npy")
        z3_reg_convs2_0 = tensorr("../para_682/head.reg_convs.2.0.conv.zero_point.npy")
        bias_f_reg_convs2_0 = tensorr("../para_682/head.reg_convs.2.0.conv.bias.npy")
        self.bias_int_reg_convs2_0 = bias_f_reg_convs2_0
        coe_name = '../data1_coe/out_hand_reg_convs2_0.coe'
        self.reg_convs2_0 = Conv2d_Q(quant_scale1=s1_reg_convs2_0, quant_zero_point1=z1_reg_convs2_0,
                                     quant_scale2=s2_reg_convs2_0,
                                     quant_zero_point2=z2_reg_convs2_0, quant_scale3=s3_reg_convs2_0,
                                     quant_zero_point3=z3_reg_convs2_0, coe_name=coe_name)

        s1_reg_convs2_1 = tensorr("../para_682/head.reg_convs.2.0.conv.scale.npy")
        z1_reg_convs2_1 = tensorr("../para_682/head.reg_convs.2.0.conv.zero_point.npy")
        s2_reg_convs2_1 = tensorr("../para_682/head.reg_convs.2.1.conv.weight.scale.npy")
        z2_reg_convs2_1 = tensorr("../para_682/head.reg_convs.2.1.conv.weight.zero_point.npy")
        s3_reg_convs2_1 = tensorr("../para_682/head.reg_convs.2.1.conv.scale.npy")
        z3_reg_convs2_1 = tensorr("../para_682/head.reg_convs.2.1.conv.zero_point.npy")
        bias_f_reg_convs2_1 = tensorr("../para_682/head.reg_convs.2.1.conv.bias.npy")
        self.bias_int_reg_convs2_1 = bias_f_reg_convs2_1
        coe_name = '../data1_coe/out_hand_reg_convs2_1.coe'
        self.reg_convs2_1 = Conv2d_Q(quant_scale1=s1_reg_convs2_1, quant_zero_point1=z1_reg_convs2_1,
                                     quant_scale2=s2_reg_convs2_1,
                                     quant_zero_point2=z2_reg_convs2_1, quant_scale3=s3_reg_convs2_1,
                                     quant_zero_point3=z3_reg_convs2_1, coe_name=coe_name)

        s1_reg_preds2 = tensorr("../para_682/head.reg_convs.2.1.conv.scale.npy")
        z1_reg_preds2 = tensorr("../para_682/head.reg_convs.2.1.conv.zero_point.npy")
        s2_reg_preds2 = tensorr("../para_682/head.reg_preds.2.weight.scale.npy")
        z2_reg_preds2 = tensorr("../para_682/head.reg_preds.2.weight.zero_point.npy")
        s3_reg_preds2 = tensorr("../para_682/head.reg_preds.2.scale.npy")
        z3_reg_preds2 = tensorr("../para_682/head.reg_preds.2.zero_point.npy")
        bias_f_reg_preds2 = tensorr("../para_682/head.reg_preds.2.bias.npy")
        self.bias_int_reg_preds2 = bias_f_reg_preds2
        coe_name = '../data1_coe/out_hand_reg_preds2.coe'
        self.reg_preds2 = Conv2d_Q(quant_scale1=s1_reg_preds2, quant_zero_point1=z1_reg_preds2,
                                   quant_scale2=s2_reg_preds2,
                                   quant_zero_point2=z2_reg_preds2, quant_scale3=s3_reg_preds2,
                                   quant_zero_point3=z3_reg_preds2, coe_name=coe_name)

        s1_obj_preds2 = tensorr("../para_682/head.reg_convs.2.1.conv.scale.npy")
        z1_obj_preds2 = tensorr("../para_682/head.reg_convs.2.1.conv.zero_point.npy")
        s2_obj_preds2 = tensorr("../para_682/head.obj_preds.2.weight.scale.npy")
        z2_obj_preds2 = tensorr("../para_682/head.obj_preds.2.weight.zero_point.npy")
        s3_obj_preds2 = tensorr("../para_682/head.obj_preds.2.scale.npy")
        z3_obj_preds2 = tensorr("../para_682/head.obj_preds.2.zero_point.npy")
        bias_f_obj_preds2 = tensorr("../para_682/head.obj_preds.2.bias.npy")
        self.bias_int_obj_preds2 = bias_f_obj_preds2
        coe_name = '../data1_coe/out_hand_obj_preds2.coe'
        self.obj_preds2 = Conv2d_Q(quant_scale1=s1_obj_preds2, quant_zero_point1=z1_obj_preds2,
                                   quant_scale2=s2_obj_preds2,
                                   quant_zero_point2=z2_obj_preds2, quant_scale3=s3_obj_preds2,
                                   quant_zero_point3=z3_obj_preds2, coe_name=coe_name)
        # ==================csp6========================               
        s1_csp6_cat2_0 = tensorr("../para_682/head.reg_preds.2.scale.npy")
        z1_csp6_cat2_0 = tensorr("../para_682/head.reg_preds.2.zero_point.npy")
        s2_csp6_cat2_0 = tensorr("../para_682/head.obj_preds.2.scale.npy")
        z2_csp6_cat2_0 = tensorr("../para_682/head.obj_preds.2.zero_point.npy")
        s3_csp6_cat2_0 = tensorr("../para_682/head.csp6.scale.npy")
        z3_csp6_cat2_0 = tensorr("../para_682/head.csp6.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp6_cat2_0.coe'
        self.csp6_cat2_0 = Conv2d_Q(quant_scale1=s1_csp6_cat2_0, quant_zero_point1=z1_csp6_cat2_0,
                                    quant_scale2=s2_csp6_cat2_0,
                                    quant_zero_point2=z2_csp6_cat2_0, quant_scale3=s3_csp6_cat2_0,
                                    quant_zero_point3=z3_csp6_cat2_0, coe_name=coe_name)

        s1_csp6_cat2_1 = tensorr("../para_682/head.csp6.scale.npy")
        z1_csp6_cat2_1 = tensorr("../para_682/head.csp6.zero_point.npy")
        s2_csp6_cat2_1 = tensorr("../para_682/head.cls_preds.2.scale.npy")
        z2_csp6_cat2_1 = tensorr("../para_682/head.cls_preds.2.zero_point.npy")
        s3_csp6_cat2_1 = tensorr("../para_682/head.csp6.scale.npy")
        z3_csp6_cat2_1 = tensorr("../para_682/head.csp6.zero_point.npy")
        coe_name = '../data1_coe/out_hand_csp6_cat2_1.coe'
        self.csp6_cat2_1 = Conv2d_Q(quant_scale1=s1_csp6_cat2_1, quant_zero_point1=z1_csp6_cat2_1,
                                    quant_scale2=s2_csp6_cat2_1,
                                    quant_zero_point2=z2_csp6_cat2_1, quant_scale3=s3_csp6_cat2_1,
                                    quant_zero_point3=z3_csp6_cat2_1, coe_name=coe_name)

    def forward(self, x):
        model = torch.jit.load('../yolox_quant_pth_682/Epoch2-yolox_quantization_post.pth')
        model.eval()
        # model = YoloBody(1,"s")
        # model.eval()

        # np.random.seed(1)
        # feature = np.random.random((1,3, 640, 640))

        # feature = get_picture()
        feature = picture_load("../img/001.jpg")

        # feature = feature/255
        # feature = feature.astype(np.float32)
        # feature = torch.from_numpy(feature)
        with torch.no_grad():
            # 1 1 640 640
            # 3 640 640
            quant = model.quant(feature)
            # 1 1 640 640 -> 1 4 320 320
            # 3 640 640 -> 12 320 320
            Focus_out = Focus(quant)

            # 1 4 320 320 -> 1 32 320 320
            # 3 640 640  ——> 32 320 320
            x = model.backbone.backbone.stem(quant)
            stem_out = x
            # * ====================dark2==============================
            dark2_0 = list(model.backbone.backbone.dark2.children())[0]
            dark2_1 = list(model.backbone.backbone.dark2.children())[1]
            # 1 32 320 320 -> 1 64 160 160
            # 32 320 320 -> 64 160 160
            x = dark2_0(x)
            dark2_0_conv_out = x
            # 64 160 160 -> 32 160 160  1x1
            # 64 160 160 -> 32 160 160  1x1
            x_1 = dark2_1.conv1(x)
            dark2_1_conv1_out = x_1
            # 64 160 160 -> 32 160 160  1x1
            # 64 160 160 -> 32 160 160  1x1
            x_2 = dark2_1.conv2(x)
            dark2_1_conv2_out = x_2

            dark2_m0 = list(dark2_1.m.children())[0]
            # 32 160 160 -> 32 160 160  1x1
            # 32 160 160 -> 32 160 160  1x1
            x = dark2_m0.conv1(x_1)
            dark2_1_m0_conv1_out = x
            # 32 160 160 -> 32 160 160
            # 32 160 160 -> 32 160 160
            x = dark2_m0.conv2(x)
            dark2_1_m0_conv2_out = x
            # 32 160 160 add 32 160 160 -> 32 160 160
            # 32 160 160 add 32 160 160 -> 32 160 160
            y = dark2_m0.csp.add(x, x_1)
            dark2_1_m0_add_out = y
            # 32 160 160 + 32 160 160-> 64 160 160
            # 32 160 160 + 32 160 160-> 64 160 160
            x = dark2_1.csp1.cat([y, x_2], dim=1)
            dark2_1_cat_out = x
            # 64 160 160 -> 64 160 160  1x1
            # 64 160 160 -> 64 160 160
            x = dark2_1.conv3(x)
            dark2_1_conv3_out = x
            # *=======================dark3==========================
            dark3_0 = list(model.backbone.backbone.dark3.children())[0]
            dark3_1 = list(model.backbone.backbone.dark3.children())[1]
            # 64 160 160 -> 128 80 80
            x = dark3_0(x)
            dark3_0_conv_out = x
            x_1 = dark3_1.conv1(x)
            dark3_1_conv1_out = x_1
            x_2 = dark3_1.conv2(x)
            dark3_1_conv2_out = x_2

            dark3_m0 = list(dark3_1.m.children())[0]
            x = dark3_m0.conv1(x_1)
            dark3_1_m0_conv1_out = x
            x = dark3_m0.conv2(x)
            dark3_1_m0_conv2_out = x
            y = dark3_m0.csp.add(x, x_1)
            dark3_1_m0_add_out = y

            dark3_m1 = list(dark3_1.m.children())[1]
            x = dark3_m1.conv1(y)
            dark3_1_m1_conv1_out = x
            x = dark3_m1.conv2(x)
            dark3_1_m1_conv2_out = x
            y = dark3_m1.csp.add(x, y)
            dark3_1_m1_add_out = y

            dark3_m2 = list(dark3_1.m.children())[2]
            x = dark3_m2.conv1(y)
            dark3_1_m2_conv1_out = x
            x = dark3_m2.conv2(x)
            dark3_1_m2_conv2_out = x
            y = dark3_m2.csp.add(x, y)
            dark3_1_m2_add_out = y

            x = dark3_1.csp1.cat([y, x_2], dim=1)
            dark3_1_cat_out = x
            x = dark3_1.conv3(x)
            dark3_1_conv3_out = x
            # *===================dark4================================
            dark4_0 = list(model.backbone.backbone.dark4.children())[0]
            dark4_1 = list(model.backbone.backbone.dark4.children())[1]

            x = dark4_0(x)
            dark4_0_conv_out = x
            x_1 = dark4_1.conv1(x)
            dark4_1_conv1_out = x_1
            x_2 = dark4_1.conv2(x)
            dark4_1_conv2_out = x_2

            dark4_m0 = list(dark4_1.m.children())[0]
            x = dark4_m0.conv1(x_1)
            dark4_1_m0_conv1_out = x
            x = dark4_m0.conv2(x)
            dark4_1_m0_conv2_out = x
            y = dark4_m0.csp.add(x, x_1)
            dark4_1_m0_add_out = y

            dark4_m1 = list(dark4_1.m.children())[1]
            x = dark4_m1.conv1(y)
            dark4_1_m1_conv1_out = x
            x = dark4_m1.conv2(x)
            dark4_1_m1_conv2_out = x
            y = dark4_m1.csp.add(x, y)
            dark4_1_m1_add_out = y

            dark4_m2 = list(dark4_1.m.children())[2]
            x = dark4_m2.conv1(y)
            dark4_1_m2_conv1_out = x
            x = dark4_m2.conv2(x)
            dark4_1_m2_conv2_out = x
            y = dark4_m2.csp.add(x, y)
            dark4_1_m2_add_out = y

            x = dark4_1.csp1.cat([y, x_2], dim=1)
            dark4_1_cat_out = x
            x = dark4_1.conv3(x)
            dark4_1_conv3_out = x
            # *===================dark5================================
            dark5_0 = list(model.backbone.backbone.dark5.children())[0]
            dark5_1 = list(model.backbone.backbone.dark5.children())[1]

            x = dark5_0(x)
            dark5_0_conv_out = x
            x_1 = dark5_1.conv1(x)
            dark5_1_conv1_out = x_1
            x_2 = dark5_1.conv2(x)
            dark5_1_conv2_out = x_2

            dark5_m0 = list(dark5_1.m.children())[0]
            x = dark5_m0.conv1(x_1)
            dark5_1_m0_conv1_out = x
            x = dark5_m0.conv2(x)
            dark5_1_m0_conv2_out = x
            # y = dark5_m0.csp.add(x,x_1)
            dark5_1_m0_add_out = y

            x = dark5_1.csp1.cat([x, x_2], dim=1)
            dark5_1_cat_out = x
            x = dark5_1.conv3(x)
            dark5_1_conv3_out = x
            # *======================FPN================================
            [feat1, feat2, feat3] = [dark3_1_conv3_out, dark4_1_conv3_out, dark5_1_conv3_out]
            P5 = model.backbone.lateral_conv0(feat3)
            lateral_conv0_out = P5

            P5_upsample = model.backbone.upsample(P5)
            p5_upsample_out = P5_upsample
            P5_upsample = model.backbone.csp2.cat([P5_upsample, feat2], 1)
            p5_upsample_cat_out = P5_upsample
            # *==================C3_p4===========================
            x_1 = model.backbone.C3_p4.conv1(P5_upsample)
            C3_p4_conv1_out = x_1
            x_2 = model.backbone.C3_p4.conv2(P5_upsample)
            C3_p4_conv2_out = x_2

            C3_p4_m0 = list(model.backbone.C3_p4.m.children())[0]
            x = C3_p4_m0.conv1(x_1)
            C3_p4_m0_conv1_out = x
            x = C3_p4_m0.conv2(x)
            C3_p4_m0_conv2_out = x
            # y = C3_p4_m0.csp.add(x,x_1)
            C3_p4_m0_add_out = x
            x = model.backbone.C3_p4.csp1.cat([x, x_2], dim=1)
            C3_p4_cat_out = x
            x = model.backbone.C3_p4.conv3(x)
            C3_p4_conv3_out = x
            # *==================================================
            P4 = model.backbone.reduce_conv1(x)
            reduce_conv1_out = P4

            P4_upsample = model.backbone.upsample(P4)
            p4_upsample_out = P4_upsample
            P4_upsample = model.backbone.csp3.cat([P4_upsample, feat1], 1)
            p4_upsample_cat_out = P4_upsample
            # *====================C3_p3========================
            x_1 = model.backbone.C3_p3.conv1(P4_upsample)
            C3_p3_conv1_out = x_1
            x_2 = model.backbone.C3_p3.conv2(P4_upsample)
            C3_p3_conv2_out = x_2

            C3_p3_m0 = list(model.backbone.C3_p3.m.children())[0]
            x = C3_p3_m0.conv1(x_1)
            C3_p3_m0_conv1_out = x
            x = C3_p3_m0.conv2(x)
            C3_p3_m0_conv2_out = x
            # y = C3_p3_m0.csp.add(x,x_1)
            C3_p3_m0_add_out = x
            x = model.backbone.C3_p3.csp1.cat([x, x_2], dim=1)
            C3_p3_cat_out = x
            x = model.backbone.C3_p3.conv3(x)
            P3_out = x
            # *==================================================
            P3_downsample = model.backbone.bu_conv2(x)
            bu_conv2_out = P3_downsample
            P3_downsample = model.backbone.csp4.cat([P3_downsample, P4], 1)
            P3_downsample_cat_out = P3_downsample
            # *====================C3_n3==========================
            x_1 = model.backbone.C3_n3.conv1(P3_downsample)
            C3_n3_conv1_out = x_1
            x_2 = model.backbone.C3_n3.conv2(P3_downsample)
            C3_n3_conv2_out = x_2

            C3_n3_m0 = list(model.backbone.C3_n3.m.children())[0]
            x = C3_n3_m0.conv1(x_1)
            C3_n3_m0_conv1_out = x
            x = C3_n3_m0.conv2(x)
            C3_n3_m0_conv2_out = x
            # y = C3_n3_m0.csp.add(x,x_1)
            C3_n3_m0_add_out = x
            x = model.backbone.C3_n3.csp1.cat([x, x_2], dim=1)
            C3_n3_cat_out = x
            x = model.backbone.C3_n3.conv3(x)
            P4_out = x
            # *=========================================================
            P4_downsample = model.backbone.bu_conv1(x)
            bu_conv1_out = P4_downsample
            #################################################################
            P4_downsample = model.backbone.csp5.cat([P4_downsample, P5], 1)
            P4_downsample_cat_out = P4_downsample
            # *=======================C3_n4============================
            x_1 = model.backbone.C3_n4.conv1(P4_downsample)
            C3_n4_conv1_out = x_1
            x_2 = model.backbone.C3_n4.conv2(P4_downsample)
            C3_n4_conv2_out = x_2

            C3_n4_m0 = list(model.backbone.C3_n4.m.children())[0]
            x = C3_n4_m0.conv1(x_1)
            C3_n4_m0_conv1_out = x
            x = C3_n4_m0.conv2(x)
            C3_n4_m0_conv2_out = x
            # y = C3_n4_m0.csp.add(x,x_1)
            C3_n4_m0_add_out = x
            x = model.backbone.C3_n4.csp1.cat([x, x_2], dim=1)
            C3_n4_cat_out = x
            x = model.backbone.C3_n4.conv3(x)
            P5_out = x
            # *====================head=======================================
            stem0 = list(model.head.stems.children())[0]
            stem1 = list(model.head.stems.children())[1]
            stem2 = list(model.head.stems.children())[2]

            cls_convs0 = list(model.head.cls_convs.children())[0]
            cls_convs1 = list(model.head.cls_convs.children())[1]
            cls_convs2 = list(model.head.cls_convs.children())[2]
            cls_convs0_0 = list(cls_convs0.children())[0]
            cls_convs0_1 = list(cls_convs0.children())[1]
            cls_convs1_0 = list(cls_convs1.children())[0]
            cls_convs1_1 = list(cls_convs1.children())[1]
            cls_convs2_0 = list(cls_convs2.children())[0]
            cls_convs2_1 = list(cls_convs2.children())[1]

            cls_preds0 = list(model.head.cls_preds.children())[0]
            cls_preds1 = list(model.head.cls_preds.children())[1]
            cls_preds2 = list(model.head.cls_preds.children())[2]

            reg_convs0 = list(model.head.reg_convs.children())[0]
            reg_convs1 = list(model.head.reg_convs.children())[1]
            reg_convs2 = list(model.head.reg_convs.children())[2]
            reg_convs0_0 = list(reg_convs0.children())[0]
            reg_convs0_1 = list(reg_convs0.children())[1]
            reg_convs1_0 = list(reg_convs1.children())[0]
            reg_convs1_1 = list(reg_convs1.children())[1]
            reg_convs2_0 = list(reg_convs2.children())[0]
            reg_convs2_1 = list(reg_convs2.children())[1]

            reg_preds0 = list(model.head.reg_preds.children())[0]
            reg_preds1 = list(model.head.reg_preds.children())[1]
            reg_preds2 = list(model.head.reg_preds.children())[2]

            obj_preds0 = list(model.head.obj_preds.children())[0]
            obj_preds1 = list(model.head.obj_preds.children())[1]
            obj_preds2 = list(model.head.obj_preds.children())[2]

            x = stem0(P3_out)
            stem0_out = x
            cls_feat = cls_convs0_0(x)
            cls_convs0_0_out = cls_feat
            cls_feat = cls_convs0_1(cls_feat)
            cls_convs0_1_out = cls_feat
            cls_output_P3 = cls_preds0(cls_feat)
            cls_preds0_out = cls_output_P3
            reg_feat = reg_convs0_0(x)
            reg_convs0_0_out = reg_feat
            reg_feat = reg_convs0_1(reg_feat)
            reg_convs0_1_out = reg_feat
            reg_output_P3 = reg_preds0(reg_feat)
            reg_preds0_out = reg_output_P3
            obj_output_P3 = obj_preds0(reg_feat)
            obj_preds0_out = obj_output_P3
            output_p3 = model.head.csp6.cat([reg_output_P3, obj_output_P3, cls_output_P3], 1)

            x = stem1(P4_out)
            stem1_out = x
            cls_feat = cls_convs1_0(x)
            cls_convs1_0_out = cls_feat
            cls_feat = cls_convs1_1(cls_feat)
            cls_convs1_1_out = cls_feat
            cls_output_P4 = cls_preds1(cls_feat)
            cls_preds1_out = cls_output_P4
            reg_feat = reg_convs1_0(x)
            reg_convs1_0_out = reg_feat
            reg_feat = reg_convs1_1(reg_feat)
            reg_convs1_1_out = reg_feat
            reg_output_P4 = reg_preds1(reg_feat)
            reg_preds1_out = reg_output_P4
            obj_output_P4 = obj_preds1(reg_feat)
            obj_preds1_out = obj_output_P4
            output_p4 = model.head.csp6.cat([reg_output_P4, obj_output_P4, cls_output_P4], 1)

            x = stem2(P5_out)
            stem2_out = x
            cls_feat = cls_convs2_0(x)
            cls_convs2_0_out = cls_feat
            cls_feat = cls_convs2_1(cls_feat)
            cls_convs2_1_out = cls_feat
            cls_output_P5 = cls_preds2(cls_feat)
            cls_preds2_out = cls_output_P5
            reg_feat = reg_convs2_0(x)
            reg_convs2_0_out = reg_feat
            reg_feat = reg_convs2_1(reg_feat)
            reg_convs2_1_out = reg_feat
            reg_output_P5 = reg_preds2(reg_feat)
            reg_preds2_out = reg_output_P5
            obj_output_P5 = obj_preds2(reg_feat)
            obj_preds2_out = obj_output_P5
            output_p5 = model.head.csp6.cat([reg_output_P5, obj_output_P5, cls_output_P5], 1)
            outputs = [output_p3, output_p4, output_p5]

        path1 = 'biasscaleshift1114.bin'
        file_name = '../ins/yolox_ins_64_682.dat'
        weight_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.weight.int.npy')
        weight_stem_conv_f = tensorr('../para_682/backbone.backbone.stem.conv.conv.weight.npy')
        weight_address = 1879048192  # 权重地址 初始HEX:7000 0000  下一层用上一层返回的地址 #1,879,046,752
        # computer_address = 16777216 #计算地址 1,879,046,752
        computer_address = 0  # 计算地址 1,879,046,752
        # 方便dsp读取将第一层的地址用P5_13_conv11返回的地址，++++第二个int(1023410176)<->int(0)+++++
        # operator == 'image_final' #1886455872
        # int(1879046752), int(0)
        # ==========================
        # 32 12 320 320 ->32 16 320 320
        weight_stem_conv_0 = weight_4to8(weight_stem_conv)
        # ==========================
        #  x[0] : 下一次要读的权重地址
        #  x[1] ：这次写入卷积结果的开始地址
        x = self.stem_conv(weight_address, computer_address, Focus_out, weight_stem_conv_f, weight_stem_conv_0,
                           self.bias_int_stem_conv, path1, stride=1, padding=1, block=0, operator='conv33')
        # exit()
        # ?===============================dark2=========================================================
        weight_dark2_0_conv = tensorr('../para_682/backbone.backbone.dark2.0.conv.weight.int.npy')
        weight_dark2_0_conv_f = tensorr('../para_682/backbone.backbone.dark2.0.conv.weight.npy')
        x = self.dark2_0_conv(x[0], x[1], stem_out, weight_dark2_0_conv_f, weight_dark2_0_conv,
                              self.bias_int_dark2_0_conv, path1, stride=2, padding=1, block=0, operator='conv33')
        # exit()
        weight_dark2_1_conv1 = tensorr('../para_682/backbone.backbone.dark2.1.conv1.conv.weight.int.npy')
        weight_dark2_1_conv1_f = tensorr('../para_682/backbone.backbone.dark2.1.conv1.conv.weight.npy')
        x_1 = self.dark2_1_conv1(x[0], x[1], dark2_0_conv_out, weight_dark2_1_conv1_f, weight_dark2_1_conv1,
                                 self.bias_int_dark2_1_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_dark2_1_conv2 = tensorr('../para_682/backbone.backbone.dark2.1.conv2.conv.weight.int.npy')
        weight_dark2_1_conv2_f = tensorr('../para_682/backbone.backbone.dark2.1.conv2.conv.weight.npy')
        x_2 = self.dark2_1_conv2(x_1[0], x[1], dark2_0_conv_out, weight_dark2_1_conv2_f, weight_dark2_1_conv2,
                                 self.bias_int_dark2_1_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                                 index=2)
        # exit()
        weight_dark2_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.weight.int.npy')
        weight_dark2_1_m0_conv1_f = tensorr('../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.weight.npy')
        x = self.dark2_1_m0_conv1(x_2[0], x_1[1], dark2_1_conv1_out, weight_dark2_1_m0_conv1_f, weight_dark2_1_m0_conv1,
                                  self.bias_int_dark2_1_m0_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11', index=2)
        # exit()
        weight_dark2_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.weight.int.npy')
        weight_dark2_1_m0_conv2_f = tensorr('../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.weight.npy')
        y = self.dark2_1_m0_conv2(x[0], x[1], dark2_1_m0_conv1_out, weight_dark2_1_m0_conv2_f, weight_dark2_1_m0_conv2,
                                  self.bias_int_dark2_1_m0_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        # exit()
        # * conaca和add都没用到权重地址，即weight_address，只需要计算结果的写地址
        x_1 = self.dark2_1_m0_add(y[0], y[1], dark2_1_m0_conv2_out, weight_dark2_1_m0_conv2_f, weight_dark2_1_m0_conv2,
                                  self.bias_int_dark2_1_m0_conv2,
                                  path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1],
                                  cat2=dark2_1_conv1_out, operator='add')
        # exit()
        x = self.dark2_1_cat(x_1[0], x_1[1], dark2_1_m0_add_out, weight_dark2_1_m0_conv2_f, weight_dark2_1_m0_conv2,
                             self.bias_int_dark2_1_m0_conv2,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1], cat2=dark2_1_conv2_out,
                             operator='concat')
        # exit()
        weight_dark2_1_conv3 = tensorr('../para_682/backbone.backbone.dark2.1.conv3.conv.weight.int.npy')
        weight_dark2_1_conv3_f = tensorr('../para_682/backbone.backbone.dark2.1.conv3.conv.weight.npy')
        x = self.dark2_1_conv3(x[0], x[1], dark2_1_cat_out, weight_dark2_1_conv3_f, weight_dark2_1_conv3,
                               self.bias_int_dark2_1_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        # ?=========================================dark3================================================================
        weight_dark3_0_conv = tensorr('../para_682/backbone.backbone.dark3.0.conv.weight.int.npy')
        weight_dark3_0_conv_f = tensorr('../para_682/backbone.backbone.dark3.0.conv.weight.npy')
        x = self.dark3_0_conv(x[0], x[1], dark2_1_conv3_out, weight_dark3_0_conv_f, weight_dark3_0_conv,
                              self.bias_int_dark3_0_conv, path1, stride=2, padding=1, block=0, operator='conv33')
        # exit()
        weight_dark3_1_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.conv1.conv.weight.int.npy')
        weight_dark3_1_conv1_f = tensorr('../para_682/backbone.backbone.dark3.1.conv1.conv.weight.npy')
        x_1 = self.dark3_1_conv1(x[0], x[1], dark3_0_conv_out, weight_dark3_1_conv1_f, weight_dark3_1_conv1,
                                 self.bias_int_dark3_1_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_dark3_1_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.conv2.conv.weight.int.npy')
        weight_dark3_1_conv2_f = tensorr('../para_682/backbone.backbone.dark3.1.conv2.conv.weight.npy')
        x_2 = self.dark3_1_conv2(x_1[0], x[1], dark3_0_conv_out, weight_dark3_1_conv2_f, weight_dark3_1_conv2,
                                 self.bias_int_dark3_1_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                                 index=2)
        # exit()
        weight_dark3_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.weight.int.npy')
        weight_dark3_1_m0_conv1_f = tensorr('../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.weight.npy')
        x = self.dark3_1_m0_conv1(x_2[0], x_1[1], dark3_1_conv1_out, weight_dark3_1_m0_conv1_f, weight_dark3_1_m0_conv1,
                                  self.bias_int_dark3_1_m0_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11', index=2)
        # exit()
        weight_dark3_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.weight.int.npy')
        weight_dark3_1_m0_conv2_f = tensorr('../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.weight.npy')
        y = self.dark3_1_m0_conv2(x[0], x[1], dark3_1_m0_conv1_out, weight_dark3_1_m0_conv2_f, weight_dark3_1_m0_conv2,
                                  self.bias_int_dark3_1_m0_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        # exit()
        x_1 = self.dark3_1_m0_add(y[0], y[1], dark3_1_m0_conv2_out, weight_dark3_1_m0_conv2_f,
                                  weight_dark3_1_m0_conv2, self.bias_int_dark3_1_m0_conv2,
                                  path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1],
                                  cat2=dark3_1_conv1_out, operator='add')
        # exit()
        weight_dark3_1_m1_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.weight.int.npy')
        weight_dark3_1_m1_conv1_f = tensorr('../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.weight.npy')
        x = self.dark3_1_m1_conv1(x_1[0], x_1[1], dark3_1_m0_add_out, weight_dark3_1_m1_conv1_f,
                                  weight_dark3_1_m1_conv1,
                                  self.bias_int_dark3_1_m1_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11')
        # exit()
        weight_dark3_1_m1_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.weight.int.npy')
        weight_dark3_1_m1_conv2_f = tensorr('../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.weight.npy')
        y = self.dark3_1_m1_conv2(x[0], x[1], dark3_1_m1_conv1_out, weight_dark3_1_m1_conv2_f, weight_dark3_1_m1_conv2,
                                  self.bias_int_dark3_1_m1_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        # exit()
        x_1 = self.dark3_1_m1_add(y[0], y[1], dark3_1_m1_conv2_out, weight_dark3_1_m1_conv2_f,
                                  weight_dark3_1_m1_conv2, self.bias_int_dark3_1_m1_conv2,
                                  path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1],
                                  cat2=dark3_1_m0_add_out, operator='add')

        weight_dark3_1_m2_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.weight.int.npy')
        weight_dark3_1_m2_conv1_f = tensorr('../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.weight.npy')
        x = self.dark3_1_m2_conv1(x_1[0], x_1[1], dark3_1_m1_add_out, weight_dark3_1_m2_conv1_f,
                                  weight_dark3_1_m2_conv1,
                                  self.bias_int_dark3_1_m2_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11')
        # exit()
        weight_dark3_1_m2_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.weight.int.npy')
        weight_dark3_1_m2_conv2_f = tensorr('../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.weight.npy')
        y = self.dark3_1_m2_conv2(x[0], x[1], dark3_1_m2_conv1_out, weight_dark3_1_m2_conv2_f, weight_dark3_1_m2_conv2,
                                  self.bias_int_dark3_1_m2_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        # exit()
        x_1 = self.dark3_1_m2_add(y[0], y[1], dark3_1_m2_conv2_out, weight_dark3_1_m2_conv2_f,
                                  weight_dark3_1_m2_conv2, self.bias_int_dark3_1_m2_conv2,
                                  path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1],
                                  cat2=dark3_1_m1_add_out, operator='add')
        # exit()
        x = self.dark3_1_cat(x_1[0], x_1[1], dark3_1_m2_add_out, weight_dark3_1_m2_conv2_f,
                             weight_dark3_1_m2_conv2, self.bias_int_dark3_1_m2_conv2,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1],
                             cat2=dark3_1_conv2_out, operator='concat')
        # exit()
        weight_dark3_1_conv3 = tensorr('../para_682/backbone.backbone.dark3.1.conv3.conv.weight.int.npy')
        weight_dark3_1_conv3_f = tensorr('../para_682/backbone.backbone.dark3.1.conv3.conv.weight.npy')
        x = self.dark3_1_conv3(x[0], x[1], dark3_1_cat_out, weight_dark3_1_conv3_f, weight_dark3_1_conv3,
                               self.bias_int_dark3_1_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        feat1 = x
        # exit()
        # ?======================================dark4====================================================
        weight_dark4_0_conv = tensorr('../para_682/backbone.backbone.dark4.0.conv.weight.int.npy')
        weight_dark4_0_conv_f = tensorr('../para_682/backbone.backbone.dark4.0.conv.weight.npy')
        x = self.dark4_0_conv(x[0], x[1], dark3_1_conv3_out, weight_dark4_0_conv_f, weight_dark4_0_conv,
                              self.bias_int_dark4_0_conv, path1, stride=2, padding=1, block=0, operator='conv33')
        # exit()
        weight_dark4_1_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.conv1.conv.weight.int.npy')
        weight_dark4_1_conv1_f = tensorr('../para_682/backbone.backbone.dark4.1.conv1.conv.weight.npy')
        x_1 = self.dark4_1_conv1(x[0], x[1], dark4_0_conv_out, weight_dark4_1_conv1_f, weight_dark4_1_conv1,
                                 self.bias_int_dark4_1_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_dark4_1_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.conv2.conv.weight.int.npy')
        weight_dark4_1_conv2_f = tensorr('../para_682/backbone.backbone.dark4.1.conv2.conv.weight.npy')
        x_2 = self.dark4_1_conv2(x_1[0], x[1], dark4_0_conv_out, weight_dark4_1_conv2_f, weight_dark4_1_conv2,
                                 self.bias_int_dark4_1_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                                 index=2)
        # exit()
        weight_dark4_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.weight.int.npy')
        weight_dark4_1_m0_conv1_f = tensorr('../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.weight.npy')
        x = self.dark4_1_m0_conv1(x_2[0], x_1[1], dark4_1_conv1_out, weight_dark4_1_m0_conv1_f, weight_dark4_1_m0_conv1,
                                  self.bias_int_dark4_1_m0_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11', index=2)
        # exit()
        weight_dark4_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.weight.int.npy')
        weight_dark4_1_m0_conv2_f = tensorr('../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.weight.npy')
        y = self.dark4_1_m0_conv2(x[0], x[1], dark4_1_m0_conv1_out, weight_dark4_1_m0_conv2_f, weight_dark4_1_m0_conv2,
                                  self.bias_int_dark4_1_m0_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        # exit()
        x_1 = self.dark4_1_m0_add(y[0], y[1], dark4_1_m0_conv2_out, weight_dark4_1_m0_conv2_f,
                                  weight_dark4_1_m0_conv2, self.bias_int_dark4_1_m0_conv2,
                                  path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1],
                                  cat2=dark4_1_conv1_out, operator='add')

        weight_dark4_1_m1_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.weight.int.npy')
        weight_dark4_1_m1_conv1_f = tensorr('../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.weight.npy')
        x = self.dark4_1_m1_conv1(x_1[0], x_1[1], dark4_1_m0_add_out, weight_dark4_1_m1_conv1_f,
                                  weight_dark4_1_m1_conv1,
                                  self.bias_int_dark4_1_m1_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11')
        # exit()
        weight_dark4_1_m1_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.weight.int.npy')
        weight_dark4_1_m1_conv2_f = tensorr('../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.weight.npy')
        y = self.dark4_1_m1_conv2(x[0], x[1], dark4_1_m1_conv1_out, weight_dark4_1_m1_conv2_f, weight_dark4_1_m1_conv2,
                                  self.bias_int_dark4_1_m1_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        # exit()
        x_1 = self.dark4_1_m1_add(y[0], y[1], dark4_1_m1_conv2_out, weight_dark4_1_m1_conv2_f,
                                  weight_dark4_1_m1_conv2, self.bias_int_dark4_1_m1_conv2,
                                  path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1],
                                  cat2=dark4_1_m0_add_out, operator='add')

        weight_dark4_1_m2_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.weight.int.npy')
        weight_dark4_1_m2_conv1_f = tensorr('../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.weight.npy')
        x = self.dark4_1_m2_conv1(x_1[0], x_1[1], dark4_1_m1_add_out, weight_dark4_1_m2_conv1_f,
                                  weight_dark4_1_m2_conv1,
                                  self.bias_int_dark4_1_m2_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11')
        # exit()
        weight_dark4_1_m2_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.weight.int.npy')
        weight_dark4_1_m2_conv2_f = tensorr('../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.weight.npy')
        y = self.dark4_1_m2_conv2(x[0], x[1], dark4_1_m2_conv1_out, weight_dark4_1_m2_conv2_f, weight_dark4_1_m2_conv2,
                                  self.bias_int_dark4_1_m2_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        add_y = x
        # exit()
        x_1 = self.dark4_1_m2_add(y[0], y[1], dark4_1_m2_conv2_out, weight_dark4_1_m2_conv2_f,
                                  weight_dark4_1_m2_conv2, self.bias_int_dark4_1_m2_conv2,
                                  path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1],
                                  cat2=dark4_1_m1_add_out, operator='add')
        # exit()
        x = self.dark4_1_cat(x_1[0], x_1[1], dark4_1_m2_add_out, weight_dark4_1_m2_conv2_f,
                             weight_dark4_1_m2_conv2, self.bias_int_dark4_1_m2_conv2,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1],
                             cat2=dark4_1_conv2_out, operator='concat')
        # exit()
        weight_dark4_1_conv3 = tensorr('../para_682/backbone.backbone.dark4.1.conv3.conv.weight.int.npy')
        weight_dark4_1_conv3_f = tensorr('../para_682/backbone.backbone.dark4.1.conv3.conv.weight.npy')
        x = self.dark4_1_conv3(x[0], x[1], dark4_1_cat_out, weight_dark4_1_conv3_f, weight_dark4_1_conv3,
                               self.bias_int_dark4_1_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        feat2 = x
        # ?=============================dark5==================================
        weight_dark5_0_conv = tensorr('../para_682/backbone.backbone.dark5.0.conv.weight.int.npy')
        weight_dark5_0_conv_f = tensorr('../para_682/backbone.backbone.dark5.0.conv.weight.npy')
        x = self.dark5_0_conv(x[0], x[1], dark4_1_conv3_out, weight_dark5_0_conv_f, weight_dark5_0_conv,
                              self.bias_int_dark5_0_conv, path1, stride=2, padding=1, block=2, operator='conv33')
        # exit()
        weight_dark5_1_conv1 = tensorr('../para_682/backbone.backbone.dark5.1.conv1.conv.weight.int.npy')
        weight_dark5_1_conv1_f = tensorr('../para_682/backbone.backbone.dark5.1.conv1.conv.weight.npy')
        x_1 = self.dark5_1_conv1(x[0], x[1], dark5_0_conv_out, weight_dark5_1_conv1_f, weight_dark5_1_conv1,
                                 self.bias_int_dark5_1_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_dark5_1_conv2 = tensorr('../para_682/backbone.backbone.dark5.1.conv2.conv.weight.int.npy')
        weight_dark5_1_conv2_f = tensorr('../para_682/backbone.backbone.dark5.1.conv2.conv.weight.npy')
        x_2 = self.dark5_1_conv2(x_1[0], x[1], dark5_0_conv_out, weight_dark5_1_conv2_f, weight_dark5_1_conv2,
                                 self.bias_int_dark5_1_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                                 index=2)
        cat_x_2 = x
        # exit()
        weight_dark5_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.weight.int.npy')
        weight_dark5_1_m0_conv1_f = tensorr('../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.weight.npy')
        x = self.dark5_1_m0_conv1(x_2[0], x_1[1], dark5_1_conv1_out, weight_dark5_1_m0_conv1_f, weight_dark5_1_m0_conv1,
                                  self.bias_int_dark5_1_m0_conv1, path1, stride=1, padding=0, block=0,
                                  operator='conv11', index=2)
        # exit()
        weight_dark5_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.weight.int.npy')
        weight_dark5_1_m0_conv2_f = tensorr('../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.weight.npy')
        y = self.dark5_1_m0_conv2(x[0], x[1], dark5_1_m0_conv1_out, weight_dark5_1_m0_conv2_f, weight_dark5_1_m0_conv2,
                                  self.bias_int_dark5_1_m0_conv2, path1, stride=1, padding=1, block=0,
                                  operator='conv33')
        x_1 = y
        # exit()
        # x_1 = self.dark5_1_m0_add(y[0], y[1], dark5_1_m0_conv2_out, weight_dark5_1_m0_conv2_f, weight_dark5_1_m0_conv2, self.bias_int_dark5_1_m0_conv2,
        #                   path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1], cat2=dark5_1_conv1_out,operator='concat_add')
        # exit()
        x = self.dark5_1_cat(x_1[0], x_1[1], dark5_1_m0_conv2_out, weight_dark5_1_m0_conv2_f, weight_dark5_1_m0_conv2,
                             self.bias_int_dark5_1_m0_conv2,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1], cat2=dark5_1_conv2_out,
                             operator='concat')
        # exit()
        weight_dark5_1_conv3 = tensorr('../para_682/backbone.backbone.dark5.1.conv3.conv.weight.int.npy')
        weight_dark5_1_conv3_f = tensorr('../para_682/backbone.backbone.dark5.1.conv3.conv.weight.npy')
        x = self.dark5_1_conv3(x[0], x[1], dark5_1_cat_out, weight_dark5_1_conv3_f, weight_dark5_1_conv3,
                               self.bias_int_dark5_1_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        feat3 = x
        # exit()
        # ?===============================FPN==========================================================
        weight_lateral_conv0 = tensorr('../para_682/backbone.lateral_conv0.conv.weight.int.npy')
        weight_lateral_conv0_f = tensorr('../para_682/backbone.lateral_conv0.conv.weight.npy')
        x = self.lateral_conv0(x[0], x[1], dark5_1_conv3_out, weight_lateral_conv0_f, weight_lateral_conv0,
                               self.bias_int_lateral_conv0, path1, stride=1, padding=0, block=0, operator='conv11')
        lateral_conv0 = x
        print("================第一个reshape==============================")
        x = reshape(x[0], x[1], lateral_conv0_out, operator="upsample", filename=file_name)
        # exit()
        print("p5_upsample_out", p5_upsample_out.shape)
        print("dark4_1_conv3_out", dark4_1_conv3_out.shape)
        x = self.csp2_cat(x[0], x[1], p5_upsample_out, weight_lateral_conv0_f, weight_lateral_conv0,
                          self.bias_int_lateral_conv0,
                          path1, stride=1, padding=1, block=0, cat2_weight_address=feat2[1], cat2=dark4_1_conv3_out,
                          operator='concat')
        # exit()
        # ?================================c3p4========================================================
        weight_C3_p4_conv1 = tensorr('../para_682/backbone.C3_p4.conv1.conv.weight.int.npy')
        weight_C3_p4_conv1_f = tensorr('../para_682/backbone.C3_p4.conv1.conv.weight.npy')
        x_1 = self.C3_p4_conv1(x[0], x[1], p5_upsample_cat_out, weight_C3_p4_conv1_f, weight_C3_p4_conv1,
                               self.bias_int_C3_p4_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_C3_p4_conv2 = tensorr('../para_682/backbone.C3_p4.conv2.conv.weight.int.npy')
        weight_C3_p4_conv2_f = tensorr('../para_682/backbone.C3_p4.conv2.conv.weight.npy')
        x_2 = self.C3_p4_conv2(x_1[0], x[1], p5_upsample_cat_out, weight_C3_p4_conv2_f, weight_C3_p4_conv2,
                               self.bias_int_C3_p4_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                               index=2)
        # exit()
        weight_C3_p4_m0_conv1 = tensorr('../para_682/backbone.C3_p4.m.0.conv1.conv.weight.int.npy')
        weight_C3_p4_m0_conv1_f = tensorr('../para_682/backbone.C3_p4.m.0.conv1.conv.weight.npy')
        x = self.C3_p4_m0_conv1(x_2[0], x_1[1], C3_p4_conv1_out, weight_C3_p4_m0_conv1_f, weight_C3_p4_m0_conv1,
                                self.bias_int_C3_p4_m0_conv1, path1, stride=1, padding=0, block=0, operator='conv11',
                                index=2)
        # exit()
        weight_C3_p4_m0_conv2 = tensorr('../para_682/backbone.C3_p4.m.0.conv2.conv.weight.int.npy')
        weight_C3_p4_m0_conv2_f = tensorr('../para_682/backbone.C3_p4.m.0.conv2.conv.weight.npy')
        y = self.C3_p4_m0_conv2(x[0], x[1], C3_p4_m0_conv1_out, weight_C3_p4_m0_conv2_f, weight_C3_p4_m0_conv2,
                                self.bias_int_C3_p4_m0_conv2, path1, stride=1, padding=1, block=0, operator='conv33')
        x_1 = y
        # exit()
        # x_1 = self.C3_p4_m0_add(y[0], y[1], C3_p4_m0_conv2_out, weight_C3_p4_m0_conv2_f, weight_C3_p4_m0_conv2, self.bias_int_C3_p4_m0_conv2,
        #                   path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1], cat2=C3_p4_conv1_out,operator='concat_add')
        cat_x_1 = x
        # exit()
        x = self.C3_p4_csp1_cat(x_1[0], x_1[1], C3_p4_m0_conv2_out, C3_p4_conv2_out, weight_C3_p4_m0_conv2,
                                self.bias_int_C3_p4_m0_conv2,
                                path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1], cat2=C3_p4_conv2_out,
                                operator='concat')
        # exit()
        weight_C3_p4_conv3 = tensorr('../para_682/backbone.C3_p4.conv3.conv.weight.int.npy')
        weight_C3_p4_conv3_f = tensorr('../para_682/backbone.C3_p4.conv3.conv.weight.npy')
        x = self.C3_p4_conv3(x[0], x[1], C3_p4_cat_out, weight_C3_p4_conv3_f, weight_C3_p4_conv3,
                             self.bias_int_C3_p4_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        # ?=========================================================================================
        weight_reduce_conv1 = tensorr('../para_682/backbone.reduce_conv1.conv.weight.int.npy')
        weight_reduce_conv1_f = tensorr('../para_682/backbone.reduce_conv1.conv.weight.npy')
        x = self.reduce_conv1(x[0], x[1], C3_p4_conv3_out, weight_reduce_conv1_f, weight_reduce_conv1,
                              self.bias_int_reduce_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        reduce_conv1 = x
        x = reshape(x[0], x[1], reduce_conv1_out, operator="upsample", filename=file_name)
        # exit()
        x = self.csp3_cat(x[0], x[1], p4_upsample_out, weight_reduce_conv1_f, weight_reduce_conv1,
                          self.bias_int_reduce_conv1,
                          path1, stride=1, padding=1, block=0, cat2_weight_address=feat1[1], cat2=dark3_1_conv3_out,
                          operator='concat')
        # exit()
        # ?================================c3p3========================================================
        weight_C3_p3_conv1 = tensorr('../para_682/backbone.C3_p3.conv1.conv.weight.int.npy')
        weight_C3_p3_conv1_f = tensorr('../para_682/backbone.C3_p3.conv1.conv.weight.npy')
        x_1 = self.C3_p3_conv1(x[0], x[1], p4_upsample_cat_out, weight_C3_p3_conv1_f, weight_C3_p3_conv1,
                               self.bias_int_C3_p3_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_C3_p3_conv2 = tensorr('../para_682/backbone.C3_p3.conv2.conv.weight.int.npy')
        weight_C3_p3_conv2_f = tensorr('../para_682/backbone.C3_p3.conv2.conv.weight.npy')
        x_2 = self.C3_p3_conv2(x_1[0], x[1], p4_upsample_cat_out, weight_C3_p3_conv2_f, weight_C3_p3_conv2,
                               self.bias_int_C3_p4_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                               index=2)
        # exit()
        weight_C3_p3_m0_conv1 = tensorr('../para_682/backbone.C3_p3.m.0.conv1.conv.weight.int.npy')
        weight_C3_p3_m0_conv1_f = tensorr('../para_682/backbone.C3_p3.m.0.conv1.conv.weight.npy')
        x = self.C3_p3_m0_conv1(x_2[0], x_1[1], C3_p3_conv1_out, weight_C3_p3_m0_conv1_f, weight_C3_p3_m0_conv1,
                                self.bias_int_C3_p3_m0_conv1, path1, stride=1, padding=0, block=0, operator='conv11',
                                index=2)
        # exit()
        weight_C3_p3_m0_conv2 = tensorr('../para_682/backbone.C3_p3.m.0.conv2.conv.weight.int.npy')
        weight_C3_p3_m0_conv2_f = tensorr('../para_682/backbone.C3_p3.m.0.conv2.conv.weight.npy')
        y = self.C3_p3_m0_conv2(x[0], x[1], C3_p3_m0_conv1_out, weight_C3_p3_m0_conv2_f, weight_C3_p3_m0_conv2,
                                self.bias_int_C3_p3_m0_conv2, path1, stride=1, padding=1, block=0, operator='conv33')
        x_1 = y
        # exit()
        # x_1 = self.C3_p3_m0_add(y[0], y[1], C3_p3_m0_conv2_out, weight_C3_p3_m0_conv2_f, weight_C3_p3_m0_conv2, self.bias_int_C3_p3_m0_conv2,
        #                   path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1], cat2=C3_p3_conv1_out,operator='concat_add')
        # exit()
        x = self.C3_p3_csp1_cat(x_1[0], x_1[1], C3_p3_m0_conv2_out, C3_p3_conv2_out, weight_C3_p3_m0_conv2,
                                self.bias_int_C3_p3_m0_conv2,
                                path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1], cat2=C3_p3_conv2_out,
                                operator='concat')
        # exit()
        weight_C3_p3_conv3 = tensorr('../para_682/backbone.C3_p3.conv3.conv.weight.int.npy')
        weight_C3_p3_conv3_f = tensorr('../para_682/backbone.C3_p3.conv3.conv.weight.npy')
        x = self.C3_p3_conv3(x[0], x[1], C3_p3_cat_out, weight_C3_p3_conv3_f, weight_C3_p3_conv3,
                             self.bias_int_C3_p3_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        P3_out_address = x
        # ?=========================================================================================
        weight_bu_conv2 = tensorr('../para_682/backbone.bu_conv2.conv.weight.int.npy')
        weight_bu_conv2_f = tensorr('../para_682/backbone.bu_conv2.conv.weight.npy')
        x = self.bu_conv2(x[0], x[1], P3_out, weight_bu_conv2_f, weight_bu_conv2,
                          self.bias_int_bu_conv2, path1, stride=2, padding=1, block=0, operator='conv33')

        x = self.csp4_cat(x[0], x[1], bu_conv2_out, weight_bu_conv2_f, weight_bu_conv2, self.bias_int_bu_conv2,
                          path1, stride=1, padding=1, block=0, cat2_weight_address=reduce_conv1[1],
                          cat2=reduce_conv1_out, operator='concat')
        # exit()
        # ?================================C3_n3========================================================
        weight_C3_n3_conv1 = tensorr('../para_682/backbone.C3_n3.conv1.conv.weight.int.npy')
        weight_C3_n3_conv1_f = tensorr('../para_682/backbone.C3_n3.conv1.conv.weight.npy')
        x_1 = self.C3_n3_conv1(x[0], x[1], P3_downsample_cat_out, weight_C3_n3_conv1_f, weight_C3_n3_conv1,
                               self.bias_int_C3_n3_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_C3_n3_conv2 = tensorr('../para_682/backbone.C3_n3.conv2.conv.weight.int.npy')
        weight_C3_n3_conv2_f = tensorr('../para_682/backbone.C3_n3.conv2.conv.weight.npy')
        x_2 = self.C3_n3_conv2(x_1[0], x[1], P3_downsample_cat_out, weight_C3_n3_conv2_f, weight_C3_n3_conv2,
                               self.bias_int_C3_n3_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                               index=2)
        # exit()
        weight_C3_n3_m0_conv1 = tensorr('../para_682/backbone.C3_n3.m.0.conv1.conv.weight.int.npy')
        weight_C3_n3_m0_conv1_f = tensorr('../para_682/backbone.C3_n3.m.0.conv1.conv.weight.npy')
        x = self.C3_n3_m0_conv1(x_2[0], x_1[1], C3_n3_conv1_out, weight_C3_n3_m0_conv1_f, weight_C3_n3_m0_conv1,
                                self.bias_C3_n3_m0_conv1, path1, stride=1, padding=0, block=0, operator='conv11',
                                index=2)
        # exit()
        weight_C3_n3_m0_conv2 = tensorr('../para_682/backbone.C3_n3.m.0.conv2.conv.weight.int.npy')
        weight_C3_n3_m0_conv2_f = tensorr('../para_682/backbone.C3_n3.m.0.conv2.conv.weight.npy')
        y = self.C3_n3_m0_conv2(x[0], x[1], C3_n3_m0_conv1_out, weight_C3_n3_m0_conv2_f, weight_C3_n3_m0_conv2,
                                self.bias_int_C3_n3_m0_conv2, path1, stride=1, padding=1, block=0, operator='conv33')
        x_1 = y
        # exit()
        # x_1 = self.C3_n3_m0_add(y[0], y[1], C3_n3_m0_conv2_out, weight_C3_n3_m0_conv2_f, weight_C3_n3_m0_conv2, self.bias_int_C3_n3_m0_conv2,
        #                   path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1], cat2=C3_n3_conv1_out,operator='concat_add')
        cat_x_1 = x
        # exit()
        x = self.C3_n3_csp1_cat(x_1[0], x_1[1], C3_n3_m0_conv2_out, C3_n3_conv2_out, weight_C3_n3_m0_conv2,
                                self.bias_int_C3_n3_m0_conv2,
                                path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1], cat2=C3_n3_conv2_out,
                                operator='concat')
        # exit()
        weight_C3_n3_conv3 = tensorr('../para_682/backbone.C3_n3.conv3.conv.weight.int.npy')
        weight_C3_n3_conv3_f = tensorr('../para_682/backbone.C3_n3.conv3.conv.weight.npy')
        x = self.C3_n3_conv3(x[0], x[1], C3_n3_cat_out, weight_C3_n3_conv3_f, weight_C3_n3_conv3,
                             self.bias_int_C3_n3_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        P4_out_address = x
        # ?=========================================================================================
        weight_bu_conv1 = tensorr('../para_682/backbone.bu_conv1.conv.weight.int.npy')
        weight_bu_conv1_f = tensorr('../para_682/backbone.bu_conv1.conv.weight.npy')
        x = self.bu_conv1(x[0], x[1], P4_out, weight_bu_conv1_f, weight_bu_conv1,
                          self.bias_int_bu_conv1, path1, stride=2, padding=1, block=0, operator='conv33')

        x = self.csp5_cat(x[0], x[1], bu_conv1_out, weight_bu_conv1_f, weight_bu_conv1, self.bias_int_bu_conv1,
                          path1, stride=1, padding=1, block=0, cat2_weight_address=lateral_conv0[1],
                          cat2=lateral_conv0_out, operator='concat')
        # exit()
        # ?================================C3_n4========================================================
        weight_C3_n4_conv1 = tensorr('../para_682/backbone.C3_n4.conv1.conv.weight.int.npy')
        weight_C3_n4_conv1_f = tensorr('../para_682/backbone.C3_n4.conv1.conv.weight.npy')
        x_1 = self.C3_n4_conv1(x[0], x[1], P4_downsample_cat_out, weight_C3_n4_conv1_f, weight_C3_n4_conv1,
                               self.bias_int_C3_n4_conv1, path1, stride=1, padding=0, block=0, operator='conv11')
        # exit()
        weight_C3_n4_conv2 = tensorr('../para_682/backbone.C3_n4.conv2.conv.weight.int.npy')
        weight_C3_n4_conv2_f = tensorr('../para_682/backbone.C3_n4.conv2.conv.weight.npy')
        x_2 = self.C3_n4_conv2(x_1[0], x[1], P4_downsample_cat_out, weight_C3_n4_conv2_f, weight_C3_n4_conv2,
                               self.bias_int_C3_n4_conv2, path1, stride=1, padding=0, block=0, operator='conv11',
                               index=2)
        # exit()
        weight_C3_n4_m0_conv1 = tensorr('../para_682/backbone.C3_n4.m.0.conv1.conv.weight.int.npy')
        weight_C3_n4_m0_conv1_f = tensorr('../para_682/backbone.C3_n4.m.0.conv1.conv.weight.npy')
        x = self.C3_n4_m0_conv1(x_2[0], x_1[1], C3_n4_conv1_out, weight_C3_n4_m0_conv1_f, weight_C3_n4_m0_conv1,
                                self.bias_C3_n4_m0_conv1, path1, stride=1, padding=0, block=0, operator='conv11',
                                index=2)
        # exit()
        weight_C3_n4_m0_conv2 = tensorr('../para_682/backbone.C3_n4.m.0.conv2.conv.weight.int.npy')
        weight_C3_n4_m0_conv2_f = tensorr('../para_682/backbone.C3_n4.m.0.conv2.conv.weight.npy')
        y = self.C3_n4_m0_conv2(x[0], x[1], C3_n4_m0_conv1_out, weight_C3_n4_m0_conv2_f, weight_C3_n4_m0_conv2,
                                self.bias_int_C3_n4_m0_conv2, path1, stride=1, padding=1, block=0, operator='conv33')
        x_1 = y
        # exit()
        # x_1 = self.C3_n4_m0_add(y[0], y[1], C3_n4_m0_conv2_out, weight_C3_n4_m0_conv2_f, weight_C3_n4_m0_conv2, self.bias_int_C3_n4_m0_conv2,
        #                   path1, stride=1, padding=1, block=0, cat2_weight_address=x_1[1], cat2=C3_n4_conv1_out,operator='concat_add')
        # exit()
        x = self.C3_n4_csp1_cat(x_1[0], x_1[1], C3_n4_m0_conv2_out, weight_C3_n4_m0_conv2_f, weight_C3_n4_m0_conv2,
                                self.bias_int_C3_n4_m0_conv2,
                                path1, stride=1, padding=1, block=0, cat2_weight_address=x_2[1], cat2=C3_n4_conv2_out,
                                operator='concat')
        # exit()
        weight_C3_n4_conv3 = tensorr('../para_682/backbone.C3_n4.conv3.conv.weight.int.npy')
        weight_C3_n4_conv3_f = tensorr('../para_682/backbone.C3_n4.conv3.conv.weight.npy')
        x = self.C3_n4_conv3(x[0], x[1], C3_n4_cat_out, weight_C3_n4_conv3_f, weight_C3_n4_conv3,
                             self.bias_int_C3_n4_conv3, path1, stride=1, padding=0, block=0, operator='conv11')
        P5_out_address = x
        # ?====================================head=======================================

        weight_stems0_conv = tensorr('../para_682/head.stems.0.conv.weight.int.npy')
        weight_stems0_conv_f = tensorr('../para_682/head.stems.0.conv.weight.npy')
        x_1 = self.stems0_conv(x[0], P3_out_address[1], P3_out, weight_stems0_conv_f, weight_stems0_conv,
                               self.bias_int_stems0_conv, path1, stride=1, padding=0, block=0, operator='conv11',
                               index=17)
        # exit()
        # 1 128 80 80  128 128 3 3
        weight_cls_convs0_0 = tensorr('../para_682/head.cls_convs.0.0.conv.weight.int.npy')
        weight_cls_convs0_0_f = tensorr('../para_682/head.cls_convs.0.0.conv.weight.npy')
        x = self.cls_convs0_0(x_1[0], x_1[1], stem0_out, weight_cls_convs0_0_f, weight_cls_convs0_0,
                              self.bias_int_cls_convs0_0, path1, stride=1, padding=1, block=0, operator='conv33')

        weight_cls_convs0_1 = tensorr('../para_682/head.cls_convs.0.1.conv.weight.int.npy')
        weight_cls_convs0_1_f = tensorr('../para_682/head.cls_convs.0.1.conv.weight.npy')
        x = self.cls_convs0_1(x[0], x[1], cls_convs0_0_out, weight_cls_convs0_1_f, weight_cls_convs0_1,
                              self.bias_int_cls_convs0_1, path1, stride=1, padding=1, block=0, operator='conv33')
        # exit()
        # ==========================================================
        weight_cls_preds0 = tensorr('../para_682/head.cls_preds.0.weight.int.npy')
        weight_cls_preds0_f = tensorr('../para_682/head.cls_preds.0.weight.npy')
        # 1 128 80 80   1 128  1  1-> 8 128 1 1  ------ 1  8 80 80
        weight_cls_preds0 = weight_4to8(weight_cls_preds0)
        x = self.cls_preds0(x[0], x[1], cls_convs0_1_out, weight_cls_preds0_f, weight_cls_preds0,
                            self.bias_int_cls_preds0, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11')
        cls_output_P3_address = x
        # exit()
        # ================================================================

        weight_reg_convs0_0 = tensorr('../para_682/head.reg_convs.0.0.conv.weight.int.npy')
        weight_reg_convs0_0_f = tensorr('../para_682/head.reg_convs.0.0.conv.weight.npy')
        x = self.reg_convs0_0(x[0], x_1[1], stem0_out, weight_reg_convs0_0_f, weight_reg_convs0_0,
                              self.bias_int_reg_convs0_0, path1, stride=1, padding=1, block=0, operator='conv33',
                              index=4)
        # exit()
        weight_reg_convs0_1 = tensorr('../para_682/head.reg_convs.0.1.conv.weight.int.npy')
        weight_reg_convs0_1_f = tensorr('../para_682/head.reg_convs.0.1.conv.weight.npy')
        reg_feat = self.reg_convs0_1(x[0], x[1], reg_convs0_0_out, weight_reg_convs0_1_f, weight_reg_convs0_1,
                                     self.bias_int_reg_convs0_1, path1, stride=1, padding=1, block=0, operator='conv33')
        # exit()
        # ======================================================
        weight_reg_preds0 = tensorr('../para_682/head.reg_preds.0.weight.int.npy')
        weight_reg_preds0_f = tensorr('../para_682/head.reg_preds.0.weight.npy')
        # 1 128 80 80           4 128 1 1 -> 8 128 1 1  --------------------- 1 8  80 80
        weight_reg_preds0 = weight_4to8(weight_reg_preds0)
        x = self.reg_preds0(reg_feat[0], reg_feat[1], reg_convs0_1_out, weight_reg_preds0_f, weight_reg_preds0,
                            self.bias_int_reg_preds0, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11')
        # exit()
        reg_output_P3_address = x

        weight_obj_preds0 = tensorr('../para_682/head.obj_preds.0.weight.int.npy')
        weight_obj_preds0_f = tensorr('../para_682/head.obj_preds.0.weight.npy')
        weight_obj_preds0 = weight_4to8(weight_obj_preds0)
        # 1 128 80 80            1 128 1 1 -> 8 128 1 1 ------------------------ 1 8 80 80
        x = self.obj_preds0(x[0], reg_feat[1], reg_convs0_1_out, weight_obj_preds0_f, weight_obj_preds0,
                            self.bias_int_obj_preds0, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11', index=2)
        # exit()
        obj_output_P3_address = x
        temp = torch.ones((1, 8, 80, 80))
        # x = self.csp6_cat0_0(reg_output_P3_address[0], reg_output_P3_address[1], reg_preds0_out, 1, 1, 1,
        #                      path1, stride=1, padding=1, block=0, cat2_weight_address=obj_output_P3_address[1],
        #                      cat2=obj_preds0_out, operator='concat')
        # 写在cat1的地址之后  所以要index
        x = self.csp6_cat0_0(reg_output_P3_address[0], reg_output_P3_address[1], temp, 1, 1, 1,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=obj_output_P3_address[1],
                             cat2=temp, operator='concat', index=2)
        # exit()
        # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        csp6_cat0_0 = torch.ones((1, 16, 80, 80))
        x = self.csp6_cat0_1(x[0], x[1], csp6_cat0_0, 1, 1, 1,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=cls_output_P3_address[1],
                             cat2=temp, operator='concat')
        # exit()
        # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        # ===================================================================================
        # print(P4_out.shape)
        weight_stems1_conv = tensorr('../para_682/head.stems.1.conv.weight.int.npy')
        weight_stems1_conv_f = tensorr('../para_682/head.stems.1.conv.weight.npy')
        x_1 = self.stems1_conv(obj_output_P3_address[0], P4_out_address[1], P4_out, weight_stems1_conv_f,
                               weight_stems1_conv,
                               self.bias_int_stems1_conv, path1, stride=1, padding=0, block=0, operator='conv11',
                               index=19)
        # exit()
        # 1 128 40 40   128 128 3 3
        weight_cls_convs1_0 = tensorr('../para_682/head.cls_convs.1.0.conv.weight.int.npy')
        weight_cls_convs1_0_f = tensorr('../para_682/head.cls_convs.1.0.conv.weight.npy')
        x = self.cls_convs1_0(x_1[0], x_1[1], stem1_out, weight_cls_convs1_0_f, weight_cls_convs1_0,
                              self.bias_int_cls_convs1_0, path1, stride=1, padding=1, block=0, operator='conv33')
        # exit()
        weight_cls_convs1_1 = tensorr('../para_682/head.cls_convs.1.1.conv.weight.int.npy')
        weight_cls_convs1_1_f = tensorr('../para_682/head.cls_convs.1.1.conv.weight.npy')
        x = self.cls_convs1_1(x[0], x[1], cls_convs1_0_out, weight_cls_convs1_1_f, weight_cls_convs1_1,
                              self.bias_int_cls_convs1_1, path1, stride=1, padding=1, block=0, operator='conv33')
        # exit()
        weight_cls_preds1 = tensorr('../para_682/head.cls_preds.1.weight.int.npy')
        weight_cls_preds1_f = tensorr('../para_682/head.cls_preds.1.weight.npy')
        weight_cls_preds1 = weight_4to8(weight_cls_preds1)
        # 1 128 40 40      1 128  1  1-> 8 128 1 1  ------    1 8 40 40
        x = self.cls_preds1(x[0], x[1], cls_convs1_1_out, weight_cls_preds1_f, weight_cls_preds1,
                            self.bias_int_cls_preds1, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11')
        # exit()
        cls_output_P4_address = x

        weight_reg_convs1_0 = tensorr('../para_682/head.reg_convs.1.0.conv.weight.int.npy')
        weight_reg_convs1_0_f = tensorr('../para_682/head.reg_convs.1.0.conv.weight.npy')
        x = self.reg_convs1_0(x[0], x_1[1], stem1_out, weight_reg_convs1_0_f, weight_reg_convs1_0,
                              self.bias_int_reg_convs1_0, path1, stride=1, padding=1, block=0, operator='conv33',
                              index=4)

        weight_reg_convs1_1 = tensorr('../para_682/head.reg_convs.1.1.conv.weight.int.npy')
        weight_reg_convs1_1_f = tensorr('../para_682/head.reg_convs.1.1.conv.weight.npy')
        reg_feat = self.reg_convs1_1(x[0], x[1], reg_convs1_0_out, weight_reg_convs1_1_f, weight_reg_convs1_1,
                                     self.bias_int_reg_convs1_1, path1, stride=1, padding=1, block=0, operator='conv33')
        # exit()
        weight_reg_preds1 = tensorr('../para_682/head.reg_preds.1.weight.int.npy')
        weight_reg_preds1_f = tensorr('../para_682/head.reg_preds.1.weight.npy')
        weight_reg_preds1 = weight_4to8(weight_reg_preds1)
        # 1 128 40 40    4 128 1 1 -> 8 128 1 1  --------------------- 1 8 40 40
        x = self.reg_preds1(reg_feat[0], reg_feat[1], reg_convs1_1_out, weight_reg_preds1_f, weight_reg_preds1,
                            self.bias_int_reg_preds1, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11')
        reg_output_P4_address = x
        # 1 128 40 40   1 128 1 1 -> 8 128 1 1 ------------------------1 8 40 40
        weight_obj_preds1 = tensorr('../para_682/head.obj_preds.1.weight.int.npy')
        weight_obj_preds1_f = tensorr('../para_682/head.obj_preds.1.weight.npy')
        weight_obj_preds1 = weight_4to8(weight_obj_preds1)
        x = self.obj_preds1(x[0], reg_feat[1], reg_convs1_1_out, weight_obj_preds1_f, weight_obj_preds1,
                            self.bias_int_obj_preds1, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11', index=2)
        obj_output_P4_address = x
        # exit()
        temp = torch.ones((1, 8, 40, 40))
        # x = self.csp6_cat1_0(reg_output_P4_address[0], reg_output_P4_address[1], reg_preds1_out, 1, 1, 1,
        #                      path1, stride=1, padding=1, block=0, cat2_weight_address=obj_output_P4_address[1],
        #                      cat2=obj_preds1_out, operator='concat')
        x = self.csp6_cat1_0(reg_output_P4_address[0], reg_output_P4_address[1], temp, 1, 1, 1,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=obj_output_P4_address[1],
                             cat2=temp, operator='concat', index=2)
        csp6_cat1_0 = torch.ones((1, 16, 40, 40))
        x = self.csp6_cat1_1(x[0], x[1], csp6_cat1_0, 1, 1, 1,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=cls_output_P4_address[1],
                             cat2=temp, operator='concat')
        # exit()
        # ===========================================================================
        weight_stems2_conv = tensorr('../para_682/head.stems.2.conv.weight.int.npy')
        weight_stems2_conv_f = tensorr('../para_682/head.stems.2.conv.weight.npy')
        x_1 = self.stems2_conv(obj_output_P4_address[0], P5_out_address[1], P5_out, weight_stems2_conv_f,
                               weight_stems2_conv,
                               self.bias_int_stems2_conv, path1, stride=1, padding=0, block=0, operator='conv11',
                               index=21)
        # exit()
        weight_cls_convs2_0 = tensorr('../para_682/head.cls_convs.2.0.conv.weight.int.npy')
        weight_cls_convs2_0_f = tensorr('../para_682/head.cls_convs.2.0.conv.weight.npy')
        x = self.cls_convs2_0(x_1[0], x_1[1], stem2_out, weight_cls_convs2_0_f, weight_cls_convs2_0,
                              self.bias_int_cls_convs2_0, path1, stride=1, padding=1, block=0, operator='conv33')

        weight_cls_convs2_1 = tensorr('../para_682/head.cls_convs.2.1.conv.weight.int.npy')
        weight_cls_convs2_1_f = tensorr('../para_682/head.cls_convs.2.1.conv.weight.npy')
        x = self.cls_convs2_1(x[0], x[1], cls_convs2_0_out, weight_cls_convs2_1_f, weight_cls_convs2_1,
                              self.bias_int_cls_convs2_1, path1, stride=1, padding=1, block=0, operator='conv33')

        weight_cls_preds2 = tensorr('../para_682/head.cls_preds.2.weight.int.npy')
        weight_cls_preds2_f = tensorr('../para_682/head.cls_preds.2.weight.npy')
        # 1 128  20   20        1 128  1  1-> 8 128 1 1  ------ 1  8 20 20
        weight_cls_preds2 = weight_4to8(weight_cls_preds2)
        x = self.cls_preds2(x[0], x[1], cls_convs2_1_out, weight_cls_preds2_f, weight_cls_preds2,
                            self.bias_int_cls_preds2, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11')
        cls_output_P5_address = x

        weight_reg_convs2_0 = tensorr('../para_682/head.reg_convs.2.0.conv.weight.int.npy')
        weight_reg_convs2_0_f = tensorr('../para_682/head.reg_convs.2.0.conv.weight.npy')
        x = self.reg_convs2_0(x[0], x_1[1], stem2_out, weight_reg_convs2_0_f, weight_reg_convs2_0,
                              self.bias_int_reg_convs2_0, path1, stride=1, padding=1, block=0, operator='conv33',
                              index=4)
        # exit()
        weight_reg_convs2_1 = tensorr('../para_682/head.reg_convs.2.1.conv.weight.int.npy')
        weight_reg_convs2_1_f = tensorr('../para_682/head.reg_convs.2.1.conv.weight.npy')
        reg_feat = self.reg_convs2_1(x[0], x[1], reg_convs2_0_out, weight_reg_convs2_1_f, weight_reg_convs2_1,
                                     self.bias_int_reg_convs2_1, path1, stride=1, padding=1, block=0, operator='conv33')
        # exit()
        weight_reg_preds2 = tensorr('../para_682/head.reg_preds.2.weight.int.npy')
        weight_reg_preds2_f = tensorr('../para_682/head.reg_preds.2.weight.npy')
        # 1 128 20 20        4 128 1 1 -> 8 128 1 1  --------------------- 1 8 20 20
        weight_reg_preds2 = weight_4to8(weight_reg_preds2)
        x = self.reg_preds2(reg_feat[0], reg_feat[1], reg_convs2_1_out, weight_reg_preds2_f, weight_reg_preds2,
                            self.bias_int_reg_preds2, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11')
        reg_output_P5_address = x

        weight_obj_preds2 = tensorr('../para_682/head.obj_preds.2.weight.int.npy')
        weight_obj_preds2_f = tensorr('../para_682/head.obj_preds.2.weight.npy')
        # 1 128 20 20      1 128 1 1 -> 8 128 1 1 ------------------------1 8 20 20
        weight_obj_preds2 = weight_4to8(weight_obj_preds2)
        x = self.obj_preds2(x[0], reg_feat[1], reg_convs2_1_out, weight_obj_preds2_f, weight_obj_preds2,
                            self.bias_int_obj_preds2, path1, stride=1, padding=0, block=0, isleakrelu=1,
                            operator='conv11', index=2)
        obj_output_P5_address = x
        temp = torch.ones((1, 8, 20, 20))
        # x = self.csp6_cat2_0(reg_output_P5_address[0], reg_output_P5_address[1], reg_preds2_out, 1, 1, 1,
        #                      path1, stride=1, padding=1, block=0, cat2_weight_address=obj_output_P5_address[1],
        #                      cat2=obj_preds2_out, operator='concat')
        x = self.csp6_cat2_0(reg_output_P5_address[0], reg_output_P5_address[1], temp, 1, 1, 1,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=obj_output_P5_address[1],
                             cat2=temp, operator='concat', index=2)
        csp6_cat2_0 = torch.ones((1, 16, 20, 20))
        x = self.csp6_cat2_1(x[0], x[1], csp6_cat2_0, 1, 1, 1,
                             path1, stride=1, padding=1, block=0, cat2_weight_address=cls_output_P5_address[1],
                             cat2=temp, operator='concat')


if __name__ == "__main__":
    # 权重的起始地址,对应的16进制为70000000
    weight_address = 1879048192
    model = QuantizableYolo_tiny()(1)
    time.sleep(1)
    ins64to32('../ins/yolox_ins_64_682.dat', '../ins/yolox_ins_32_682.dat')
