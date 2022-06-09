# -*- coding: UTF-8 -*-
# encoding=utf-8
import torch
import torch.nn as nn
import numpy as np
from picture_load import *
from ins_conv_new_831 import *
import cv2
# np.set_printoptions(threshold=np.inf)
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys

sys.path.append("..")
import torch.nn.functional as F
import torch.quantization
from torch.nn.quantized import functional as qF
# # Setup warnings
import warnings
from torch.quantization import QuantStub, DeQuantStub


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

    '''
    --weight_address:权重地址
    --computer_address: 
    --feature:进入该操作之前的feature map
    --f_weight:该层的浮点值权重
    --q_weight:量化后的权重值
    --bias:该层的浮点值bias
    --path1:bin文件地址
    --stride:
    --padding:
    --block:是否分块 0不分块
    --operator:操作类型

    '''

    def forward(self, weight_address, computer_address, feature, f_weight, q_weight, bias=0, path1=0, stride=2,
                padding=1, block=0, cat2_weight_address=1, isleakrelu=0, add_write_address=16777216,
                cat2=torch.tensor((1, 1, 1, 1)),
                operator=''):  # weight_address权重地址 computer_address计算地址 add_write_address加上这个地址变成写地址
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
            file_name = 'mytest.txt'
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
                shape = q_weight.shape
                # 在fpga中一个地址可以存储8bit的数据，weight_size是用来计算当前层的权重数据所占的地址的总数，从而能够计算出该层权重数据的结束地址
                # 方便fpga的数据读取
                # 第一层每个卷积核有9个值 加上补的23个零组一共32个 占32*8=256bit
                # 32个卷积核在补零之后所占的bit = 32*256 所占的地址数 = 32*256/8 =1024
                # bias，scale，shift(N_REAL)占 32*3*32/8=384个地址
                # 指令256bit 占 256/8=32个地址，一共占1024+384+32=1440
                weight_size = int(1440)

                # ----------------conv33权重指令-------------------
                # 计算权重的reg4
                # dataSizeW是权重每行多少bit, dataSizeB是bias每行多少bit
                # dataSizeW = 8
                # dataSizeB = 8
                reg4 = default_data
                reg5 = default_data
                switch = default_data

                with open(file_name, 'a+') as f:
                    # f.write('13')
                    # 权重第一个指令:读地址 100000 1C
                    f.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    f.write('%08X' % int(weight_address))
                    f.write('\n')
                    # 权重第二个指令:读数量 100000 20
                    f.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    f.write('%08X' % weight_size)
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
                    f.write('1000000000000001')
                    f.write('\n')
                    f.write('110000040000000F')
                    f.write('\n')
                # ----------------conv33计算指令-------------------
                # 计算图片所含值的数量
                feature_shape = feature.shape
                feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
                # 计算写地址
                computer_write_address = computer_address + add_write_address  # 计算地址+add_write_address=写地址
                # 计算输出图片的大小
                out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1
                # 计算输出的feature所含值的数量
                write_size = feature_shape[0] * shape[0] * out_size * out_size
                # ------------------写入计算的指令----------------------
                with open(file_name, 'a+') as fp:

                    fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                    fp.write('%08X' % int(0))
                    fp.write('\n')

                    fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
                    fp.write('%08X' % int(feature_size))
                    fp.write('\n')

                    # 计算的第三个指令写地址
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
                    fp.write('%08X' % int(16777216))
                    fp.write('\n')
                    # 计算的第四个指令写数量
                    fp.write('100000' + ins_address['TJPU_DMA_Write_Num'])
                    fp.write('%08X' % int(write_size))
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
                    fp.write('%08X' % int(2))
                    fp.write('\n')
                    fp.write('110000040000000F')
                    fp.write('\n')

                weight_address = weight_address + weight_size  # 1886457312
                return weight_address, computer_write_address
            elif (operator == "conv33"):  # 其他层是conv33
                shape = q_weight.shape
                # 计算权重的数量（计算weight+bias+scale+shift，单位是B）
                weight_size = (shape[0] * shape[1] * shape[2] * shape[3])
                # 最终weight_size的含义和第一层的是一样的
                # shape[0]是m输出通道个数，bias+scale+shift是3，每个数是32bit=4B
                weight_size += ((shape[0]) * 3 * 4)

                # ----------------conv33权重指令-------------------
                # 计算权重的reg4
                # dataSizeW是权重每行多少bit, dataSizeB是bias每行多少bit
                dataSizeW = 64
                dataSizeB = 64
                #
                reg4 = conv33para(shape[0], shape[1], dataSizeW, dataSizeB)
                # 计算权重的reg5
                reg5 = '00000000'
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
                    # 计算的第五个指令,switch  switch总共32位  0001
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
                feature_size = feature_shape[0] * feature_shape[1] * feature_shape[2] * feature_shape[3]
                # 计算写地址
                computer_write_address = computer_address + add_write_address
                # 计算输出图片的大小
                out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1
                # 计算写地址的数量
                write_size = feature_shape[0] * shape[0] * out_size * out_size
                # 计算的reg4 计算reg5
                computer_reg4, computer_reg5, computer_reg6, computer_reg7 = conv33compute(shape[0], shape[1],
                                                                                           dataSizeW,
                                                                                           dataSizeB, feature.shape[2],
                                                                                           stride, padding,
                                                                                           self.quant_zero_point1,
                                                                                           self.quant_zero_point3,
                                                                                           self.quant_scale3)

                # -------------------写入计算的指令----------------------
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
                return weight_address, computer_write_address
            elif (operator == "concat"):
                # feature 是resblock conv3的结果，cat2是conv2的结果
                cat1_shape = feature.shape
                cat2_shape = cat2.shape
                cat1_address = computer_address
                cat2_address = cat2_weight_address

                # cat1_size 需要cat操作的第一个feature map中值的数量
                # cat2_size 需要cat操作的第二个feature map中值的数量
                cat1_size = cat1_shape[0] * cat1_shape[1] * cat1_shape[2] * cat1_shape[3]
                cat2_size = cat2_shape[0] * cat2_shape[1] * cat2_shape[2] * cat2_shape[3]
                # 计算写地址
                computer_write_address = cat1_address + add_write_address
                # 计算concat的reg4 10bit feature map的通道数
                cat1_channel = '{:010b}'.format(
                    cat2_shape[1])

                reg4 = cat1_channel + '0000000000000000000000'

                reg4 = str(int(reg4, 2))
                # 计算concat的reg5()
                feature_h = '{:011b}'.format(cat2_shape[2])  # 11 位 输入图片 高  (行数)
                cat2_channel = '{:010b}'.format(cat1_shape[1])  # 10 位，concat 中 cat1 的通道数
                feature_w = '{:011b}'.format(cat2_shape[3])  # 11 位，输入的图片 宽  (列数)
                # 下面位二进制
                reg5 = feature_h + cat2_channel + feature_w
                # 将二进制转成10进制
                reg5 = str(int(reg5, 2))
                # 计算reg6,reg7,reg8,reg9  M1 M2  zero_point_one zero_point_two
                reg6, reg7, reg8, reg9 = reg_cat(self.quant_scale1, self.quant_scale2, self.quant_scale3,
                                                 self.quant_zero_point1,
                                                 self.quant_zero_point2, self.quant_zero_point3)
                default_data = 1300005000000000

                ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                               'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                               'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                               'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                               'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                               'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                               'Image_Reg1': '0C'}
                # ----------------concat权重指令-------------------
                # 6个全都是默认的
                with open(file_name, 'a+') as fp:

                    # -------------concat的计算指令--------------
                    # 第一个指令:读第一个concat地址
                    print('%08X' % cat2_address)
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
                    fp.write('00000100')
                    fp.write('\n')
                    fp.write('1100001400000F00')
                    fp.write('\n')
                return weight_address, computer_write_address
            elif (operator == "conv11"):

                shape = q_weight.shape
                dataSizeW = 64
                dataSizeB = 64
                # 计算权重的数量
                weight_size = (shape[0] * shape[1] * shape[2] * shape[3])
                weight_size += ((shape[0]) * 3 * 4)
                # 权重11conv的reg4
                reg4 = conv11para(shape[0], shape[1], dataSizeW, dataSizeB)
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
                computer_write_address = computer_address + add_write_address
                # 计算输出图片的大小
                out_size = int((feature_shape[2] - 1 + 2 * padding) / stride) + 1
                # 计算写地址的数量
                write_size = feature_shape[0] * shape[0] * out_size * out_size
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
                if self.coe_name == '../yolo_head_P4_26_coe/hand_P4_26_conv11.coe':
                    computer_reg6 = 0
                print(self.coe_name)
                if self.coe_name == '../yolo_head_p5_13_coe/hand_P5_13_conv11.coe':
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

                return weight_address, computer_write_address


def conv33_block(coe_name, weight_address, computer_address, feature, f_weight, q_weight, bias, path1, stride, padding,
                 block, cat2_weight_address, isleakrelu, add_write_address, operator, quant_zero_point1,
                 quant_zero_point3,
                 quant_scale3):  # 分成4块，然后做concat
    default_data = 1300005000000000
    file_name = 'mytest.txt'
    # reg表对应的地址
    ins_address = {'TJPU_Control': '10', 'TJPU_State': '14',
                   'TJPU_Switch': '18', 'TJPU_DMA_Read_Addr': '1C',
                   'TJPU_DMA_Read_Num': '20', 'TJPU_DMA_Write_Addr': '24',
                   'TJPU_DMA_Write_Num': '28', 'TJPU_Reg4': '2C',
                   'TJPU_Reg5': '30', 'TJPU_Reg6': '34',
                   'TJPU_Reg7': '38', 'TJPU_Reg8': '3C', 'TJPU_Reg9': '40', 'Image_Reg0': '08',
                   'Image_Reg1': '0C'}
    # exit()
    if block != 0:

        shape = q_weight.shape  # m输出通道数 c输入通道数 k卷积核大小 k
        # 计算权重的数量
        channel_num = shape[0] / block
        weight_size = (channel_num * shape[1] * shape[2] * shape[3])
        weight_size += (channel_num * 3 * 4)
        # weight_size为权重的数量

        # ----------------conv33权重指令-------------------
        # 计算权重的reg4
        # dataSizeW是权重每行多少bit, dataSizeB是bias每行多少bit
        dataSizeW = 64
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
        out_size = int((feature_shape[2] - 3 + 2 * padding) / stride) + 1
        # 计算写地址的数量
        write_size = feature_shape[0] * channel_num * out_size * out_size
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
                fp.write('%08X' % int(computer_write_address + write_size * index + 16777216 * add_new))
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
                fp.write('\n')
                # 第二个指令:读第一个concat数量
                fp.write('100000' + ins_address['Image_Reg1'])
                fp.write('%08X' % int(write_size))
                fp.write('\n')
                # 第三个指令:读第二个concat地址
                fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
                fp.write(
                    '%08X' % int(computer_write_address + write_size * cat_index * 2 + write_size + 16777216 * add_new))
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
            print('***********************************')
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
                print(computer_write_address)
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


'''
add 量化公式
s3(q3-zp3) = s1(q1-zp1)+s2(q2-zp2)
q3 = (s1/s3)(q1-zp1)+(s2/s3)(q2-zp2)+zp3 pytorch对激活使用非对称量化
q3 = (s1/s3)(q1-zp1+(s3/s1)(zp3/2))+(s2/s3)(q2-zp2+(s3/s2)(zp3/2))
'''


def reg_add(x_scale, y_scale, z_scale, x_zp, y_zp, z_zp):
    final_zp_one = (-x_zp + (z_scale / x_scale) * (z_zp / 2))
    final_zp_one = (torch.round(final_zp_one * (2 ** 16)))
    final_zp_one = final_zp_one.numpy().astype(np.uint32)
    M1 = (torch.round((x_scale / z_scale) * (2 ** 16)))
    M1 = M1.numpy().astype(np.uint32)

    final_zp_two = (-y_zp + (z_scale / y_scale) * (z_zp / 2))
    final_zp_two = (torch.round(final_zp_two * (2 ** 16)))
    final_zp_two = final_zp_two.numpy().astype(np.uint32)
    M2 = (torch.round((y_scale / z_scale) * (2 ** 16)))
    M2 = M2.numpy().astype(np.uint32)

    return M1, M2, final_zp_one, final_zp_two


'''
  reg_cat
  --cat1_scale:需要cat的第一部分feature map 的scale
  --cat2_scale:需要cat的第二部分 feature map的 scale
  --cat3_scale:cat完成之后 feature map 的scale
  --cat1_zero_point:需要cat的第一部分feature map 的zp
  --cat2_zero_point:需要cat的第二部分 feature map的zp
  --cat3_zero_point:cat完成之后 feature map 的zp
   concat量化公式: s1(q1-zp1) = s3(q3-zp3)  q3 = s1/s3(q1-zp1)+zp3 = s1/s3(q1-zp1+s3/s1*zp3)
'''


def reg_cat(cat1_scale, cat2_scale, cat3_scale, cat1_zero_point, cat2_zero_point, cat3_zero_point):  # 做量化 两组r3=r1和r3=r2
    zero_point_one = (cat3_scale / cat1_scale) * cat3_zero_point - cat1_zero_point  # -zp1+s3/s1*zp3
    zero_point_one = (torch.round(zero_point_one * (2 ** 16)))  # 左移16位后取整
    zero_point_one = zero_point_one.numpy().astype(np.uint32)  # 转换成uint32
    M1 = (torch.round((cat1_scale / cat3_scale) * (2 ** 16)))  # 定义M s1/s3 左移16位取整
    M1 = M1.numpy().astype(np.uint32)  # 转换成uint32
    print('%08x' % zero_point_one)
    print('---------------------')
    print('%08x' % M1)
    zero_point_two = (cat3_scale / cat2_scale) * cat3_zero_point - cat2_zero_point
    zero_point_two = (torch.round(zero_point_two * (2 ** 16)))
    zero_point_two = zero_point_two.numpy().astype(np.uint32)
    M2 = (torch.round((cat2_scale / cat3_scale) * (2 ** 16)))
    M2 = M2.numpy().astype(np.uint32)
    print('%08x' % zero_point_two)
    print('%08x' % M2)
    return M1, M2, zero_point_one, zero_point_two


def reshape(weight_address, computer_address, feature, operator="", add_write_address=16777216, filename=' '):
    default_data = 1300005000000000
    # 计算读的数量
    shape = feature.shape  # b c  h w
    print('reshape shape',shape)

    channel_in = shape[1]
    feature_in = shape[2]
    # 128,20,20
    if operator == "upsample":
        feature_in = int(feature_in / 2)  # 1593行reshape传的是P5_Upsample，是p5做了1*1和Upsample之后的结果扩大了2倍
    feature_size = shape[0] * channel_in * feature_in * feature_in
    # 1* 128*128*10*10
    # 计算写地址
    computer_write_address = computer_address + add_write_address
    # 计算写数量
    write_size = 0
    if (operator == "maxpool"):
        write_size = int(feature_size / 4)  # 长宽缩小一半
    elif (operator == "upsample"):
        write_size = int(feature_size * 4)  # 一个写四遍
        # 128*20*20

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
        print('%08X' % computer_address)

        fp.write('100000' + ins_address['TJPU_DMA_Read_Addr'])
        fp.write('%08X' % computer_address)
        fp.write('\n')
        # 第二个指令:读第一个reshape数量
        fp.write('100000' + ins_address['TJPU_DMA_Read_Num'])
        fp.write('%08X' % feature_size)
        fp.write('\n')
        # 第三个指令:写地址
        fp.write('100000' + ins_address['TJPU_DMA_Write_Addr'])
        fp.write('%08X' % computer_write_address)
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
            fp.write('00000400')
            fp.write('\n')
            fp.write('1100001400000F00')
            fp.write('\n')
        elif (operator == "upsample"):
            fp.write('100000' + ins_address['TJPU_Control'])
            fp.write('00000800')
            fp.write('\n')
            fp.write('1100001400000F00')
            fp.write('\n')

    return weight_address, computer_write_address


def tensorr(x):
    tensor_py = torch.from_numpy(np.load(x))
    return tensor_py


class QuantizableYolo_tiny(nn.Module):

    def __init__(self, img_size=416):
        super(QuantizableYolo_tiny, self).__init__()
        ###############################
        # print(self)
        # exit()
        s1_conv1 = tensorr('../para800_1028_new/quant.scale.npy')
        z1_conv1 = tensorr('../para800_1028_new/quant.zero_point.npy')

        s2_conv1 = tensorr('../para800_1028_new/conv1.conv.0.weight.scale.npy')
        z2_conv1 = tensorr('../para800_1028_new/conv1.conv.0.weight.zero_point.npy')
        s3_conv1 = tensorr('../para800_1028_new/conv1.conv.0.scale.npy')
        z3_conv1 = tensorr('../para800_1028_new/conv1.conv.0.zero_point.npy')
        bias_f_conv1 = tensorr('../para800_1028_new/conv1.conv.0.bias.npy')

        self.bias_int_conv1 = bias_f_conv1
        coe_name = '../data1_coe/out_hand_conv1_leak.coe'
        self.conv1 = Conv2d_Q(quant_scale1=s1_conv1, quant_zero_point1=z1_conv1, quant_scale2=s2_conv1,
                              quant_zero_point2=z2_conv1, quant_scale3=s3_conv1,
                              quant_zero_point3=z3_conv1,
                              first_convs=1, coe_name=coe_name)
        s1_conv2 = tensorr('../para800_1028_new/conv1.conv.0.scale.npy')
        z1_conv2 = tensorr('../para800_1028_new/conv1.conv.0.zero_point.npy')
        s2_conv2 = tensorr('../para800_1028_new/conv2.conv.0.weight.scale.npy')
        z2_conv2 = tensorr('../para800_1028_new/conv2.conv.0.weight.zero_point.npy')
        s3_conv2 = tensorr('../para800_1028_new/conv2.conv.0.scale.npy')
        z3_conv2 = tensorr('../para800_1028_new/conv2.conv.0.zero_point.npy')

        bias_f_conv2 = tensorr('../para800_1028_new/conv2.conv.0.bias.npy')

        self.bias_int_conv2 = bias_f_conv2
        #
        coe_name = '../data2_coe/out_hand_conv2_leak.coe'
        self.conv2 = Conv2d_Q(quant_scale1=s1_conv2, quant_zero_point1=z1_conv2, quant_scale2=s2_conv2,
                              quant_zero_point2=z2_conv2, quant_scale3=s3_conv2,
                              quant_zero_point3=z3_conv2,
                              first_convs=0, coe_name=coe_name)
        s1_rs1_conv1 = tensorr('../para800_1028_new/conv2.conv.0.scale.npy')
        z1_rs1_conv1 = tensorr('../para800_1028_new/conv2.conv.0.zero_point.npy')
        s2_rs1_conv1 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.weight.scale.npy')
        z2_rs1_conv1 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.weight.zero_point.npy')
        s3_rs1_conv1 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.scale.npy')
        z3_rs1_conv1 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.zero_point.npy')

        bias_rs1_f_conv1 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.bias.npy')

        self.bias_rs1_int_conv1 = bias_rs1_f_conv1
        #
        coe_name = '../rs1_coe/out_hand_rs1_conv1_leak.coe'
        self.rs1_conv1 = Conv2d_Q(quant_scale1=s1_rs1_conv1, quant_zero_point1=z1_rs1_conv1, quant_scale2=s2_rs1_conv1,
                                  quant_zero_point2=z2_rs1_conv1, quant_scale3=s3_rs1_conv1,
                                  quant_zero_point3=z3_rs1_conv1,
                                  first_convs=0, coe_name=coe_name)
        s1_rs1_conv2 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.scale.npy')
        z1_rs1_conv2 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.zero_point.npy')
        s2_rs1_conv2 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.weight.scale.npy')
        z2_rs1_conv2 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.weight.zero_point.npy')
        s3_rs1_conv2 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.scale.npy')
        z3_rs1_conv2 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.zero_point.npy')

        bias_rs1_f_conv2 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.bias.npy')

        self.bias_rs1_int_conv2 = bias_rs1_f_conv2
        #
        coe_name = '../rs1_coe/out_hand_rs1_conv2_leak.coe'
        self.rs1_conv2 = Conv2d_Q(quant_scale1=s1_rs1_conv2, quant_zero_point1=z1_rs1_conv2, quant_scale2=s2_rs1_conv2,
                                  quant_zero_point2=z2_rs1_conv2, quant_scale3=s3_rs1_conv2,
                                  quant_zero_point3=z3_rs1_conv2, first_convs=0, coe_name=coe_name)
        s1_rs1_conv3 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.scale.npy')
        z1_rs1_conv3 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.zero_point.npy')
        s2_rs1_conv3 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.weight.scale.npy')
        z2_rs1_conv3 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.weight.zero_point.npy')
        s3_rs1_conv3 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.scale.npy')
        z3_rs1_conv3 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.zero_point.npy')

        bias_rs1_f_conv3 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.bias.npy')

        self.bias_rs1_int_conv3 = bias_rs1_f_conv3

        coe_name = '../rs1_coe/out_hand_rs1_conv3_leak.coe'
        self.rs1_conv3 = Conv2d_Q(quant_scale1=s1_rs1_conv3, quant_zero_point1=z1_rs1_conv3, quant_scale2=s2_rs1_conv3,
                                  quant_zero_point2=z2_rs1_conv3, quant_scale3=s3_rs1_conv3,
                                  quant_zero_point3=z3_rs1_conv3,
                                  first_convs=0, coe_name=coe_name)

        s1_rs1_conv5 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.scale.npy')
        z1_rs1_conv5 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.zero_point.npy')
        s2_rs1_conv5 = tensorr('../para800_1028_new/resblock_body1.conv5.conv.0.weight.scale.npy')
        z2_rs1_conv5 = tensorr('../para800_1028_new/resblock_body1.conv5.conv.0.weight.zero_point.npy')
        s3_rs1_conv5 = tensorr('../para800_1028_new/resblock_body1.conv5.conv.0.scale.npy')
        z3_rs1_conv5 = tensorr('../para800_1028_new/resblock_body1.conv5.conv.0.zero_point.npy')

        bias_rs1_f_conv5 = tensorr('../para800_1028_new/resblock_body1.conv5.conv.0.bias.npy')

        self.bias_rs1_int_conv5 = bias_rs1_f_conv5
        #
        coe_name = '../rs1_coe/out_hand_rs1_conv5_leak.coe'
        self.rs1_conv5 = Conv2d_Q(quant_scale1=s1_rs1_conv5, quant_zero_point1=z1_rs1_conv5, quant_scale2=s2_rs1_conv5,
                                  quant_zero_point2=z2_rs1_conv5, quant_scale3=s3_rs1_conv5,
                                  quant_zero_point3=z3_rs1_conv5,
                                  first_convs=0, coe_name=coe_name)
        s1_rs1_conv4 = tensorr('../para800_1028_new/resblock_body1.routeq0.scale.npy')
        z1_rs1_conv4 = tensorr('../para800_1028_new/resblock_body1.routeq0.zero_point.npy')
        s2_rs1_conv4 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.weight.scale.npy')
        z2_rs1_conv4 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.weight.zero_point.npy')
        s3_rs1_conv4 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.scale.npy')
        z3_rs1_conv4 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.zero_point.npy')

        bias_rs1_f_conv4 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.bias.npy')

        self.bias_rs1_int_conv4 = bias_rs1_f_conv4
        #
        coe_name = '../rs1_coe/out_hand_rs1_conv4_leak.coe'
        self.rs1_conv4 = Conv2d_Q(quant_scale1=s1_rs1_conv4, quant_zero_point1=z1_rs1_conv4, quant_scale2=s2_rs1_conv4,
                                  quant_zero_point2=z2_rs1_conv4, quant_scale3=s3_rs1_conv4,
                                  quant_zero_point3=z3_rs1_conv4,
                                  first_convs=0, coe_name=coe_name)
        self.rs1_cat1 = Conv2d_Q(quant_scale1=s3_rs1_conv3, quant_zero_point1=z3_rs1_conv3, quant_scale2=s3_rs1_conv2,
                                 quant_zero_point2=z3_rs1_conv2, quant_scale3=s1_rs1_conv4,
                                 quant_zero_point3=z1_rs1_conv4,
                                 first_convs=0, coe_name=coe_name)

        s1_rs2_conv1 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.scale.npy')
        z1_rs2_conv1 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.zero_point.npy')
        s2_rs2_conv1 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.weight.scale.npy')
        z2_rs2_conv1 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.weight.zero_point.npy')
        s3_rs2_conv1 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.scale.npy')
        z3_rs2_conv1 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.zero_point.npy')

        bias_rs2_f_conv1 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.bias.npy')
        self.bias_rs2_int_conv1 = bias_rs2_f_conv1
        #
        coe_name = '../rs2_coe/out_hand_rs2_conv1_leak.coe'
        self.rs2_conv1 = Conv2d_Q(quant_scale1=s1_rs2_conv1, quant_zero_point1=z1_rs2_conv1, quant_scale2=s2_rs2_conv1,
                                  quant_zero_point2=z2_rs2_conv1, quant_scale3=s3_rs2_conv1,
                                  quant_zero_point3=z3_rs2_conv1,
                                  first_convs=0, coe_name=coe_name)
        s1_rs2_conv2 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.scale.npy')
        z1_rs2_conv2 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.zero_point.npy')
        s2_rs2_conv2 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.weight.scale.npy')
        z2_rs2_conv2 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.weight.zero_point.npy')
        s3_rs2_conv2 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.scale.npy')
        z3_rs2_conv2 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.zero_point.npy')

        bias_rs2_f_conv2 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.bias.npy')
        self.bias_rs2_int_conv2 = bias_rs2_f_conv2
        #
        coe_name = '../rs2_coe/out_hand_rs2_conv2_leak.coe'
        self.rs2_conv2 = Conv2d_Q(quant_scale1=s1_rs2_conv2, quant_zero_point1=z1_rs2_conv2, quant_scale2=s2_rs2_conv2,
                                  quant_zero_point2=z2_rs2_conv2, quant_scale3=s3_rs2_conv2,
                                  quant_zero_point3=z3_rs2_conv2, first_convs=0, coe_name=coe_name)
        s1_rs2_conv3 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.scale.npy')
        z1_rs2_conv3 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.zero_point.npy')
        s2_rs2_conv3 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.weight.scale.npy')
        z2_rs2_conv3 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.weight.zero_point.npy')
        s3_rs2_conv3 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.scale.npy')
        z3_rs2_conv3 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.zero_point.npy')

        bias_rs2_f_conv3 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.bias.npy')
        self.bias_rs2_int_conv3 = bias_rs2_f_conv3
        #
        coe_name = '../rs2_coe/out_hand_rs2_conv3_leak.coe'
        self.rs2_conv3 = Conv2d_Q(quant_scale1=s1_rs2_conv3, quant_zero_point1=z1_rs2_conv3, quant_scale2=s2_rs2_conv3,
                                  quant_zero_point2=z2_rs2_conv3, quant_scale3=s3_rs2_conv3,
                                  quant_zero_point3=z3_rs2_conv3,
                                  first_convs=0, coe_name=coe_name)
        s1_rs2_conv5 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.scale.npy')
        z1_rs2_conv5 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.zero_point.npy')
        s2_rs2_conv5 = tensorr('../para800_1028_new/resblock_body2.conv5.conv.0.weight.scale.npy')
        z2_rs2_conv5 = tensorr('../para800_1028_new/resblock_body2.conv5.conv.0.weight.zero_point.npy')
        s3_rs2_conv5 = tensorr('../para800_1028_new/resblock_body2.conv5.conv.0.scale.npy')
        z3_rs2_conv5 = tensorr('../para800_1028_new/resblock_body2.conv5.conv.0.zero_point.npy')

        bias_rs2_f_conv5 = tensorr('../para800_1028_new/resblock_body2.conv5.conv.0.bias.npy')

        self.bias_rs2_int_conv5 = bias_rs2_f_conv5
        #
        coe_name = '../rs2_coe/out_hand_rs2_conv5_leak.coe'
        self.rs2_conv5 = Conv2d_Q(quant_scale1=s1_rs2_conv5, quant_zero_point1=z1_rs2_conv5, quant_scale2=s2_rs2_conv5,
                                  quant_zero_point2=z2_rs2_conv5, quant_scale3=s3_rs2_conv5,
                                  quant_zero_point3=z3_rs2_conv5,
                                  first_convs=0, coe_name=coe_name)
        s1_rs2_conv4 = tensorr('../para800_1028_new/resblock_body2.routeq0.scale.npy')
        z1_rs2_conv4 = tensorr('../para800_1028_new/resblock_body2.routeq0.zero_point.npy')
        s2_rs2_conv4 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.weight.scale.npy')
        z2_rs2_conv4 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.weight.zero_point.npy')
        s3_rs2_conv4 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.scale.npy')
        z3_rs2_conv4 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.zero_point.npy')

        bias_rs2_f_conv4 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.bias.npy')
        self.bias_rs2_int_conv4 = bias_rs2_f_conv4
        #
        coe_name = '../rs2_coe/out_hand_rs2_conv4_leak.coe'
        self.rs2_conv4 = Conv2d_Q(quant_scale1=s1_rs2_conv4, quant_zero_point1=z1_rs2_conv4, quant_scale2=s2_rs2_conv4,
                                  quant_zero_point2=z2_rs2_conv4, quant_scale3=s3_rs2_conv4,
                                  quant_zero_point3=z3_rs2_conv4,
                                  first_convs=0, coe_name=coe_name)
        self.rs2_cat1 = Conv2d_Q(quant_scale1=s3_rs2_conv3, quant_zero_point1=z3_rs2_conv3, quant_scale2=s3_rs2_conv2,
                                 quant_zero_point2=z3_rs2_conv2, quant_scale3=s1_rs2_conv4,
                                 quant_zero_point3=z1_rs2_conv4,
                                 first_convs=0, coe_name=coe_name)
        s1_rs3_conv1 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.scale.npy')
        z1_rs3_conv1 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.zero_point.npy')
        s2_rs3_conv1 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.weight.scale.npy')
        z2_rs3_conv1 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.weight.zero_point.npy')
        s3_rs3_conv1 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.scale.npy')
        z3_rs3_conv1 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.zero_point.npy')

        bias_rs3_f_conv1 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.bias.npy')
        self.bias_rs3_int_conv1 = bias_rs3_f_conv1
        #
        coe_name = '../rs3_coe/out_hand_rs3_conv1_leak.coe'
        self.rs3_conv1 = Conv2d_Q(quant_scale1=s1_rs3_conv1, quant_zero_point1=z1_rs3_conv1, quant_scale2=s2_rs3_conv1,
                                  quant_zero_point2=z2_rs3_conv1, quant_scale3=s3_rs3_conv1,
                                  quant_zero_point3=z3_rs3_conv1,
                                  first_convs=0, coe_name=coe_name)
        s1_rs3_conv2 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.scale.npy')
        z1_rs3_conv2 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.zero_point.npy')
        s2_rs3_conv2 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.weight.scale.npy')
        z2_rs3_conv2 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.weight.zero_point.npy')
        s3_rs3_conv2 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.scale.npy')
        z3_rs3_conv2 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.zero_point.npy')

        bias_rs3_f_conv2 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.bias.npy')
        self.bias_rs3_int_conv2 = bias_rs3_f_conv2
        #
        coe_name = '../rs3_coe/out_hand_rs3_conv2_leak.coe'
        self.rs3_conv2 = Conv2d_Q(quant_scale1=s1_rs3_conv2, quant_zero_point1=z1_rs3_conv2, quant_scale2=s2_rs3_conv2,
                                  quant_zero_point2=z2_rs3_conv2, quant_scale3=s3_rs3_conv2,
                                  quant_zero_point3=z3_rs3_conv2, first_convs=0, coe_name=coe_name)
        s1_rs3_conv3 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.scale.npy')
        z1_rs3_conv3 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.zero_point.npy')
        s2_rs3_conv3 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.weight.scale.npy')
        z2_rs3_conv3 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.weight.zero_point.npy')
        s3_rs3_conv3 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.scale.npy')
        z3_rs3_conv3 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.zero_point.npy')

        bias_rs3_f_conv3 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.bias.npy')
        self.bias_rs3_int_conv3 = bias_rs3_f_conv3
        #
        coe_name = '../rs3_coe/out_hand_rs3_conv3_leak.coe'
        self.rs3_conv3 = Conv2d_Q(quant_scale1=s1_rs3_conv3, quant_zero_point1=z1_rs3_conv3, quant_scale2=s2_rs3_conv3,
                                  quant_zero_point2=z2_rs3_conv3, quant_scale3=s3_rs3_conv3,
                                  quant_zero_point3=z3_rs3_conv3,
                                  first_convs=0, coe_name=coe_name)
        s1_rs3_conv5 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.scale.npy')
        z1_rs3_conv5 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.zero_point.npy')
        s2_rs3_conv5 = tensorr('../para800_1028_new/resblock_body3.conv5.conv.0.weight.scale.npy')
        z2_rs3_conv5 = tensorr('../para800_1028_new/resblock_body3.conv5.conv.0.weight.zero_point.npy')
        s3_rs3_conv5 = tensorr('../para800_1028_new/resblock_body3.conv5.conv.0.scale.npy')
        z3_rs3_conv5 = tensorr('../para800_1028_new/resblock_body3.conv5.conv.0.zero_point.npy')

        bias_rs3_f_conv5 = tensorr('../para800_1028_new/resblock_body3.conv5.conv.0.bias.npy')

        self.bias_rs3_int_conv5 = bias_rs3_f_conv5
        #
        coe_name = '../rs3_coe/out_hand_rs3_conv5_leak.coe'
        self.rs3_conv5 = Conv2d_Q(quant_scale1=s1_rs3_conv5, quant_zero_point1=z1_rs3_conv5, quant_scale2=s2_rs3_conv5,
                                  quant_zero_point2=z2_rs3_conv5, quant_scale3=s3_rs3_conv5,
                                  quant_zero_point3=z3_rs3_conv5,
                                  first_convs=0, coe_name=coe_name)
        s1_rs3_conv4 = tensorr('../para800_1028_new/resblock_body3.routeq0.scale.npy')
        z1_rs3_conv4 = tensorr('../para800_1028_new/resblock_body3.routeq0.zero_point.npy')
        s2_rs3_conv4 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.weight.scale.npy')
        z2_rs3_conv4 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.weight.zero_point.npy')
        s3_rs3_conv4 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.scale.npy')
        z3_rs3_conv4 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.zero_point.npy')

        bias_rs3_f_conv4 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.bias.npy')
        self.bias_rs3_int_conv4 = bias_rs3_f_conv4
        #
        coe_name = '../rs3_coe/out_hand_rs3_conv4_leak.coe'
        self.rs3_conv4 = Conv2d_Q(quant_scale1=s1_rs3_conv4, quant_zero_point1=z1_rs3_conv4, quant_scale2=s2_rs3_conv4,
                                  quant_zero_point2=z2_rs3_conv4, quant_scale3=s3_rs3_conv4,
                                  quant_zero_point3=z3_rs3_conv4,
                                  first_convs=0, coe_name=coe_name)
        self.rs3_cat1 = Conv2d_Q(quant_scale1=s3_rs3_conv3, quant_zero_point1=z3_rs3_conv3, quant_scale2=s3_rs3_conv2,
                                 quant_zero_point2=z3_rs3_conv2, quant_scale3=s1_rs3_conv4,
                                 quant_zero_point3=z1_rs3_conv4,
                                 first_convs=0, coe_name=coe_name)
        s1_conv3 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.scale.npy')
        z1_conv3 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.zero_point.npy')
        s2_conv3 = tensorr('../para800_1028_new/conv3.conv.0.weight.scale.npy')
        z2_conv3 = tensorr('../para800_1028_new/conv3.conv.0.weight.zero_point.npy')
        s3_conv3 = tensorr('../para800_1028_new/conv3.conv.0.scale.npy')
        z3_conv3 = tensorr('../para800_1028_new/conv3.conv.0.zero_point.npy')

        bias_f_conv3 = tensorr('../para800_1028_new/conv3.conv.0.bias.npy')
        self.bias_int_conv3 = bias_f_conv3
        #
        coe_name = '../conv3_coe/out_hand_conv3_leak.coe'
        self.conv3 = Conv2d_Q(quant_scale1=s1_conv3, quant_zero_point1=z1_conv3, quant_scale2=s2_conv3,
                              quant_zero_point2=z2_conv3, quant_scale3=s3_conv3,
                              quant_zero_point3=z3_conv3,
                              first_convs=0, coe_name=coe_name)
        s1_conv_forP5 = tensorr('../para800_1028_new/conv3.conv.0.scale.npy')
        z1_conv_forP5 = tensorr('../para800_1028_new/conv3.conv.0.zero_point.npy')
        s2_conv_forP5 = tensorr('../para800_1028_new/conv_for_P5.conv.0.weight.scale.npy')
        z2_conv_forP5 = tensorr('../para800_1028_new/conv_for_P5.conv.0.weight.zero_point.npy')
        z3_conv_forP5 = tensorr('../para800_1028_new/conv_for_P5.conv.0.zero_point.npy')
        s3_conv_forP5 = tensorr('../para800_1028_new/conv_for_P5.conv.0.scale.npy')

        bias_conv_forP5_f = tensorr('../para800_1028_new/conv_for_P5.conv.0.bias.npy')
        self.bias_conv_forP5_int = bias_conv_forP5_f
        coe_name = '../conv_for_p5_coe/out_hand_conv_for_p5_leak.coe'
        self.conv_forP5 = Conv2d_Q(quant_scale1=s1_conv_forP5, quant_zero_point1=z1_conv_forP5,
                                   quant_scale2=s2_conv_forP5,
                                   quant_zero_point2=z2_conv_forP5, quant_scale3=s3_conv_forP5,
                                   quant_zero_point3=z3_conv_forP5,
                                   first_convs=0, coe_name=coe_name)
        s1_upsample_conv11 = tensorr('../para800_1028_new/conv_for_P5.conv.0.scale.npy')
        z1_upsample_conv11 = tensorr('../para800_1028_new/conv_for_P5.conv.0.zero_point.npy')
        s2_upsample_conv11 = tensorr('../para800_1028_new/upsample.upsample.0.conv.0.weight.scale.npy')
        z2_upsample_conv11 = tensorr('../para800_1028_new/upsample.upsample.0.conv.0.weight.zero_point.npy')
        s3_upsample_conv11 = tensorr('../para800_1028_new/upsample.upsample.0.conv.0.scale.npy')
        z3_upsample_conv11 = tensorr('../para800_1028_new/upsample.upsample.0.conv.0.zero_point.npy')

        bias_f_upsample_conv11 = tensorr('../para800_1028_new/upsample.upsample.0.conv.0.bias.npy')
        self.bias_upsample_conv11 = bias_f_upsample_conv11
        #
        coe_name = '../upsample_coe/out_hand_upsample_leak.coe'
        self.upsample_conv11 = Conv2d_Q(quant_scale1=s1_upsample_conv11, quant_zero_point1=z1_upsample_conv11,
                                        quant_scale2=s2_upsample_conv11,
                                        quant_zero_point2=z2_upsample_conv11, quant_scale3=s3_upsample_conv11,
                                        quant_zero_point3=z3_upsample_conv11,
                                        first_convs=0, coe_name=coe_name)

        s1_P4_26_conv33 = tensorr('../para800_1028_new/route0.scale.npy')
        z1_P4_26_conv33 = tensorr('../para800_1028_new/route0.zero_point.npy')
        s2_P4_26_conv33 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.weight.scale.npy')
        z2_P4_26_conv33 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.weight.zero_point.npy')
        s3_P4_26_conv33 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.scale.npy')
        z3_P4_26_conv33 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.zero_point.npy')

        bias_P4_26_conv33 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.bias.npy')
        self.bias_P4_26_conv33 = bias_P4_26_conv33
        coe_name = '../yolo_head_P4_26_coe/out_hand_P4_26_conv33.coe'
        self.P4_26_conv33 = Conv2d_Q(quant_scale1=s1_P4_26_conv33, quant_zero_point1=z1_P4_26_conv33,
                                     quant_scale2=s2_P4_26_conv33,
                                     quant_zero_point2=z2_P4_26_conv33, quant_scale3=s3_P4_26_conv33,
                                     quant_zero_point3=z3_P4_26_conv33,
                                     first_convs=0, coe_name=coe_name)
        self.big_cat = Conv2d_Q(quant_scale1=s3_upsample_conv11, quant_zero_point1=z3_upsample_conv11,
                                quant_scale2=s3_rs3_conv5,
                                quant_zero_point2=z3_rs3_conv5, quant_scale3=s1_P4_26_conv33,
                                quant_zero_point3=z1_P4_26_conv33,
                                first_convs=0, coe_name=coe_name)
        s1_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.scale.npy')
        z1_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.zero_point.npy')
        s2_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.1.weight.scale.npy')
        z2_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.1.weight.zero_point.npy')
        s3_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.1.scale.npy')
        z3_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.1.zero_point.npy')
        print('P4_40:S', s3_P4_26_conv11, 'P4_40:z', z3_P4_26_conv11)
        bias_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.1.bias.npy')
        self.bias_P4_26_conv11 = bias_P4_26_conv11
        coe_name = '../yolo_head_P4_26_coe/hand_P4_26_conv11.coe'
        self.P4_26_conv11 = Conv2d_Q(quant_scale1=s1_P4_26_conv11, quant_zero_point1=z1_P4_26_conv11,
                                     quant_scale2=s2_P4_26_conv11,
                                     quant_zero_point2=z2_P4_26_conv11, quant_scale3=s3_P4_26_conv11,
                                     quant_zero_point3=z3_P4_26_conv11,
                                     first_convs=0, coe_name=coe_name)
        s1_P5_13_conv33 = tensorr('../para800_1028_new/conv_for_P5.conv.0.scale.npy')
        z1_P5_13_conv33 = tensorr('../para800_1028_new/conv_for_P5.conv.0.zero_point.npy')
        s2_P5_13_conv33 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.weight.scale.npy')
        z2_P5_13_conv33 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.weight.zero_point.npy')
        s3_P5_13_conv33 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.scale.npy')
        z3_P5_13_conv33 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.zero_point.npy')

        bias_f_P5_13_conv33 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.bias.npy')
        self.bias_P5_13_conv33 = bias_f_P5_13_conv33
        #
        coe_name = '../yolo_head_p5_13_coe/hand_P5_13_conv33.coe'
        self.P5_13_conv33 = Conv2d_Q(quant_scale1=s1_P5_13_conv33, quant_zero_point1=z1_P5_13_conv33,
                                     quant_scale2=s2_P5_13_conv33,
                                     quant_zero_point2=z2_P5_13_conv33, quant_scale3=s3_P5_13_conv33,
                                     quant_zero_point3=z3_P5_13_conv33,
                                     first_convs=0, coe_name=coe_name)
        s1_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.scale.npy')
        z1_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.zero_point.npy')
        s2_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.1.weight.scale.npy')
        z2_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.1.weight.zero_point.npy')
        s3_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.1.scale.npy')
        z3_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.1.zero_point.npy')
        # print('p5_20_z',z3_P5_13_conv11,'p5_20_s',s3_P5_13_conv11)
        # exit()
        bias_f_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.1.bias.npy')
        self.bias_P5_13_conv11 = bias_f_P5_13_conv11
        #
        coe_name = '../yolo_head_p5_13_coe/hand_P5_13_conv11.coe'
        self.P5_13_conv11 = Conv2d_Q(quant_scale1=s1_P5_13_conv11, quant_zero_point1=z1_P5_13_conv11,
                                     quant_scale2=s2_P5_13_conv11,
                                     quant_zero_point2=z2_P5_13_conv11, quant_scale3=s3_P5_13_conv11,
                                     quant_zero_point3=z3_P5_13_conv11,
                                     first_convs=0, coe_name=coe_name)

    def forward(self, x):
        model = torch.jit.load('../Epoch1-YOLOV4_tiny_quantization_post.pth')
        model.eval()
        # np.random.seed(3)
        # feature = np.random.random((1, 1, 640, 640))
        '''
        
        '''
        feature = get_picture()
        feature = feature / 255
        # feature = feature/255
        feature = feature.astype(np.float32)
        feature = torch.from_numpy(feature)
        with torch.no_grad():
            quant = model.quant(feature)
            feature_conv1 = quant
            conv1 = model.conv1(quant)
            feature_conv2 = conv1
            conv2 = model.conv2(conv1)
            feature_rs1_conv1 = conv2
            x = model.resblock_body1.conv1(conv2)
            route = x
            feature_rs1_conv2 = x
            x = model.resblock_body1.conv2(x)
            out_rs1_conv2 = x
            route1 = x
            feature_rs1_conv3 = x
            x = model.resblock_body1.conv3(x)
            out_rs1_conv3 = x
            feat = x
            feature_rs1_conv5 = x
            feat = model.resblock_body1.conv5(feat)
            x = model.resblock_body1.routeq0.cat([x, route1], dim=1)
            feature_rs1_conv4 = x
            x = model.resblock_body1.conv4(x)
            x = model.resblock_body1.maxpool(x)
            feature_rs2_conv1 = x
            x = model.resblock_body2.conv1(x)
            route = x
            feature_rs2_conv2 = x
            x = model.resblock_body2.conv2(x)
            out_rs2_conv2 = x
            route1 = x
            feature_rs2_conv3 = x
            x = model.resblock_body2.conv3(x)
            out_rs2_conv3 = x

            feat = x
            feature_rs2_conv5 = x
            feat = model.resblock_body2.conv5(feat)
            x = model.resblock_body2.routeq0.cat([x, route1], dim=1)
            feature_rs2_conv4 = x
            x = model.resblock_body2.conv4(x)
            x = model.resblock_body2.maxpool(x)
            feature_rs3_conv1 = x
            x = model.resblock_body3.conv1(x)
            route = x
            feature_rs3_conv2 = x
            x = model.resblock_body3.conv2(x)
            out_rs3_conv2 = x

            route1 = x
            feature_rs3_conv3 = x
            x = model.resblock_body3.conv3(x)
            out_rs3_conv3 = x
            feat = x
            feature_rs3_conv5 = x
            feat = model.resblock_body3.conv5(feat)
            rs3_conv5_out = feat
            x = model.resblock_body3.routeq0.cat([x, route1], dim=1)
            feature_rs3_conv4 = x
            x = model.resblock_body3.conv4(x)
            x = model.resblock_body3.maxpool(x)
            feature_conv3 = x
            x = model.conv3(x)
            feat2 = x
            feature_conv_for_p5 = feat2
            P5 = model.conv_for_P5(feat2)
            feature_upsample11 = P5
            # P5 upsample 为做完upsample之后的结果
            P5_Upsample = model.upsample(P5)


            P4 = model.route0.cat([P5_Upsample, rs3_conv5_out], 1)
            feature_yolo_head_P4_26 = P4
            out1 = model.yolo_headP4(P4)
            feature_yolo_head_P5_13 = P5
            out0 = model.yolo_headP5(P5)
        path1 = 'biasscaleshift1114.bin'
        file_name = 'mytest.txt'
        weight_conv1 = tensorr('../para800_1028_new/conv1.conv.0.weight.int.npy')
        weight_conv1_f = tensorr('../para800_1028_new/conv1.conv.0.weight.npy')
        weight_address = 1879048192  # 权重地址 初始HEX:7000 0000  下一层用上一层返回的地址
        computer_address = 16777216  # 计算地址  0100 0000
        # 方便dsp读取将第一层的地址用P5_13_conv11返回的地址，
        x = self.conv1(int(1886455872), int(0), feature_conv1, weight_conv1_f, weight_conv1,
                       self.bias_int_conv1, path1, stride=2, padding=1, block=0, operator='image_final')
        weight_conv2 = tensorr('../para800_1028_new/conv2.conv.0.weight.int.npy')
        weight_conv2_f = tensorr('../para800_1028_new/conv2.conv.0.weight.npy')
        x = self.conv2(weight_address, computer_address, feature_conv2, weight_conv2_f, weight_conv2,
                       self.bias_int_conv2, path1, stride=2, padding=1, block=0, operator='conv33')

        conv2_data = x
        weight_rs1_conv1 = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.weight.int.npy')
        weight_rs1_conv1_f = tensorr('../para800_1028_new/resblock_body1.conv1.conv.0.weight.npy')
        # exit()
        x = self.rs1_conv1(conv2_data[0], conv2_data[1], feature_rs1_conv1, weight_rs1_conv1_f, weight_rs1_conv1,
                           self.bias_rs1_int_conv1,
                           path1, stride=1, padding=1, block=0, operator='conv33')
        weight_rs1_conv2 = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.weight.int.npy')
        weight_rs1_conv2_f = tensorr('../para800_1028_new/resblock_body1.conv2.conv.0.weight.npy')
        x = self.rs1_conv2(x[0], x[1], feature_rs1_conv2, weight_rs1_conv2_f, weight_rs1_conv2, self.bias_rs1_int_conv2,
                           stride=1, padding=1, block=0, operator='conv33')
        # exit()
        rs1_conv2 = x
        weight_rs1_conv3 = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.weight.int.npy')
        weight_rs1_conv3_f = tensorr('../para800_1028_new/resblock_body1.conv3.conv.0.weight.npy')
        x = self.rs1_conv3(x[0], x[1], feature_rs1_conv3, weight_rs1_conv3_f, weight_rs1_conv3, self.bias_rs1_int_conv3,
                           path1, stride=1, padding=1, block=0, operator='conv33')

        x = self.rs1_cat1(x[0], x[1], out_rs1_conv3, weight_rs1_conv3_f, weight_rs1_conv3, self.bias_rs1_int_conv3,
                          path1, stride=1, padding=1, block=0, cat2_weight_address=rs1_conv2[1], cat2=out_rs1_conv2,
                          operator='concat')
        # weight_rs1_conv5 = tensorr('../para800_1028_new/resblock_body1.conv5.conv.0.weight.int.npy')
        # weight_rs1_conv5_f = tensorr('../para800_1028_new/resblock_body1.conv5.conv.0.weight.npy')
        # x = self.rs1_conv5(feature_rs1_conv5, weight_rs1_conv5_f, weight_rs1_conv5, self.bias_rs1_int_conv5, path1, stride=1, padding=1, block=0)
        weight_rs1_conv4 = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.weight.int.npy')
        weight_rs1_conv4_f = tensorr('../para800_1028_new/resblock_body1.conv4.conv.0.weight.npy')

        x = self.rs1_conv4(x[0], x[1], feature_rs1_conv4, weight_rs1_conv4_f, weight_rs1_conv4, self.bias_rs1_int_conv4,
                           path1,
                           stride=1, padding=0, block=0, operator='conv11')
        x = reshape(x[0], x[1], feature_rs1_conv4, operator="maxpool", filename=file_name)
        weight_rs2_conv1 = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.weight.int.npy')
        weight_rs2_conv1_f = tensorr('../para800_1028_new/resblock_body2.conv1.conv.0.weight.npy')
        x = self.rs2_conv1(x[0], x[1], feature_rs2_conv1, weight_rs2_conv1_f, weight_rs2_conv1, self.bias_rs2_int_conv1,
                           path1,
                           stride=1, padding=1, block=0, operator='conv33')

        weight_rs2_conv2 = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.weight.int.npy')
        weight_rs2_conv2_f = tensorr('../para800_1028_new/resblock_body2.conv2.conv.0.weight.npy')
        x = self.rs2_conv2(x[0], x[1], feature_rs2_conv2, weight_rs2_conv2_f, weight_rs2_conv2, self.bias_rs2_int_conv2,
                           path1,
                           stride=1, padding=1, block=0, operator='conv33')

        rs2_conv2 = x
        weight_rs2_conv3 = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.weight.int.npy')

        weight_rs2_conv3_f = tensorr('../para800_1028_new/resblock_body2.conv3.conv.0.weight.npy')
        x = self.rs2_conv3(x[0], x[1], feature_rs2_conv3, weight_rs2_conv3_f, weight_rs2_conv3, self.bias_rs2_int_conv3,
                           path1,
                           stride=1, padding=1, block=0, operator='conv33')
        x = self.rs2_cat1(x[0], x[1], out_rs2_conv3, weight_rs2_conv3_f, weight_rs2_conv3, self.bias_rs2_int_conv3,
                          path1, stride=1, padding=1, block=0, cat2_weight_address=rs2_conv2[1], cat2=out_rs2_conv2,
                          operator='concat')

        weight_rs2_conv4 = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.weight.int.npy')
        weight_rs2_conv4_f = tensorr('../para800_1028_new/resblock_body2.conv4.conv.0.weight.npy')
        x = self.rs2_conv4(x[0], x[1], feature_rs2_conv4, weight_rs2_conv4_f, weight_rs2_conv4, self.bias_rs2_int_conv4,
                           path1,
                           stride=1, padding=0, block=0, operator='conv11')
        x = reshape(x[0], x[1], feature_rs2_conv4, operator="maxpool", filename=file_name)

        weight_rs3_conv1 = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.weight.int.npy')
        weight_rs3_conv1_f = tensorr('../para800_1028_new/resblock_body3.conv1.conv.0.weight.npy')
        x = self.rs3_conv1(x[0], x[1], feature_rs3_conv1, weight_rs3_conv1_f, weight_rs3_conv1, self.bias_rs3_int_conv1,
                           path1,
                           stride=1, padding=1, block=0, operator='conv33')
        weight_rs3_conv2 = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.weight.int.npy')
        weight_rs3_conv2_f = tensorr('../para800_1028_new/resblock_body3.conv2.conv.0.weight.npy')
        x = self.rs3_conv2(x[0], x[1], feature_rs3_conv2, weight_rs3_conv2_f, weight_rs3_conv2, self.bias_rs3_int_conv2,
                           path1,
                           stride=1, padding=1, block=0, operator='conv33')
        rs3_conv2 = x
        weight_rs3_conv3 = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.weight.int.npy')
        weight_rs3_conv3_f = tensorr('../para800_1028_new/resblock_body3.conv3.conv.0.weight.npy')
        x = self.rs3_conv3(x[0], x[1], feature_rs3_conv3, weight_rs3_conv3_f, weight_rs3_conv3, self.bias_rs3_int_conv3,
                           path1,
                           stride=1, padding=1, block=0, operator='conv33')
        rs3_conv3 = x
        weight_rs3_conv5 = tensorr('../para800_1028_new/resblock_body3.conv5.conv.0.weight.int.npy')
        weight_rs3_conv5_f = tensorr('../para800_1028_new/resblock_body3.conv5.conv.0.weight.npy')

        x = self.rs3_conv5(x[0], x[1], feature_rs3_conv5, weight_rs3_conv5_f, weight_rs3_conv5, self.bias_rs3_int_conv5,
                           path1,
                           stride=1, padding=0, block=0, operator='conv11', add_write_address=67108864)
        rs3_conv5 = x

        x = self.rs3_cat1(rs3_conv3[0], rs3_conv3[1], out_rs3_conv3, weight_rs3_conv3_f, weight_rs3_conv3,
                          self.bias_rs3_int_conv3,
                          path1, stride=1, padding=1, block=0, cat2_weight_address=rs3_conv2[1], cat2=out_rs3_conv2,
                          operator='concat')

        weight_rs3_conv4 = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.weight.int.npy')
        weight_rs3_conv4_f = tensorr('../para800_1028_new/resblock_body3.conv4.conv.0.weight.npy')

        # x[1] + 117440512:将写入地址由12000000到20000000
        x = self.rs3_conv4(rs3_conv5[0], x[1], feature_rs3_conv4, weight_rs3_conv4_f, weight_rs3_conv4,
                           self.bias_rs3_int_conv4, path1,
                           stride=1, padding=0, block=0, operator='conv11', add_write_address=234881024)
        # exit()
        x = reshape(x[0], x[1], feature_rs3_conv4, operator="maxpool", filename=file_name)
        weight_conv3 = tensorr('../para800_1028_new/conv3.conv.0.weight.int.npy')
        weight_conv3_f = tensorr('../para800_1028_new/conv3.conv.0.weight.npy')
        # with open(file_name, 'a+') as fp:
        #     fp.write('----------------------------------------')

        x = self.conv3(x[0], x[1], feature_conv3, weight_conv3_f, weight_conv3, self.bias_int_conv3, path1, stride=1,
                       padding=1,
                       block=4, operator='conv33')

        weight_conv_forP5 = tensorr('../para800_1028_new/conv_for_P5.conv.0.weight.int.npy')
        weight_conv_forP5_f = tensorr('../para800_1028_new/conv_for_P5.conv.0.weight.npy')
        x = self.conv_forP5(x[0], x[1], feature_conv_for_p5, weight_conv_forP5_f, weight_conv_forP5,
                            self.bias_conv_forP5_int,
                            path1, stride=1, padding=0, block=0, operator='conv11')

        conv_forP5_out = x
        weight_upsample_conv11 = tensorr('../para800_1028_new/upsample.upsample.0.conv.0.weight.int.npy')
        weight_upsample_conv11_f = tensorr('../para800_1028_new/upsample.upsample.0.conv.0.weight.npy')
        x = self.upsample_conv11(x[0], x[1], feature_upsample11, weight_upsample_conv11_f, weight_upsample_conv11,
                                 self.bias_upsample_conv11, path1, stride=1, padding=0, block=0, operator='conv11')
        upsample_out = reshape(x[0], x[1], P5_Upsample, operator="upsample", filename=file_name)
        exit()

        big_cat = self.big_cat(upsample_out[0], upsample_out[1], P5_Upsample, weight_upsample_conv11_f,
                               weight_upsample_conv11,
                               self.bias_upsample_conv11,
                               path1, stride=1, padding=1, block=0, cat2_weight_address=rs3_conv5[1],
                               cat2=rs3_conv5_out, operator='concat', )

        weight_P4_26_conv33 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.weight.int.npy')
        weight_P4_26_conv33_f = tensorr('../para800_1028_new/yolo_headP4.yolo_head.0.conv.0.weight.npy')
        x = self.P4_26_conv33(big_cat[0], big_cat[1], feature_yolo_head_P4_26, weight_P4_26_conv33_f,
                              weight_P4_26_conv33,
                              self.bias_P4_26_conv33, path1, stride=1, padding=1, block=2, operator='conv33')

        weight_P4_26_conv11 = tensorr('../para800_1028_new/yolo_headP4.yolo_head.1.weight.int.npy')
        weight_P4_26_conv11_f = tensorr('../para800_1028_new/yolo_headP4.yolo_head.1.weight.npy')
        # 对于11卷积如果想要生成api结果给out_api=11

        p4_26 = torch.ones((1, 256, 40, 40))
        weight_P4_26 = torch.ones((24, 256, 1, 1))
        x = self.P4_26_conv11(x[0], x[1], p4_26, weight_P4_26, weight_P4_26, self.bias_P4_26_conv11,
                              path1, stride=1, padding=0, block=0, isleakrelu=1, operator='conv11')

        weight_P5_13_conv33 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.weight.int.npy')
        weight_P5_13_conv33_f = tensorr('../para800_1028_new/yolo_headP5.yolo_head.0.conv.0.weight.npy')

        x = self.P5_13_conv33(x[0], conv_forP5_out[1], feature_yolo_head_P5_13, weight_P5_13_conv33_f,
                              weight_P5_13_conv33,
                              self.bias_P5_13_conv33, path1, stride=1, padding=1, block=2, operator='conv33')

        weight_P5_13_conv11 = tensorr('../para800_1028_new/yolo_headP5.yolo_head.1.weight.int.npy')
        weight_P5_13_conv11_f = tensorr('../para800_1028_new/yolo_headP5.yolo_head.1.weight.npy')
        p5_13 = torch.ones((1, 512, 20, 20))
        weight_P5_13 = torch.ones((24, 512, 1, 1))
        x = self.P5_13_conv11(x[0], x[1], p5_13, weight_P5_13, weight_P5_13, self.bias_P5_13_conv11, path1,
                              stride=1, padding=0, block=0, isleakrelu=1, operator='conv11')


if __name__ == "__main__":
    # 权重的起始地址,对应的16进制为70000000
    # weight_address = 1879048192
    model = QuantizableYolo_tiny()(1)
