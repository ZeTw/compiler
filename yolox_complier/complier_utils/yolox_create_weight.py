# -*- coding: UTF-8 -*-
# encoding=utf-8
import torch
import torch.nn as nn
import numpy as np
import cv2
from ins_conv_new_831 import *

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


def weight_4to8(weight_stem_conv):
    weight_shape = weight_stem_conv.shape
    channel_out_num = weight_shape[0]
    if weight_shape[1] == 4:
        channel_in_num = weight_shape[1] * 2
    else:
        channel_in_num = weight_shape[1] + 4
    new_feature = np.zeros((channel_out_num, channel_in_num, weight_shape[2], weight_shape[3]), dtype=np.uint8)
    weight_stem_conv_0 = add_weight_channel(new_feature, weight_stem_conv, weight_shape)
    weight_stem_conv_0 = torch.from_numpy(weight_stem_conv_0)
    return weight_stem_conv_0


# def gen_M_N1X1(S1, S2, S3): #没用
#     M = (S1 * S2) / S3   #M定义为(S1 * S2) / S3
#     M = M.numpy()        #转换成numpy
#     daxiao = 32
#     SCALE = np.zeros(daxiao, dtype=np.uint32, order='C')  #order:可选参数，c代表与c语言类似，行优先；F代表列优先
#     N_REAL = np.zeros(daxiao, dtype=np.uint32, order='C')
#     for i, ii in enumerate(M):  #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
#
#         while not (ii >= 0.5 and ii <= 1.0):  #左移到（0.5，1）
#             ii *= 2
#         pass
#         mmmm = ii * (2 ** 32)
#         SCALE[i] = mmmm.astype(np.uint32)
#     for i, ii in enumerate(M):
#         N_REAL[i] = round(math.log(SCALE[i] / ii, 2)) - 32 - 1
#     return SCALE, N_REAL


def gen_B(S1, S2, S3):  # 第一组  β即N_REAL  +++++求shitf++++
    M = (S1 * S2) / S3
    M = M.numpy()

    daxiao = S2.shape[0]  # 第一层权重的shape[0]是32 shape[0]表示行数 是一维大小位32的列向量
    SCALE = np.zeros(daxiao, dtype=np.uint32, order='C')  # 相当于32个输出通道 每个对应一组shift
    N_REAL = np.zeros(daxiao, dtype=np.uint32, order='C')
    for i, ii in enumerate(M):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

        while not (ii >= 0.5 and ii <= 1.0):  # 左移到（0.5，1） 左移一次相当于*2
            ii *= 2
        pass
        mmmm = ii * (2 ** 32)  # 乘2^32

        SCALE[i] = mmmm.astype(np.int32)

    for i, ii in enumerate(M):
        N_REAL[i] = round(math.log(SCALE[i] / ii, 2)) - 32 - 1  # fpga加了1这里要减1,  β值也是m维 相当于存的mmmm

    return N_REAL


def gen_M_N(S1, S2, S3):  # 第二组求M'即M  ++++++返回scale和shift+++++
    daxiao = S2.shape[0]
    M = np.zeros(daxiao, dtype=np.uint32, order='C')
    N_REAL = gen_B(S1, S2, S3)
    M = np.zeros(S2.shape[0])
    for i, ii in enumerate(M):
        M[i] = (torch.round((S1 * S2[i]) / S3 * (2 ** (32 + N_REAL[i] + 1)))).numpy()  # s1s2/s3 *2^(32+β+1)
    M = M.astype(np.uint32)
    # exit()
    return M, N_REAL


# r_b=s1*s2*q_b
# r_b是量化前的bias,q_b是量化后的bias
def gen_int_bias(s1, s2, bias_float):  # 求bias/s1s2即bb
    aa = bias_float / s1
    # print(bias_float)
    # exit()
    bb = torch.div(aa, s2)  # 对应元素做除法

    # for i, m in enumerate(bb):
    #     bb[i] = round(m.item())
    # bias = bb.int()
    return bb


def gen_M(s1, s2, s3):
    aa = s1 * s2
    M = aa / s3
    return M


# def new_bias(z1, q2, bias):
#     q2 = torch.as_tensor(q2, dtype=torch.int32)
#     bias1 = -z1 * q2
#     shape = bias1.shape
#     n_bias = np.zeros(shape[0], dtype=np.int32, order='C')
#     for m in range(shape[0]):
#         n_bias[m] = bias1[m, :, :, :].sum()
#         n_bias[m] = n_bias[m] + bias[m]
#     return n_bias
def new_bias(z1, q2, bias):  # 求最终的bias=bias/s1s2-q2z1
    q2 = q2.type(torch.float64)
    bias1 = z1 * q2
    shape = bias1.shape
    n_bias = np.zeros(shape[0], dtype=np.float64, order='C')
    for m in range(shape[0]):  # bias1的维度是M C K K 将C K K 做累加 变成M维
        n_bias[m] = bias1[m, :, :, :].sum()  # 从第一组开始一直有m组，m是输出通道数
        # print()
        n_bias[m] = (bias[m] - n_bias[m])  # 做减法
    # print(n_bias) 第一层n_bias是一维32个
    # exit()
    daxiao = shape[0]  # 第一层是32
    SCALE = np.zeros(daxiao, dtype=np.float64, order='C')
    # N_REAL = np.zeros(daxiao, dtype=np.float32, order='C')
    N_REAL = []
    for i, ii in enumerate(n_bias):  # i和ii就是从n_bias中取值
        index = 0

        while not (abs(ii) >= (2 ** 23) and abs(ii) <= (2 ** 24)):
            if index >= 16:  # fpga里面最多移动16位,所有成到16就停止了,这样精度也够了
                break
            else:
                ii *= 2
                index = index + 1

        N_REAL.append(index)
        SCALE[i] = round(ii)
    out_bias = []
    for index in range(shape[0]):
        data_integer_old = ('{:024b}'.format(int(SCALE[index]) & 0xffffff))  # {:024b} 24位2二进制不足补0；& 0xffffff按位与
        n = N_REAL[index]
        symbol = '0'
        if n_bias[index] < 0:  # 符号位
            symbol = '1'
        elif n_bias[index] > 0:
            symbol = '0'
        data_integer = data_integer_old[8:]
        data_decimal = '{:07b}'.format(int(n))
        out_bias1 = symbol + str(data_decimal) + str(data_integer_old)  # 1bit+7bit+24bit
        a = int(out_bias1, 2)  # 转成int型 out_bias1为二进制 ；a是十进制
        out_bias.append(a)  # 一个一个写入out_bias
    # print(out_bias)
    # exit()
    return out_bias


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

    def forward(self, x, q_weight, bias, path1, coee=1, block=0, inchannel=8):

        coe_name = self.coe_name
        bias = np.array(bias.data.cpu().numpy(),
                        dtype=np.float64)  # 将tensor转成numpy;  bias.data.cpu()把bias.data放入cpu；a.data.cpu().numpy()把tensor转换成numpy的格式
        SCALE, N_REAL = gen_M_N(self.quant_scale1, self.quant_scale2, self.quant_scale3)
        bias = new_bias(self.quant_zero_point1, q_weight, bias)
        # bias = bias.astype(np.uint32)
        q_weight = np.array(q_weight.data.cpu().numpy(), dtype=np.int8)
        # print("q_weight.shape=", q_weight.shape)  # 输出每一层的权重
        shape = q_weight.shape  # 第一层[32,1,3,3]
        if (shape[1] % 8 != 0):  # 变成8的倍数 FPGA必须要8的倍数  shape[0]是输出通道数 不足补0；（kernel_num卷积核的数量即输出通道数）
            channel_in_num = shape[1] + 8 - shape[1] % 8
        else:
            channel_in_num = shape[1]

        # kernel_num = shape[0]  # shape[0]是输出通道数
        if (shape[0] % 8 != 0):  # 变成8的倍数 FPGA必须要8的倍数  shape[0]是输出通道数 不足补0；（kernel_num卷积核的数量即输出通道数）
            kernel_num = shape[0] + 8 - shape[0] % 8
        else:
            kernel_num = shape[0]

        new_weig = np.zeros((kernel_num, channel_in_num, shape[2], shape[3]))  # 全0  存储新权重
        # new_weight是补0后的四维权重
        new_weight = add_weight_channel(new_weig, q_weight, shape)  # 补0   因为kernel_num变了多出来的补成0，用旧权重覆盖。
        new_shape = new_weight.shape  # new_shape是变成8的倍数后的权重的shape

        daxiao = new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3]  # 四个维度总共有多少点
        # print("有多少卷积点daxiao=", daxiao)
        weight = np.zeros(daxiao, dtype=np.uint8, order='C')  # 生成右daxiao个的ny数组(一维！！！)；c代表与c语言类似，行优先；F代表列优先
        weight_name = "../weight/yolox_weight682.dat"  # 生成权重文件的名字
        # * ========image层===================
        # if (shape[1] % 8 != 0):  # 如果是输出通道第一层，第一层写入256bit指令;只有第一层输出通道是1，1%8!=0,其他输出通道都是8的倍数
        #     # ============第一层先写入256bit的指令============
        #     feature_size = 640  # 图片大小640*640
        #     # 计算reg4
        #     # 输入图片宽高数
        #     feature_size_new = '{:011b}'.format(int(feature_size))  # 存储成二进制 因为要拼成reg4     是format()后面的内容，填入大括号中；b是二进制不足11位补0
        #     # 卷积操作之后宽高数,stride等于1
        #     outPictureSize = int((feature_size - 3 + 2 * 1) / 1 + 1)  # 输出图片大小   N=(W-F+2P)/S+1  N：输出大小 W：输入大小 W：输入大小 P：填充值的大小 S：步长大小
        #     outPictureSize = '{:011b}'.format(int(outPictureSize))  # 存成二进制
        #     # 图片的输出通道数
        #     out_channel = shape[0]  # 图片的输出通道数=权重的输出通道数
        #     out_channel = '{:08b}'.format(int(out_channel))  # 存成二进制

        #     out_reg4 = str(feature_size_new) + str(outPictureSize) + str('00') + str(out_channel)  # reg4是拼接的 reg5 reg6 reg7是给FPGA的
        #     # 计算reg5
        #     out_reg5 = str('11001000000000000000000000000000')  # reg5写死的
        #     # 计算reg6
        #     out_reg6 = reg6_leaky(self.quant_scale3)  # reg6有一个修正表
        #     # 计算reg7
        #     out_reg7 = '{:08b}'.format(int(self.quant_zero_point3))
        #     out_reg7 = str(out_reg7) + str('000000000000000000000000')  # reg7是一个拼接
        #     # 指令前面加128个0凑够256bit （凑256bit）
        #     weight_ins = '0'
        #     for index_add_zero in range(127):  # weight_ins是128个0
        #         weight_ins += '0'
        #     weight_ins = weight_ins + str(out_reg7) + str(out_reg6) + str(out_reg5) + str(out_reg4)
        #     weight_ins = int(weight_ins, 2)  # weight_ins是二进制，转成int十进制
        #     weight_ins = '{:064x}'.format(int(weight_ins))  # 转成十六进制64位不足补0, 给FPGA的是十六进制
        #     out = []
        #     # print(weight_ins)
        #     with open(weight_name, "a+") as fp:  # a+读写  追加模式
        #         for r in range(len(weight_ins)):
        #             out.append(weight_ins[63 - r])  # append在列表最后添加元素 相当于倒着写
        #             if len(out) == 8:  # append 8个,再reverse()，再换行
        #                 out.reverse()
        #                 fp.write('0x')  # 先写0x
        #                 for m in out:
        #                     # m = m.item()
        #                     fp.write(m)  # 逐元素写入m
        #                 fp.write('\n')
        #                 out = []
        #     # =========================================
        #     # ============将权重补0存成一维数组============
        #     a = 0
        #     new_image_weight = np.zeros(1024, dtype=np.uint8,order='C')  # m输出通道数 c输入通道数 k k；32*1*3*3=288个卷积点;一行写72bit 8bit一个点 9个卷积点;288/9=32行要写32行 23个0 + 9个卷积点=一行32个数  32行*32个数=1024
        #     get_weight2(new_weight, new_shape,
        #                 weight)  # 补0后的四维new_weight存入---->weight ;int8的new_weight存入uint8的weight，小数会变成补码的十进制
        #     add_zero = np.zeros(23, dtype=np.uint8)  # 补23个0
        #     shape_new = weight.shape[0] + 1  # +1是因为 range要从1开始
        #     print(weight.shape[0])
        #     # 第一层权重补0补到256bit,一行从72到256
        #     for index in range(1, shape_new):  # 先写指令再补0
        #         if index != 0 and index % 9 == 0:  # 从9开始 18 27 ...
        #             new_image_weight[int(index / 9 - 1) * 32:int(index / 9) * 32] = np.append(weight[index - 9:index],add_zero)  # np.append(weight[index - 9:index],add_zero)  先写9个数补23个0就是一行；第一行是[0,32)

        #     weight = new_image_weight  # 补0之后weight 是大小为1024的一维数组
        #     # print(weight)
        #     # =========================================

        # 如果不是第一层，
        if block != 0 and coee == 1:  # 如果分块；block!=0分块 coee=1才写文件
            shape_block = []
            daxiao_new = []
            # 如果将weight分成4块  例如256 384 3 3 分成四个64 384 3 3  让每个分别进行kkmc生成coe
            for shape_num in range(block):  # 分块操作
                print(int(new_shape[0] / block * (
                        1 + shape_num)))  # 输出通道数m/block就是每块多少个 * （1+shaoe_num） 如果分成四块：第一次循环0到64 第二次64到128  第三次128到192 第四次192到256
                shape_block.append(int(new_shape[0] / block * (1 + shape_num)))  # 分四块是（64，128，192，256）
                block_shape = (shape_block[0], new_shape[1], new_shape[2], new_shape[
                    3])  # m c k k 如果分成四块：第一次循环是（0~64，3，3，3）第二次（64~128，3，3，3） 第三次（128~192，3，3，3） 第四次（192~256，3，3，3）
                daxiao_new.append(shape_block[shape_num] * new_shape[1] * new_shape[2] * new_shape[3])  # 分块的大小
                # print(new_weight[:daxiao_new[shape_num], :, :, :], block_shape, weight[:daxiao_new[shape_num]])
                if shape_num == 0:
                    get_weight(new_weight[:daxiao_new[shape_num], :, :, :], block_shape, weight[:daxiao_new[shape_num]],
                               inchannel)  # 分四块；开始0到64  shape_num=0
                else:
                    get_weight(new_weight[shape_block[shape_num - 1]:shape_block[shape_num], :, :, :], block_shape,
                               weight[daxiao_new[shape_num - 1]:daxiao_new[shape_num]],
                               inchannel)  # 分四块；64到128 shape_num=1;128到192 shape_num=2;192到256 shape_num=3
            # print(weight.shape)
            # exit()
            for index in range(block):  # 写coe文件 每次写一块的weight+bias+SCALE+N_REAL
                coe_name = coe_name + 'block' + str(index) + '.coe'
                out = []
                with open(weight_name, "a+") as fp:  # 写入权重

                    for r in weight[index * daxiao_new[0]:daxiao_new[0] * (index + 1)]:
                        out.append(r)
                        if len(out) == 4:  # 每次写入四个数，一个数是两位十六进制,一个十六进制的数是4bit，一行32bit
                            out.reverse()
                            fp.write('0x')
                            for m in out:
                                m = m.item()
                                fp.write('%02x' % m)
                            fp.write('\n')
                            out = []

                with open(weight_name, "a+") as fp:  # 写入bias   #bias是一维 每块有shape_block[0]个
                    for r in bias[
                             index * shape_block[0]:shape_block[0] * (index + 1)]:  # 分量块 第一次是[0,128]  第二次是[128,256]
                        out.append(r)
                        fp.write('0x')
                        if len(out) == 1:  # bias是一个8位十六进制的数，32bit
                            out.reverse()
                            for m in out:
                                fp.write('%08x' % int(m))
                            fp.write('\n')
                            out = []

                with open(weight_name, "a+") as fp:  # 写入SCALE
                    for r in SCALE[index * shape_block[0]:shape_block[0] * (index + 1)]:
                        out.append(r)
                        fp.write('0x')
                        if len(out) == 1:  # SCALE是一个8位十六进制的数，32bit
                            out.reverse()
                            for m in out:
                                m = m.item()
                                fp.write('%08x' % m)

                            fp.write('\n')
                            out = []

                with open(weight_name, "a+") as fp:  # 写入N_REAL
                    for r in N_REAL[index * shape_block[0]:shape_block[0] * (index + 1)]:
                        out.append(r)
                        fp.write('0x')
                        if len(out) == 1:  # N_REAL是一个8位十六进制的数，32bit
                            out.reverse()
                            for m in out:
                                m = m.item()
                                fp.write('%08x' % m)
                            fp.write('\n')
                            out = []

                coe_name = self.coe_name

        else:  # 如果不分块
            get_weight(new_weight, new_shape, weight, inchannel)  # 做8入8出
        if new_weight.shape[0] != shape[0]:  # bias SCALE NREAL进行补0操作
            new_dimen_bias = np.zeros(kernel_num, dtype=np.uint32)
            new_dimen_SCALE = np.zeros(kernel_num, dtype=np.uint32)
            new_dimen_NREAL = np.zeros(kernel_num, dtype=np.uint32)
            bias = get_add_bias(new_dimen_bias, shape[0], bias)  # 覆盖
            SCALE = get_add_SCALE(new_dimen_SCALE, shape[0], SCALE)
            N_REAL = get_add_NREAL(new_dimen_NREAL, shape[0], N_REAL)

        coe_name = self.coe_name
        if block == 0 and coee == 1:  # 不分块就直接写coe文件
            out = []
            with open(weight_name, "a+") as fp:  # 写入权重

                for r in weight:
                    out.append(r)
                    if len(out) == 4:
                        out.reverse()
                        fp.write('0x')
                        for m in out:
                            m = m.item()
                            fp.write('%02x' % m)
                        fp.write('\n')
                        out = []

            with open(weight_name, "a+") as fp:  # 写入bias
                for r in bias:
                    # for index in range(len(bias)):
                    out.append(r)
                    fp.write('0x')
                    if len(out) == 1:
                        out.reverse()
                        for m in out:
                            fp.write('%08x' % int(m))
                        fp.write('\n')
                        out = []
            # print(bias)
            with open(weight_name, "a+") as fp:  # 写入SCALE
                for r in SCALE:
                    fp.write('0x')
                    out.append(r)
                    if len(out) == 1:
                        out.reverse()
                        for m in out:
                            m = m.item()
                            fp.write('%08x' % m)
                        fp.write('\n')
                        out = []
            # fp.write('==========================')
            with open(weight_name, "a+") as fp:  # 写入N_REAL
                for r in N_REAL:
                    fp.write('0x')
                    out.append(r)
                    if len(out) == 1:
                        out.reverse()
                        for m in out:
                            m = m.item()
                            fp.write('%08x' % m)
                        fp.write('\n')
                        out = []


def get_add_bias(new, shape, old):  # 补0
    for kernel_num in range(shape):
        new[kernel_num] = old[kernel_num]
    return new


def get_add_SCALE(new, shape, old):  # 补0
    for kernel_num in range(shape):
        new[kernel_num] = old[kernel_num]
    return new


def get_add_NREAL(new, shape, old):  # 补0
    for kernel_num in range(shape):
        new[kernel_num] = old[kernel_num]
    return new


def get_weight(new_weight, shape, weight, inchannel):  # 八入八出操作 输入通道inchannel不固定 输出通道outchannel为8
    j = 0
    # shift_num = math.sqrt(inchannel)  #平方根
    # shift_num = int(shift_num)
    # print(new_weight.shape)
    # print(new_weight)
    shift_num = 0
    for index in range(inchannel):
        if (inchannel == (1 << index)):  # index不停左移一位直到与inchannel相等 来判断移了几位   2的shift_num次方=inchannel
            shift_num = index
            break
    for i in range(shape[2]):  # kkmc
        for ii in range(shape[3]):
            for kernel_times in range(shape[0] >> 3):  # >>右移三位 因为输出通道outchannel默认为8
                for channel_in_times in range(shape[1] >> shift_num):  # 右移shift_num
                    for iii in range(8):  # shape[0] >> 3 右移了3次即2^3 要补8次
                        for iiii in range(
                                inchannel):  # shape[1] >> shift_num 右移了shift_num即2^shift_num 要补inchannel次(2^shift_num =inchannel)
                            # print('++++++++++++++++++')
                            weight[j] = new_weight[kernel_times * 8 + iii][channel_in_times * inchannel + iiii][i][ii]
                            j += 1
    return weight


def add_weight_channel(new_weig, weig, shape):  # 补0 把weig存入new_weig；new_weig是全0，
    for kernel_num in range(shape[0]):
        for channel_in_num in range(shape[1]):
            for row in range(shape[2]):
                for col in range(shape[3]):
                    new_weig[kernel_num][channel_in_num][row][col] = weig[kernel_num][channel_in_num][row][col]
    return new_weig


def tensorr(x):
    tensor_py = torch.from_numpy(np.load(x))  # 创建tensor
    return tensor_py


def get_weight2(new_weight, shape, weight):  # 四维权重写成一维
    j = 0
    for kernel_times in range(shape[0]):
        for channel_in_times in range(shape[1]):
            for i in range(shape[2]):
                for ii in range(shape[3]):
                    # print('++++++++++++++++++')
                    weight[j] = new_weight[kernel_times][channel_in_times][i][ii]
                    j += 1
    return weight


class QuantizableYolo_tiny(nn.Module):

    def __init__(self, img_size=416):
        super(QuantizableYolo_tiny, self).__init__()
        ###############################
        s1_stem_conv = tensorr('../para_682/backbone.backbone.stem.csp0.scale.npy')
        z1_stem_conv = tensorr('../para_682/backbone.backbone.stem.csp0.zero_point.npy')
        # stem_conv 权重
        s2_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.weight.scale.npy')
        z2_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.weight.zero_point.npy')
        # stem的输出
        s3_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.scale.npy')
        z3_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.zero_point.npy')

        bias_f_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.bias.npy')
        bias_int_stem_conv = gen_int_bias(s1_stem_conv, s2_stem_conv, bias_f_stem_conv)
        self.bias_int_stem_conv = bias_int_stem_conv
        coe_name = '../data1_coe/weight_stem_conv_leak.coe'
        self.stem_conv = Conv2d_Q(quant_scale1=s1_stem_conv, quant_zero_point1=z1_stem_conv, quant_scale2=s2_stem_conv,
                                  quant_zero_point2=z2_stem_conv, quant_scale3=s3_stem_conv,
                                  quant_zero_point3=z3_stem_conv, first_convs=0, coe_name=coe_name)

        s1_dark2_0_conv = tensorr("../para_682/backbone.backbone.stem.conv.conv.scale.npy")
        z1_dark2_0_conv = tensorr("../para_682/backbone.backbone.stem.conv.conv.zero_point.npy")
        s2_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.weight.scale.npy")
        z2_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.weight.zero_point.npy")
        s3_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.scale.npy")
        z3_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.zero_point.npy")
        bias_f_dark2_0_conv = tensorr("../para_682/backbone.backbone.dark2.0.conv.bias.npy")
        self.bias_int_dark2_0_conv = gen_int_bias(s1_dark2_0_conv, s2_dark2_0_conv, bias_f_dark2_0_conv)
        coe_name = '../data1_coe/weight_dark2_0_conv_leak.coe'
        self.dark2_0_conv = Conv2d_Q(quant_scale1=s1_dark2_0_conv, quant_zero_point1=z1_dark2_0_conv,
                                     quant_scale2=s2_dark2_0_conv,
                                     quant_zero_point2=z2_dark2_0_conv, quant_scale3=s3_dark2_0_conv,
                                     quant_zero_point3=z3_dark2_0_conv, first_convs=0, coe_name=coe_name)

        s1_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.0.conv.scale.npy")
        z1_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.0.conv.zero_point.npy")
        s2_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.weight.scale.npy")
        z2_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.weight.zero_point.npy")
        s3_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.scale.npy")
        z3_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.zero_point.npy")
        bias_f_dark2_1_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.bias.npy")
        self.bias_int_dark2_1_conv1 = gen_int_bias(s1_dark2_1_conv1, s2_dark2_1_conv1, bias_f_dark2_1_conv1)
        coe_name = '../data1_coe/out_hand_dark2_1_conv1_leak.coe'
        self.dark2_1_conv1 = Conv2d_Q(quant_scale1=s1_dark2_1_conv1, quant_zero_point1=z1_dark2_1_conv1,
                                      quant_scale2=s2_dark2_1_conv1,
                                      quant_zero_point2=z2_dark2_1_conv1, quant_scale3=s3_dark2_1_conv1,
                                      quant_zero_point3=z3_dark2_1_conv1, first_convs=0, coe_name=coe_name)

        s1_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.0.conv.scale.npy")
        z1_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.0.conv.zero_point.npy")
        s2_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.weight.scale.npy")
        z2_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.weight.zero_point.npy")
        s3_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.scale.npy")
        z3_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.zero_point.npy")
        bias_f_dark2_1_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.conv2.conv.bias.npy")
        self.bias_int_dark2_1_conv2 = gen_int_bias(s1_dark2_1_conv2, s2_dark2_1_conv2, bias_f_dark2_1_conv2)
        coe_name = '../data1_coe/weight_dark2_1_conv2_leak.coe'
        self.dark2_1_conv2 = Conv2d_Q(quant_scale1=s1_dark2_1_conv2, quant_zero_point1=z1_dark2_1_conv2,
                                      quant_scale2=s2_dark2_1_conv2,
                                      quant_zero_point2=z2_dark2_1_conv2, quant_scale3=s3_dark2_1_conv2,
                                      quant_zero_point3=z3_dark2_1_conv2, first_convs=0, coe_name=coe_name)

        s1_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.scale.npy")
        z1_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.conv1.conv.zero_point.npy")
        s2_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.weight.scale.npy")
        z2_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.weight.zero_point.npy")
        s3_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.scale.npy")
        z3_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.zero_point.npy")
        bias_f_dark2_1_m0_conv1 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.bias.npy")
        self.bias_int_dark2_1_m0_conv1 = gen_int_bias(s1_dark2_1_m0_conv1, s2_dark2_1_m0_conv1, bias_f_dark2_1_m0_conv1)
        coe_name = '../data1_coe/weight_dark2_1_m0_conv1_leak.coe'
        self.dark2_1_m0_conv1 = Conv2d_Q(quant_scale1=s1_dark2_1_m0_conv1, quant_zero_point1=z1_dark2_1_m0_conv1,
                                         quant_scale2=s2_dark2_1_m0_conv1,
                                         quant_zero_point2=z2_dark2_1_m0_conv1, quant_scale3=s3_dark2_1_m0_conv1,
                                         quant_zero_point3=z3_dark2_1_m0_conv1, first_convs=0, coe_name=coe_name)

        s1_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.scale.npy")
        z1_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.zero_point.npy")
        s2_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.weight.scale.npy")
        z2_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.weight.zero_point.npy")
        s3_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.scale.npy")
        z3_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.zero_point.npy")
        bias_f_dark2_1_m0_conv2 = tensorr("../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.bias.npy")
        self.bias_int_dark2_1_m0_conv2 = gen_int_bias(s1_dark2_1_m0_conv2, s2_dark2_1_m0_conv2, bias_f_dark2_1_m0_conv2)
        coe_name = '../data1_coe/weight_dark2_1_m0_conv2_leak.coe'
        self.dark2_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark2_1_m0_conv2, quant_zero_point1=z1_dark2_1_m0_conv2,
                                         quant_scale2=s2_dark2_1_m0_conv2,
                                         quant_zero_point2=z2_dark2_1_m0_conv2, quant_scale3=s3_dark2_1_m0_conv2,
                                         quant_zero_point3=z3_dark2_1_m0_conv2, first_convs=0, coe_name=coe_name)

        s1_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.csp1.scale.npy")
        z1_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.csp1.zero_point.npy")
        s2_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.weight.scale.npy")
        z2_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.weight.zero_point.npy")
        s3_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.scale.npy")
        z3_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.zero_point.npy")
        bias_f_dark2_1_conv3 = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.bias.npy")
        self.bias_int_dark2_1_conv3 = gen_int_bias(s1_dark2_1_conv3, s2_dark2_1_conv3, bias_f_dark2_1_conv3)
        coe_name = '../data1_coe/weight_dark2_1_conv3_leak.coe'
        self.dark2_1_conv3 = Conv2d_Q(quant_scale1=s1_dark2_1_conv3, quant_zero_point1=z1_dark2_1_conv3,
                                      quant_scale2=s2_dark2_1_conv3,
                                      quant_zero_point2=z2_dark2_1_conv3, quant_scale3=s3_dark2_1_conv3,
                                      quant_zero_point3=z3_dark2_1_conv3, first_convs=0, coe_name=coe_name)

        s1_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.scale.npy")
        z1_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark2.1.conv3.conv.zero_point.npy")
        s2_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.weight.scale.npy")
        z2_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.weight.zero_point.npy")
        s3_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.scale.npy")
        z3_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.zero_point.npy")
        bias_f_dark3_0_conv = tensorr("../para_682/backbone.backbone.dark3.0.conv.bias.npy")
        self.bias_int_dark3_0_conv = gen_int_bias(s1_dark3_0_conv, s2_dark3_0_conv, bias_f_dark3_0_conv)
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
        self.bias_int_dark3_1_conv1 = gen_int_bias(s1_dark3_1_conv1, s2_dark3_1_conv1, bias_f_dark3_1_conv1)
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
        self.bias_int_dark3_1_conv2 = gen_int_bias(s1_dark3_1_conv2, s2_dark3_1_conv2, bias_f_dark3_1_conv2)
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
        self.bias_int_dark3_1_m0_conv1 = gen_int_bias(s1_dark3_1_m0_conv1, s2_dark3_1_m0_conv1, bias_f_dark3_1_m0_conv1)
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
        self.bias_int_dark3_1_m0_conv2 = gen_int_bias(s1_dark3_1_m0_conv2, s2_dark3_1_m0_conv2, bias_f_dark3_1_m0_conv2)
        coe_name = '../data1_coe/out_hand_dark3_1_m0_conv2_leak.coe'
        self.dark3_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark3_1_m0_conv2, quant_zero_point1=z1_dark3_1_m0_conv2,
                                         quant_scale2=s2_dark3_1_m0_conv2,
                                         quant_zero_point2=z2_dark3_1_m0_conv2, quant_scale3=s3_dark3_1_m0_conv2,
                                         quant_zero_point3=z3_dark3_1_m0_conv2, coe_name=coe_name)

        s1_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.scale.npy")
        z1_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.0.csp.zero_point.npy")
        s2_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.weight.scale.npy")
        z2_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.weight.zero_point.npy")
        s3_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.scale.npy")
        z3_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.zero_point.npy")
        bias_f_dark3_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.bias.npy")
        self.bias_int_dark3_1_m1_conv1 = gen_int_bias(s1_dark3_1_m1_conv1, s2_dark3_1_m1_conv1, bias_f_dark3_1_m1_conv1)
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
        self.bias_int_dark3_1_m1_conv2 = gen_int_bias(s1_dark3_1_m1_conv2, s2_dark3_1_m1_conv2, bias_f_dark3_1_m1_conv2)
        coe_name = '../data1_coe/out_hand_dark3_1_m1_conv2_leak.coe'
        self.dark3_1_m1_conv2 = Conv2d_Q(quant_scale1=s1_dark3_1_m1_conv2, quant_zero_point1=z1_dark3_1_m1_conv2,
                                         quant_scale2=s2_dark3_1_m1_conv2,
                                         quant_zero_point2=z2_dark3_1_m1_conv2, quant_scale3=s3_dark3_1_m1_conv2,
                                         quant_zero_point3=z3_dark3_1_m1_conv2, coe_name=coe_name)

        s1_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.scale.npy")
        z1_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.1.csp.zero_point.npy")
        s2_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.weight.scale.npy")
        z2_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.weight.zero_point.npy")
        s3_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.scale.npy")
        z3_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.zero_point.npy")
        bias_f_dark3_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.bias.npy")
        self.bias_int_dark3_1_m2_conv1 = gen_int_bias(s1_dark3_1_m2_conv1, s2_dark3_1_m2_conv1, bias_f_dark3_1_m2_conv1)
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
        self.bias_int_dark3_1_m2_conv2 = gen_int_bias(s1_dark3_1_m2_conv2, s2_dark3_1_m2_conv2, bias_f_dark3_1_m2_conv2)
        coe_name = '../data1_coe/out_hand_dark3_1_m2_conv2_leak.coe'
        self.dark3_1_m2_conv2 = Conv2d_Q(quant_scale1=s1_dark3_1_m2_conv2, quant_zero_point1=z1_dark3_1_m2_conv2,
                                         quant_scale2=s2_dark3_1_m2_conv2,
                                         quant_zero_point2=z2_dark3_1_m2_conv2, quant_scale3=s3_dark3_1_m2_conv2,
                                         quant_zero_point3=z3_dark3_1_m2_conv2, coe_name=coe_name)

        s1_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.csp1.scale.npy")
        z1_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.csp1.zero_point.npy")
        s2_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.weight.scale.npy")
        z2_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.weight.zero_point.npy")
        s3_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.scale.npy")
        z3_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.zero_point.npy")
        bias_f_dark3_1_conv3 = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.bias.npy")
        self.bias_int_dark3_1_conv3 = gen_int_bias(s1_dark3_1_conv3, s2_dark3_1_conv3, bias_f_dark3_1_conv3)
        coe_name = '../data1_coe/out_hand_dark3_1_conv2_leak.coe'
        self.dark3_1_conv3 = Conv2d_Q(quant_scale1=s1_dark3_1_conv3, quant_zero_point1=z1_dark3_1_conv3,
                                      quant_scale2=s2_dark3_1_conv3,
                                      quant_zero_point2=z2_dark3_1_conv3, quant_scale3=s3_dark3_1_conv3,
                                      quant_zero_point3=z3_dark3_1_conv3, coe_name=coe_name)

        # = == == == == == == == == == == == == == == == == == == dark4 == == == == == == == == == == == == == == == == == == == == == == == == == =
        s1_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.scale.npy")
        z1_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark3.1.conv3.conv.zero_point.npy")
        s2_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.weight.scale.npy")
        z2_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.weight.zero_point.npy")
        s3_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.scale.npy")
        z3_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.zero_point.npy")
        bias_f_dark4_0_conv = tensorr("../para_682/backbone.backbone.dark4.0.conv.bias.npy")
        self.bias_int_dark4_0_conv = gen_int_bias(s1_dark4_0_conv, s2_dark4_0_conv, bias_f_dark4_0_conv)
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
        self.bias_int_dark4_1_conv1 = gen_int_bias(s1_dark4_1_conv1, s2_dark4_1_conv1, bias_f_dark4_1_conv1)
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
        self.bias_int_dark4_1_conv2 = gen_int_bias(s1_dark4_1_conv2, s2_dark4_1_conv2, bias_f_dark4_1_conv2)
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
        self.bias_int_dark4_1_m0_conv1 = gen_int_bias(s1_dark4_1_m0_conv1, s2_dark4_1_m0_conv1, bias_f_dark4_1_m0_conv1)
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
        self.bias_int_dark4_1_m0_conv2 = gen_int_bias(s1_dark4_1_m0_conv2, s2_dark4_1_m0_conv2, bias_f_dark4_1_m0_conv2)
        coe_name = '../data1_coe/out_hand_dark4_1_m0_conv2_leak.coe'
        self.dark4_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark4_1_m0_conv2, quant_zero_point1=z1_dark4_1_m0_conv2,
                                         quant_scale2=s2_dark4_1_m0_conv2,
                                         quant_zero_point2=z2_dark4_1_m0_conv2, quant_scale3=s3_dark4_1_m0_conv2,
                                         quant_zero_point3=z3_dark4_1_m0_conv2, coe_name=coe_name)

        s1_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.scale.npy")
        z1_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.0.csp.zero_point.npy")
        s2_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.weight.scale.npy")
        z2_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.weight.zero_point.npy")
        s3_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.scale.npy")
        z3_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.zero_point.npy")
        bias_f_dark4_1_m1_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.bias.npy")
        self.bias_int_dark4_1_m1_conv1 = gen_int_bias(s1_dark4_1_m1_conv1, s2_dark4_1_m1_conv1, bias_f_dark4_1_m1_conv1)
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
        self.bias_int_dark4_1_m1_conv2 = gen_int_bias(s1_dark4_1_m1_conv2, s2_dark4_1_m1_conv2, bias_f_dark4_1_m1_conv2)
        coe_name = '../data1_coe/out_hand_dark4_1_m1_conv2_leak.coe'
        self.dark4_1_m1_conv2 = Conv2d_Q(quant_scale1=s1_dark4_1_m1_conv2, quant_zero_point1=z1_dark4_1_m1_conv2,
                                         quant_scale2=s2_dark4_1_m1_conv2,
                                         quant_zero_point2=z2_dark4_1_m1_conv2, quant_scale3=s3_dark4_1_m1_conv2,
                                         quant_zero_point3=z3_dark4_1_m1_conv2, coe_name=coe_name)

        s1_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.scale.npy")
        z1_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.1.csp.zero_point.npy")
        s2_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.weight.scale.npy")
        z2_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.weight.zero_point.npy")
        s3_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.scale.npy")
        z3_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.zero_point.npy")
        bias_f_dark4_1_m2_conv1 = tensorr("../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.bias.npy")
        self.bias_int_dark4_1_m2_conv1 = gen_int_bias(s1_dark4_1_m2_conv1, s2_dark4_1_m2_conv1, bias_f_dark4_1_m2_conv1)
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
        self.bias_int_dark4_1_m2_conv2 = gen_int_bias(s1_dark4_1_m2_conv2, s2_dark4_1_m2_conv2, bias_f_dark4_1_m2_conv2)
        coe_name = '../data1_coe/out_hand_dark4_1_m2_conv2_leak.coe'
        self.dark4_1_m2_conv2 = Conv2d_Q(quant_scale1=s1_dark4_1_m2_conv2, quant_zero_point1=z1_dark4_1_m2_conv2,
                                         quant_scale2=s2_dark4_1_m2_conv2,
                                         quant_zero_point2=z2_dark4_1_m2_conv2, quant_scale3=s3_dark4_1_m2_conv2,
                                         quant_zero_point3=z3_dark4_1_m2_conv2, coe_name=coe_name)

        s1_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.csp1.scale.npy")
        z1_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.csp1.zero_point.npy")
        s2_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.weight.scale.npy")
        z2_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.weight.zero_point.npy")
        s3_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.scale.npy")
        z3_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.zero_point.npy")
        bias_f_dark4_1_conv3 = tensorr("../para_682/backbone.backbone.dark4.1.conv3.conv.bias.npy")
        self.bias_int_dark4_1_conv3 = gen_int_bias(s1_dark4_1_conv3, s2_dark4_1_conv3, bias_f_dark4_1_conv3)
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
        self.bias_int_dark5_0_conv = gen_int_bias(s1_dark5_0_conv, s2_dark5_0_conv, bias_f_dark5_0_conv)
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
        self.bias_int_dark5_1_conv1 = gen_int_bias(s1_dark5_1_conv1, s2_dark5_1_conv1, bias_f_dark5_1_conv1)
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
        self.bias_int_dark5_1_conv2 = gen_int_bias(s1_dark5_1_conv2, s2_dark5_1_conv2, bias_f_dark5_1_conv2)
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
        self.bias_int_dark5_1_m0_conv1 = gen_int_bias(s1_dark5_1_m0_conv1, s2_dark5_1_m0_conv1, bias_f_dark5_1_m0_conv1)
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
        self.bias_int_dark5_1_m0_conv2 = gen_int_bias(s1_dark5_1_m0_conv2, s2_dark5_1_m0_conv2, bias_f_dark5_1_m0_conv2)
        coe_name = '../data1_coe/out_hand_dark5_1_m0_conv2_leak.coe'
        self.dark5_1_m0_conv2 = Conv2d_Q(quant_scale1=s1_dark5_1_m0_conv2, quant_zero_point1=z1_dark5_1_m0_conv2,
                                         quant_scale2=s2_dark5_1_m0_conv2,
                                         quant_zero_point2=z2_dark5_1_m0_conv2, quant_scale3=s3_dark5_1_m0_conv2,
                                         quant_zero_point3=z3_dark5_1_m0_conv2, coe_name=coe_name)

        s1_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.csp1.scale.npy")
        z1_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.csp1.zero_point.npy")
        s2_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.weight.scale.npy")
        z2_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.weight.zero_point.npy")
        s3_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.scale.npy")
        z3_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.zero_point.npy")
        bias_f_dark5_1_conv3 = tensorr("../para_682/backbone.backbone.dark5.1.conv3.conv.bias.npy")
        self.bias_int_dark5_1_conv3 = gen_int_bias(s1_dark5_1_conv3, s2_dark5_1_conv3, bias_f_dark5_1_conv3)
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
        self.bias_int_lateral_conv0 = gen_int_bias(s1_lateral_conv0, s2_lateral_conv0, bias_f_lateral_conv0)
        coe_name = '../data1_coe/out_hand_lateral_conv0_leak.coe'
        self.lateral_conv0 = Conv2d_Q(quant_scale1=s1_lateral_conv0, quant_zero_point1=z1_lateral_conv0,
                                      quant_scale2=s2_lateral_conv0,
                                      quant_zero_point2=z2_lateral_conv0, quant_scale3=s3_lateral_conv0,
                                      quant_zero_point3=z3_lateral_conv0, coe_name=coe_name)

        # *==================C3_p4===========================
        s1_C3_p4_conv1 = tensorr("../para_682/backbone.csp2.scale.npy")
        z1_C3_p4_conv1 = tensorr("../para_682/backbone.csp2.zero_point.npy")
        s2_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.weight.scale.npy")
        z2_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.weight.zero_point.npy")
        s3_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.scale.npy")
        z3_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.zero_point.npy")
        bias_f_C3_p4_conv1 = tensorr("../para_682/backbone.C3_p4.conv1.conv.bias.npy")
        self.bias_int_C3_p4_conv1 = gen_int_bias(s1_C3_p4_conv1, s2_C3_p4_conv1, bias_f_C3_p4_conv1)
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
        self.bias_int_C3_p4_conv2 = gen_int_bias(s1_C3_p4_conv2, s2_C3_p4_conv2, bias_f_C3_p4_conv2)
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
        self.bias_int_C3_p4_m0_conv1 = gen_int_bias(s1_C3_p4_m0_conv1, s2_C3_p4_m0_conv1, bias_f_C3_p4_m0_conv1)
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
        self.bias_int_C3_p4_m0_conv2 = gen_int_bias(s1_C3_p4_m0_conv2, s2_C3_p4_m0_conv2, bias_f_C3_p4_m0_conv2)
        coe_name = '../data1_coe/out_hand_C3_p4_m0_conv2_leak.coe'
        self.C3_p4_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_p4_m0_conv2, quant_zero_point1=z1_C3_p4_m0_conv2,
                                       quant_scale2=s2_C3_p4_m0_conv2,
                                       quant_zero_point2=z2_C3_p4_m0_conv2, quant_scale3=s3_C3_p4_m0_conv2,
                                       quant_zero_point3=z3_C3_p4_m0_conv2, coe_name=coe_name)

        s1_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.csp1.scale.npy")
        z1_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.csp1.zero_point.npy")
        s2_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.weight.scale.npy")
        z2_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.weight.zero_point.npy")
        s3_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.scale.npy")
        z3_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.zero_point.npy")
        bias_f_C3_p4_conv3 = tensorr("../para_682/backbone.C3_p4.conv3.conv.bias.npy")
        self.bias_int_C3_p4_conv3 = gen_int_bias(s1_C3_p4_conv3, s2_C3_p4_conv3, bias_f_C3_p4_conv3)
        coe_name = '../data1_coe/out_hand_C3_p4_conv3.coe'
        self.C3_p4_conv3 = Conv2d_Q(quant_scale1=s1_C3_p4_conv3, quant_zero_point1=z1_C3_p4_conv3,
                                    quant_scale2=s2_C3_p4_conv3,
                                    quant_zero_point2=z2_C3_p4_conv3, quant_scale3=s3_C3_p4_conv3,
                                    quant_zero_point3=z3_C3_p4_conv3, coe_name=coe_name)
        # *======================reduce============================
        s1_reduce_conv1 = tensorr("../para_682/backbone.C3_p4.conv3.conv.scale.npy")
        z1_reduce_conv1 = tensorr("../para_682/backbone.C3_p4.conv3.conv.zero_point.npy")
        s2_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.weight.scale.npy")
        z2_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.weight.zero_point.npy")
        s3_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.scale.npy")
        z3_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.zero_point.npy")
        bias_reduce_conv1 = tensorr("../para_682/backbone.reduce_conv1.conv.bias.npy")
        self.bias_int_reduce_conv1 = gen_int_bias(s1_reduce_conv1, s2_reduce_conv1, bias_reduce_conv1)
        coe_name = '../data1_coe/out_hand_reduce_conv1.coe'
        self.reduce_conv1 = Conv2d_Q(quant_scale1=s1_reduce_conv1, quant_zero_point1=z1_reduce_conv1,
                                     quant_scale2=s2_reduce_conv1,
                                     quant_zero_point2=z2_reduce_conv1, quant_scale3=s3_reduce_conv1,
                                     quant_zero_point3=z3_reduce_conv1, coe_name=coe_name)

        # *==========================C3_p3=======================================
        s1_C3_p3_conv1 = tensorr("../para_682/backbone.csp3.scale.npy")
        z1_C3_p3_conv1 = tensorr("../para_682/backbone.csp3.zero_point.npy")
        s2_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.weight.scale.npy")
        z2_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.weight.zero_point.npy")
        s3_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.scale.npy")
        z3_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.zero_point.npy")
        bias_f_C3_p3_conv1 = tensorr("../para_682/backbone.C3_p3.conv1.conv.bias.npy")
        self.bias_int_C3_p3_conv1 = gen_int_bias(s1_C3_p3_conv1, s2_C3_p3_conv1, bias_f_C3_p3_conv1)
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
        self.bias_int_C3_p3_conv2 = gen_int_bias(s1_C3_p3_conv2, s2_C3_p3_conv2, bias_f_C3_p3_conv2)
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
        self.bias_int_C3_p3_m0_conv1 = gen_int_bias(s1_C3_p3_m0_conv1, s2_C3_p3_m0_conv1, bias_f_C3_p3_m0_conv1)
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
        self.bias_int_C3_p3_m0_conv2 = gen_int_bias(s1_C3_p3_m0_conv2, s2_C3_p3_m0_conv2, bias_f_C3_p3_m0_conv2)
        coe_name = '../data1_coe/out_hand_C3_p3_m0_conv2_leak.coe'
        self.C3_p3_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_p3_m0_conv2, quant_zero_point1=z1_C3_p3_m0_conv2,
                                       quant_scale2=s2_C3_p3_m0_conv2,
                                       quant_zero_point2=z2_C3_p3_m0_conv2, quant_scale3=s3_C3_p3_m0_conv2,
                                       quant_zero_point3=z3_C3_p3_m0_conv2, coe_name=coe_name)

        s1_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.csp1.scale.npy")
        z1_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.csp1.zero_point.npy")
        s2_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.weight.scale.npy")
        z2_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.weight.zero_point.npy")
        s3_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.scale.npy")
        z3_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.zero_point.npy")
        bias_f_C3_p3_conv3 = tensorr("../para_682/backbone.C3_p3.conv3.conv.bias.npy")
        self.bias_int_C3_p3_conv3 = gen_int_bias(s1_C3_p3_conv3, s2_C3_p3_conv3, bias_f_C3_p3_conv3)
        coe_name = '../data1_coe/out_hand_C3_p3_conv3.coe'
        self.C3_p3_conv3 = Conv2d_Q(quant_scale1=s1_C3_p3_conv3, quant_zero_point1=z1_C3_p3_conv3,
                                    quant_scale2=s2_C3_p3_conv3,
                                    quant_zero_point2=z2_C3_p3_conv3, quant_scale3=s3_C3_p3_conv3,
                                    quant_zero_point3=z3_C3_p3_conv3, coe_name=coe_name)
        # *=============================bu_conv2=============================================
        s1_bu_conv2 = tensorr("../para_682/backbone.C3_p3.conv3.conv.scale.npy")
        z1_bu_conv2 = tensorr("../para_682/backbone.C3_p3.conv3.conv.zero_point.npy")
        s2_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.weight.scale.npy")
        z2_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.weight.zero_point.npy")
        s3_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.scale.npy")
        z3_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.zero_point.npy")
        bias_bu_conv2 = tensorr("../para_682/backbone.bu_conv2.conv.bias.npy")
        self.bias_int_bu_conv2 = gen_int_bias(s1_bu_conv2, s2_bu_conv2, bias_bu_conv2)
        coe_name = '../data1_coe/out_hand_bu_conv2.coe'
        self.bu_conv2 = Conv2d_Q(quant_scale1=s1_bu_conv2, quant_zero_point1=z1_bu_conv2, quant_scale2=s2_bu_conv2,
                                 quant_zero_point2=z2_bu_conv2, quant_scale3=s3_bu_conv2,
                                 quant_zero_point3=z3_bu_conv2, coe_name=coe_name)

        # *====================C3_n3==========================
        s1_C3_n3_conv1 = tensorr("../para_682/backbone.csp4.scale.npy")
        z1_C3_n3_conv1 = tensorr("../para_682/backbone.csp4.zero_point.npy")
        s2_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.weight.scale.npy")
        z2_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.weight.zero_point.npy")
        s3_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.scale.npy")
        z3_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.zero_point.npy")
        bias_f_C3_n3_conv1 = tensorr("../para_682/backbone.C3_n3.conv1.conv.bias.npy")
        self.bias_int_C3_n3_conv1 = gen_int_bias(s1_C3_n3_conv1, s2_C3_n3_conv1, bias_f_C3_n3_conv1)
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
        self.bias_int_C3_n3_conv2 = gen_int_bias(s1_C3_n3_conv2, s2_C3_n3_conv2, bias_f_C3_n3_conv2)
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
        self.bias_C3_n3_m0_conv1 = gen_int_bias(s1_C3_n3_m0_conv1, s2_C3_n3_m0_conv1, bias_f_C3_n3_m0_conv1)
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
        self.bias_int_C3_n3_m0_conv2 = gen_int_bias(s1_C3_n3_m0_conv2, s2_C3_n3_m0_conv2, bias_f_C3_n3_m0_conv2)
        coe_name = '../data1_coe/out_hand_C3_n3_m0_conv2_leak.coe'
        self.C3_n3_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_n3_m0_conv2, quant_zero_point1=z1_C3_n3_m0_conv2,
                                       quant_scale2=s2_C3_n3_m0_conv2,
                                       quant_zero_point2=z2_C3_n3_m0_conv2, quant_scale3=s3_C3_n3_m0_conv2,
                                       quant_zero_point3=z3_C3_n3_m0_conv2, coe_name=coe_name)

        s1_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.csp1.scale.npy")
        z1_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.csp1.zero_point.npy")
        s2_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.weight.scale.npy")
        z2_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.weight.zero_point.npy")
        s3_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.scale.npy")
        z3_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.zero_point.npy")
        bias_f_C3_n3_conv3 = tensorr("../para_682/backbone.C3_n3.conv3.conv.bias.npy")
        self.bias_int_C3_n3_conv3 = gen_int_bias(s1_C3_n3_conv3, s2_C3_n3_conv3, bias_f_C3_n3_conv3)
        coe_name = '../data1_coe/out_hand_C3_n3_conv3.coe'
        self.C3_n3_conv3 = Conv2d_Q(quant_scale1=s1_C3_n3_conv3, quant_zero_point1=z1_C3_n3_conv3,
                                    quant_scale2=s2_C3_n3_conv3,
                                    quant_zero_point2=z2_C3_n3_conv3, quant_scale3=s3_C3_n3_conv3,
                                    quant_zero_point3=z3_C3_n3_conv3, coe_name=coe_name)
        # *=======================bu_conv1===========================================================
        s1_bu_conv1 = tensorr("../para_682/backbone.C3_n3.conv3.conv.scale.npy")
        z1_bu_conv1 = tensorr("../para_682/backbone.C3_n3.conv3.conv.zero_point.npy")
        s2_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.weight.scale.npy")
        z2_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.weight.zero_point.npy")
        s3_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.scale.npy")
        z3_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.zero_point.npy")
        bias_bu_conv1 = tensorr("../para_682/backbone.bu_conv1.conv.bias.npy")
        self.bias_int_bu_conv1 = gen_int_bias(s1_bu_conv1, s2_bu_conv1, bias_bu_conv1)
        coe_name = '../data1_coe/out_hand_bu_conv1.coe'
        self.bu_conv1 = Conv2d_Q(quant_scale1=s1_bu_conv1, quant_zero_point1=z1_bu_conv1, quant_scale2=s2_bu_conv1,
                                 quant_zero_point2=z2_bu_conv1, quant_scale3=s3_bu_conv1,
                                 quant_zero_point3=z3_bu_conv1, coe_name=coe_name)

        # *=======================C3_n4============================
        s1_C3_n4_conv1 = tensorr("../para_682/backbone.csp5.scale.npy")
        z1_C3_n4_conv1 = tensorr("../para_682/backbone.csp5.zero_point.npy")
        s2_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.weight.scale.npy")
        z2_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.weight.zero_point.npy")
        s3_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.scale.npy")
        z3_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.zero_point.npy")
        bias_f_C3_n4_conv1 = tensorr("../para_682/backbone.C3_n4.conv1.conv.bias.npy")
        self.bias_int_C3_n4_conv1 = gen_int_bias(s1_C3_n4_conv1, s2_C3_n4_conv1, bias_f_C3_n4_conv1)
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
        self.bias_int_C3_n4_conv2 = gen_int_bias(s1_C3_n4_conv2, s2_C3_n4_conv2, bias_f_C3_n4_conv2)
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
        self.bias_C3_n4_m0_conv1 = gen_int_bias(s1_C3_n4_m0_conv1, s2_C3_n4_m0_conv1, bias_f_C3_n4_m0_conv1)
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
        self.bias_int_C3_n4_m0_conv2 = gen_int_bias(s1_C3_n4_m0_conv2, s2_C3_n4_m0_conv2, bias_f_C3_n4_m0_conv2)
        coe_name = '../data1_coe/out_hand_C3_n4_m0_conv2_leak.coe'
        self.C3_n4_m0_conv2 = Conv2d_Q(quant_scale1=s1_C3_n4_m0_conv2, quant_zero_point1=z1_C3_n4_m0_conv2,
                                       quant_scale2=s2_C3_n4_m0_conv2,
                                       quant_zero_point2=z2_C3_n4_m0_conv2, quant_scale3=s3_C3_n4_m0_conv2,
                                       quant_zero_point3=z3_C3_n4_m0_conv2, coe_name=coe_name)

        s1_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.csp1.scale.npy")
        z1_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.csp1.zero_point.npy")
        s2_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.weight.scale.npy")
        z2_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.weight.zero_point.npy")
        s3_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.scale.npy")
        z3_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.zero_point.npy")
        bias_f_C3_n4_conv3 = tensorr("../para_682/backbone.C3_n4.conv3.conv.bias.npy")
        self.bias_int_C3_n4_conv3 = gen_int_bias(s1_C3_n4_conv3, s2_C3_n4_conv3, bias_f_C3_n4_conv3)
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
        self.bias_int_stems0_conv = gen_int_bias(s1_stems0_conv, s2_stems0_conv, bias_f_stems0_conv)
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
        self.bias_int_cls_convs0_0 = gen_int_bias(s1_cls_convs0_0, s2_cls_convs0_0, bias_f_cls_convs0_0)
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
        self.bias_int_cls_convs0_1 = gen_int_bias(s1_cls_convs0_1, s2_cls_convs0_1, bias_f_cls_convs0_1)
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
        self.bias_int_cls_preds0 = gen_int_bias(s1_cls_preds0, s2_cls_preds0, bias_f_cls_preds0)
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
        self.bias_int_reg_convs0_0 = gen_int_bias(s1_reg_convs0_0, s2_reg_convs0_0, bias_f_reg_convs0_0)
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
        self.bias_int_reg_convs0_1 = gen_int_bias(s1_reg_convs0_1, s2_reg_convs0_1, bias_f_reg_convs0_1)
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
        self.bias_int_reg_preds0 = gen_int_bias(s1_reg_preds0, s2_reg_preds0, bias_f_reg_preds0)
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
        self.bias_int_obj_preds0 = gen_int_bias(s1_obj_preds0, s2_obj_preds0, bias_f_obj_preds0)
        coe_name = '../data1_coe/out_hand_obj_preds0.coe'
        self.obj_preds0 = Conv2d_Q(quant_scale1=s1_obj_preds0, quant_zero_point1=z1_obj_preds0,
                                   quant_scale2=s2_obj_preds0,
                                   quant_zero_point2=z2_obj_preds0, quant_scale3=s3_obj_preds0,
                                   quant_zero_point3=z3_obj_preds0, coe_name=coe_name)

        # *================================output_p4===================================================
        s1_stems1_conv = tensorr("../para_682/backbone.C3_n3.conv3.conv.scale.npy")
        z1_stems1_conv = tensorr("../para_682/backbone.C3_n3.conv3.conv.zero_point.npy")
        s2_stems1_conv = tensorr("../para_682/head.stems.1.conv.weight.scale.npy")
        z2_stems1_conv = tensorr("../para_682/head.stems.1.conv.weight.zero_point.npy")
        s3_stems1_conv = tensorr("../para_682/head.stems.1.conv.scale.npy")
        z3_stems1_conv = tensorr("../para_682/head.stems.1.conv.zero_point.npy")
        bias_f_stems1_conv = tensorr("../para_682/head.stems.1.conv.bias.npy")
        self.bias_int_stems1_conv = gen_int_bias(s1_stems1_conv, s2_stems1_conv, bias_f_stems1_conv)
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
        self.bias_int_cls_convs1_0 = gen_int_bias(s1_cls_convs1_0, s2_cls_convs1_0, bias_f_cls_convs1_0)
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
        self.bias_int_cls_convs1_1 = gen_int_bias(s1_cls_convs1_1, s2_cls_convs1_1, bias_f_cls_convs1_1)
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
        self.bias_int_cls_preds1 = gen_int_bias(s1_cls_preds1, s2_cls_preds1, bias_f_cls_preds1)
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
        self.bias_int_reg_convs1_0 = gen_int_bias(s1_reg_convs1_0, s2_reg_convs1_0, bias_f_reg_convs1_0)
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
        self.bias_int_reg_convs1_1 = gen_int_bias(s1_reg_convs1_1, s2_reg_convs1_1, bias_f_reg_convs1_1)
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
        self.bias_int_reg_preds1 = gen_int_bias(s1_reg_preds1, s2_reg_preds1, bias_f_reg_preds1)
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
        self.bias_int_obj_preds1 = gen_int_bias(s1_obj_preds1, s2_obj_preds1, bias_f_obj_preds1)
        coe_name = '../data1_coe/out_hand_obj_preds1.coe'
        self.obj_preds1 = Conv2d_Q(quant_scale1=s1_obj_preds1, quant_zero_point1=z1_obj_preds1,
                                   quant_scale2=s2_obj_preds1,
                                   quant_zero_point2=z2_obj_preds1, quant_scale3=s3_obj_preds1,
                                   quant_zero_point3=z3_obj_preds1, coe_name=coe_name)

        # *================================output_p5===================================================
        s1_stems2_conv = tensorr("../para_682/backbone.C3_n4.conv3.conv.scale.npy")
        z1_stems2_conv = tensorr("../para_682/backbone.C3_n4.conv3.conv.zero_point.npy")
        s2_stems2_conv = tensorr("../para_682/head.stems.2.conv.weight.scale.npy")
        z2_stems2_conv = tensorr("../para_682/head.stems.2.conv.weight.zero_point.npy")
        s3_stems2_conv = tensorr("../para_682/head.stems.2.conv.scale.npy")
        z3_stems2_conv = tensorr("../para_682/head.stems.2.conv.zero_point.npy")
        bias_f_stems2_conv = tensorr("../para_682/head.stems.2.conv.bias.npy")
        self.bias_int_stems2_conv = gen_int_bias(s1_stems2_conv, s2_stems2_conv, bias_f_stems2_conv)
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
        self.bias_int_cls_convs2_0 = gen_int_bias(s1_cls_convs2_0, s2_cls_convs2_0, bias_f_cls_convs2_0)
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
        self.bias_int_cls_convs2_1 = gen_int_bias(s1_cls_convs2_1, s2_cls_convs2_1, bias_f_cls_convs2_1)
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
        self.bias_int_cls_preds2 = gen_int_bias(s1_cls_preds2, s2_cls_preds2, bias_f_cls_preds2)
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
        self.bias_int_reg_convs2_0 = gen_int_bias(s1_reg_convs2_0, s2_reg_convs2_0, bias_f_reg_convs2_0)
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
        self.bias_int_reg_convs2_1 = gen_int_bias(s1_reg_convs2_1, s2_reg_convs2_1, bias_f_reg_convs2_1)
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
        self.bias_int_reg_preds2 = gen_int_bias(s1_reg_preds2, s2_reg_preds2, bias_f_reg_preds2)
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
        self.bias_int_obj_preds2 = gen_int_bias(s1_obj_preds2, s2_obj_preds2, bias_f_obj_preds2)
        coe_name = '../data1_coe/out_hand_obj_preds2.coe'
        self.obj_preds2 = Conv2d_Q(quant_scale1=s1_obj_preds2, quant_zero_point1=z1_obj_preds2,
                                   quant_scale2=s2_obj_preds2,
                                   quant_zero_point2=z2_obj_preds2, quant_scale3=s3_obj_preds2,
                                   quant_zero_point3=z3_obj_preds2, coe_name=coe_name)

    def forward(self, x):
        path1 = 'biasscaleshift1114.bin'

        weight_stem_conv = tensorr('../para_682/backbone.backbone.stem.conv.conv.weight.int.npy')

        weight_numpy = weight_stem_conv.numpy()
        weight_numpy[:, [2, 1], :, :] = weight_numpy[:, [1, 2], :, :]
        weight = torch.from_numpy(weight_numpy)
        x = self.stem_conv(1, weight, self.bias_int_stem_conv, path1, coee=1, block=0, inchannel=8)
        # exit()
        weight_dark2_0_conv = tensorr('../para_682/backbone.backbone.dark2.0.conv.weight.int.npy')
        x = self.dark2_0_conv(1, weight_dark2_0_conv, self.bias_int_dark2_0_conv, path1, coee=1, block=0, inchannel=8)

        weight_dark2_1_conv1 = tensorr('../para_682/backbone.backbone.dark2.1.conv1.conv.weight.int.npy')
        x = self.dark2_1_conv1(1, weight_dark2_1_conv1, self.bias_int_dark2_1_conv1, path1, coee=1, block=0,
                               inchannel=32)

        weight_dark2_1_conv2 = tensorr('../para_682/backbone.backbone.dark2.1.conv2.conv.weight.int.npy')
        x = self.dark2_1_conv2(1, weight_dark2_1_conv2, self.bias_int_dark2_1_conv2, path1, coee=1, block=0,
                               inchannel=32)

        weight_dark2_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark2.1.m.0.conv1.conv.weight.int.npy')
        x = self.dark2_1_m0_conv1(1, weight_dark2_1_m0_conv1, self.bias_int_dark2_1_m0_conv1, path1, coee=1, block=0,
                                  inchannel=32)

        weight_dark2_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark2.1.m.0.conv2.conv.weight.int.npy')
        x = self.dark2_1_m0_conv2(1, weight_dark2_1_m0_conv2, self.bias_int_dark2_1_m0_conv2, path1, coee=1, block=0,
                                  inchannel=8)
        # exit()
        weight_dark2_1_conv3 = tensorr('../para_682/backbone.backbone.dark2.1.conv3.conv.weight.int.npy')
        x = self.dark2_1_conv3(1, weight_dark2_1_conv3, self.bias_int_dark2_1_conv3, path1, coee=1, block=0,
                               inchannel=32)

        weight_dark3_0_conv = tensorr('../para_682/backbone.backbone.dark3.0.conv.weight.int.npy')
        x = self.dark3_0_conv(1, weight_dark3_0_conv, self.bias_int_dark3_0_conv, path1, coee=1, block=0, inchannel=8)

        weight_dark3_1_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.conv1.conv.weight.int.npy')
        x = self.dark3_1_conv1(1, weight_dark3_1_conv1, self.bias_int_dark3_1_conv1, path1, coee=1, block=0,
                               inchannel=32)

        weight_dark3_1_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.conv2.conv.weight.int.npy')
        x = self.dark3_1_conv2(1, weight_dark3_1_conv2, self.bias_int_dark3_1_conv2, path1, coee=1, block=0,
                               inchannel=32)

        weight_dark3_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.m.0.conv1.conv.weight.int.npy')
        x = self.dark3_1_m0_conv1(1, weight_dark3_1_m0_conv1, self.bias_int_dark3_1_m0_conv1, path1, coee=1, block=0,
                                  inchannel=32)

        weight_dark3_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.m.0.conv2.conv.weight.int.npy')
        x = self.dark3_1_m0_conv2(1, weight_dark3_1_m0_conv2, self.bias_int_dark3_1_m0_conv2, path1, coee=1, block=0,
                                  inchannel=8)

        weight_dark3_1_m1_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.m.1.conv1.conv.weight.int.npy')
        x = self.dark3_1_m1_conv1(1, weight_dark3_1_m1_conv1, self.bias_int_dark3_1_m1_conv1, path1, coee=1, block=0,
                                  inchannel=32)

        weight_dark3_1_m1_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.m.1.conv2.conv.weight.int.npy')
        x = self.dark3_1_m1_conv2(1, weight_dark3_1_m1_conv2, self.bias_int_dark3_1_m1_conv2, path1, coee=1, block=0,
                                  inchannel=8)

        weight_dark3_1_m2_conv1 = tensorr('../para_682/backbone.backbone.dark3.1.m.2.conv1.conv.weight.int.npy')
        x = self.dark3_1_m2_conv1(1, weight_dark3_1_m2_conv1, self.bias_int_dark3_1_m2_conv1, path1, coee=1, block=0,
                                  inchannel=32)

        weight_dark3_1_m2_conv2 = tensorr('../para_682/backbone.backbone.dark3.1.m.2.conv2.conv.weight.int.npy')
        x = self.dark3_1_m2_conv2(1, weight_dark3_1_m2_conv2, self.bias_int_dark3_1_m2_conv2, path1, coee=1, block=0,
                                  inchannel=8)

        weight_dark3_1_conv3 = tensorr('../para_682/backbone.backbone.dark3.1.conv3.conv.weight.int.npy')
        x = self.dark3_1_conv3(1, weight_dark3_1_conv3, self.bias_int_dark3_1_conv3, path1, coee=1, block=0,
                               inchannel=32)

        weight_dark4_0_conv = tensorr('../para_682/backbone.backbone.dark4.0.conv.weight.int.npy')
        x = self.dark4_0_conv(1, weight_dark4_0_conv, self.bias_int_dark4_0_conv, path1, block=0, inchannel=8)

        weight_dark4_1_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.conv1.conv.weight.int.npy')
        x = self.dark4_1_conv1(1, weight_dark4_1_conv1, self.bias_int_dark4_1_conv1, path1, block=0, inchannel=32)

        weight_dark4_1_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.conv2.conv.weight.int.npy')
        x = self.dark4_1_conv2(1, weight_dark4_1_conv2, self.bias_int_dark4_1_conv2, path1, block=0, inchannel=32)

        weight_dark4_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.m.0.conv1.conv.weight.int.npy')
        x = self.dark4_1_m0_conv1(1, weight_dark4_1_m0_conv1, self.bias_int_dark4_1_m0_conv1, path1, block=0,
                                  inchannel=32)

        weight_dark4_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.m.0.conv2.conv.weight.int.npy')
        x = self.dark4_1_m0_conv2(1, weight_dark4_1_m0_conv2, self.bias_int_dark4_1_m0_conv2, path1, block=0,
                                  inchannel=8)

        weight_dark4_1_m1_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.m.1.conv1.conv.weight.int.npy')
        x = self.dark4_1_m1_conv1(1, weight_dark4_1_m1_conv1, self.bias_int_dark4_1_m1_conv1, path1, block=0,
                                  inchannel=32)

        weight_dark4_1_m1_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.m.1.conv2.conv.weight.int.npy')
        x = self.dark4_1_m1_conv2(1, weight_dark4_1_m1_conv2, self.bias_int_dark4_1_m1_conv2, path1, block=0,
                                  inchannel=8)

        weight_dark4_1_m2_conv1 = tensorr('../para_682/backbone.backbone.dark4.1.m.2.conv1.conv.weight.int.npy')

        x = self.dark4_1_m2_conv1(1, weight_dark4_1_m2_conv1, self.bias_int_dark4_1_m2_conv1, path1, block=0,
                                  inchannel=32)

        weight_dark4_1_m2_conv2 = tensorr('../para_682/backbone.backbone.dark4.1.m.2.conv2.conv.weight.int.npy')
        x = self.dark4_1_m2_conv2(1, weight_dark4_1_m2_conv2, self.bias_int_dark4_1_m2_conv2, path1, block=0,
                                  inchannel=8)

        weight_dark4_1_conv3 = tensorr('../para_682/backbone.backbone.dark4.1.conv3.conv.weight.int.npy')
        x = self.dark4_1_conv3(1, weight_dark4_1_conv3, self.bias_int_dark4_1_conv3, path1, block=0, inchannel=32)

        # ?=============================dark5==================================
        weight_dark5_0_conv = tensorr('../para_682/backbone.backbone.dark5.0.conv.weight.int.npy')
        x = self.dark5_0_conv(1, weight_dark5_0_conv, self.bias_int_dark5_0_conv, path1, block=2, inchannel=8)

        weight_dark5_1_conv1 = tensorr('../para_682/backbone.backbone.dark5.1.conv1.conv.weight.int.npy')
        x = self.dark5_1_conv1(1, weight_dark5_1_conv1, self.bias_int_dark5_1_conv1, path1, block=0, inchannel=32)

        weight_dark5_1_conv2 = tensorr('../para_682/backbone.backbone.dark5.1.conv2.conv.weight.int.npy')
        x = self.dark5_1_conv2(1, weight_dark5_1_conv2, self.bias_int_dark5_1_conv2, path1, block=0, inchannel=32)

        weight_dark5_1_m0_conv1 = tensorr('../para_682/backbone.backbone.dark5.1.m.0.conv1.conv.weight.int.npy')
        x = self.dark5_1_m0_conv1(1, weight_dark5_1_m0_conv1, self.bias_int_dark5_1_m0_conv1, path1, block=0,
                                  inchannel=32)

        weight_dark5_1_m0_conv2 = tensorr('../para_682/backbone.backbone.dark5.1.m.0.conv2.conv.weight.int.npy')
        x = self.dark5_1_m0_conv2(1, weight_dark5_1_m0_conv2, self.bias_int_dark5_1_m0_conv2, path1, block=0,
                                  inchannel=8)

        weight_dark5_1_conv3 = tensorr('../para_682/backbone.backbone.dark5.1.conv3.conv.weight.int.npy')
        x = self.dark5_1_conv3(1, weight_dark5_1_conv3, self.bias_int_dark5_1_conv3, path1, block=0, inchannel=32)

        weight_lateral_conv0 = tensorr('../para_682/backbone.lateral_conv0.conv.weight.int.npy')
        x = self.lateral_conv0(1, weight_lateral_conv0, self.bias_int_lateral_conv0, path1, block=0, inchannel=32)

        # ?================================c3p4========================================================
        weight_C3_p4_conv1 = tensorr('../para_682/backbone.C3_p4.conv1.conv.weight.int.npy')
        x = self.C3_p4_conv1(1, weight_C3_p4_conv1, self.bias_int_C3_p4_conv1, path1, block=0, inchannel=32)

        weight_C3_p4_conv2 = tensorr('../para_682/backbone.C3_p4.conv2.conv.weight.int.npy')
        x = self.C3_p4_conv2(1, weight_C3_p4_conv2, self.bias_int_C3_p4_conv2, path1, block=0, inchannel=32)

        weight_C3_p4_m0_conv1 = tensorr('../para_682/backbone.C3_p4.m.0.conv1.conv.weight.int.npy')
        x = self.C3_p4_m0_conv1(1, weight_C3_p4_m0_conv1, self.bias_int_C3_p4_m0_conv1, path1, block=0, inchannel=32)

        weight_C3_p4_m0_conv2 = tensorr('../para_682/backbone.C3_p4.m.0.conv2.conv.weight.int.npy')
        x = self.C3_p4_m0_conv2(1, weight_C3_p4_m0_conv2, self.bias_int_C3_p4_m0_conv2, path1, block=0, inchannel=8)

        weight_C3_p4_conv3 = tensorr('../para_682/backbone.C3_p4.conv3.conv.weight.int.npy')
        x = self.C3_p4_conv3(1, weight_C3_p4_conv3, self.bias_int_C3_p4_conv3, path1, block=0, inchannel=32)

        weight_reduce_conv1 = tensorr('../para_682/backbone.reduce_conv1.conv.weight.int.npy')
        x = self.reduce_conv1(1, weight_reduce_conv1, self.bias_int_reduce_conv1, path1, block=0, inchannel=32)

        # ?================================c3p3========================================================
        weight_C3_p3_conv1 = tensorr('../para_682/backbone.C3_p3.conv1.conv.weight.int.npy')
        x = self.C3_p3_conv1(1, weight_C3_p3_conv1, self.bias_int_C3_p3_conv1, path1, block=0, inchannel=32)

        weight_C3_p3_conv2 = tensorr('../para_682/backbone.C3_p3.conv2.conv.weight.int.npy')
        x = self.C3_p3_conv2(1, weight_C3_p3_conv2, self.bias_int_C3_p3_conv2, path1, block=0, inchannel=32)

        weight_C3_p3_m0_conv1 = tensorr('../para_682/backbone.C3_p3.m.0.conv1.conv.weight.int.npy')
        x = self.C3_p3_m0_conv1(1, weight_C3_p3_m0_conv1, self.bias_int_C3_p3_m0_conv1, path1, block=0, inchannel=32)

        weight_C3_p3_m0_conv2 = tensorr('../para_682/backbone.C3_p3.m.0.conv2.conv.weight.int.npy')
        x = self.C3_p3_m0_conv2(1, weight_C3_p3_m0_conv2, self.bias_int_C3_p3_m0_conv2, path1, block=0, inchannel=8)

        weight_C3_p3_conv3 = tensorr('../para_682/backbone.C3_p3.conv3.conv.weight.int.npy')
        x = self.C3_p3_conv3(1, weight_C3_p3_conv3, self.bias_int_C3_p3_conv3, path1, block=0, inchannel=32)

        # ?============bu_conv2=============================================================================
        weight_bu_conv2 = tensorr('../para_682/backbone.bu_conv2.conv.weight.int.npy')
        x = self.bu_conv2(1, weight_bu_conv2, self.bias_int_bu_conv2, path1, block=0, inchannel=8)

        # ?================================C3_n3========================================================
        weight_C3_n3_conv1 = tensorr('../para_682/backbone.C3_n3.conv1.conv.weight.int.npy')
        x = self.C3_n3_conv1(1, weight_C3_n3_conv1, self.bias_int_C3_n3_conv1, path1, block=0, inchannel=32)

        weight_C3_n3_conv2 = tensorr('../para_682/backbone.C3_n3.conv2.conv.weight.int.npy')
        x = self.C3_n3_conv2(1, weight_C3_n3_conv2, self.bias_int_C3_n3_conv2, path1, block=0, inchannel=32)

        weight_C3_n3_m0_conv1 = tensorr('../para_682/backbone.C3_n3.m.0.conv1.conv.weight.int.npy')
        x = self.C3_n3_m0_conv1(1, weight_C3_n3_m0_conv1,
                                self.bias_C3_n3_m0_conv1, path1, block=0, inchannel=32)

        weight_C3_n3_m0_conv2 = tensorr('../para_682/backbone.C3_n3.m.0.conv2.conv.weight.int.npy')
        x = self.C3_n3_m0_conv2(1, weight_C3_n3_m0_conv2, self.bias_int_C3_n3_m0_conv2, path1, block=0, inchannel=8)

        weight_C3_n3_conv3 = tensorr('../para_682/backbone.C3_n3.conv3.conv.weight.int.npy')
        x = self.C3_n3_conv3(1, weight_C3_n3_conv3, self.bias_int_C3_n3_conv3, path1, block=0, inchannel=32)

        # ?=======================================bu_conv1==================================================
        weight_bu_conv1 = tensorr('../para_682/backbone.bu_conv1.conv.weight.int.npy')
        x = self.bu_conv1(1, weight_bu_conv1, self.bias_int_bu_conv1, path1, block=0, inchannel=8)

        # ?=======================================c3_n4==================================================
        weight_C3_n4_conv1 = tensorr('../para_682/backbone.C3_n4.conv1.conv.weight.int.npy')
        x = self.C3_n4_conv1(1, weight_C3_n4_conv1, self.bias_int_C3_n4_conv1, path1, block=0, inchannel=32)

        weight_C3_n4_conv2 = tensorr('../para_682/backbone.C3_n4.conv2.conv.weight.int.npy')
        x = self.C3_n4_conv2(1, weight_C3_n4_conv2, self.bias_int_C3_n4_conv2, path1, block=0, inchannel=32)

        weight_C3_n4_m0_conv1 = tensorr('../para_682/backbone.C3_n4.m.0.conv1.conv.weight.int.npy')
        x = self.C3_n4_m0_conv1(1, weight_C3_n4_m0_conv1, self.bias_C3_n4_m0_conv1, path1, block=0, inchannel=32)

        weight_C3_n4_m0_conv2 = tensorr('../para_682/backbone.C3_n4.m.0.conv2.conv.weight.int.npy')
        x = self.C3_n4_m0_conv2(1, weight_C3_n4_m0_conv2, self.bias_int_C3_n4_m0_conv2, path1, block=0, inchannel=8)

        weight_C3_n4_conv3 = tensorr('../para_682/backbone.C3_n4.conv3.conv.weight.int.npy')
        x = self.C3_n4_conv3(1, weight_C3_n4_conv3, self.bias_int_C3_n4_conv3, path1, block=0, inchannel=32)

        # ?====================head=======================================
        weight_stems0_conv = tensorr('../para_682/head.stems.0.conv.weight.int.npy')
        x_1 = self.stems0_conv(1, weight_stems0_conv, self.bias_int_stems0_conv, path1, block=0, inchannel=32)

        weight_cls_convs0_0 = tensorr('../para_682/head.cls_convs.0.0.conv.weight.int.npy')
        x = self.cls_convs0_0(1, weight_cls_convs0_0, self.bias_int_cls_convs0_0, path1, block=0, inchannel=8)

        weight_cls_convs0_1 = tensorr('../para_682/head.cls_convs.0.1.conv.weight.int.npy')
        x = self.cls_convs0_1(1, weight_cls_convs0_1, self.bias_int_cls_convs0_1, path1, block=0, inchannel=8)

        weight_cls_preds0 = tensorr('../para_682/head.cls_preds.0.weight.int.npy')
        x = self.cls_preds0(1, weight_cls_preds0, self.bias_int_cls_preds0, path1, block=0, inchannel=32)

        weight_reg_convs0_0 = tensorr('../para_682/head.reg_convs.0.0.conv.weight.int.npy')
        x = self.reg_convs0_0(1, weight_reg_convs0_0, self.bias_int_reg_convs0_0, path1, block=0, inchannel=8)

        weight_reg_convs0_1 = tensorr('../para_682/head.reg_convs.0.1.conv.weight.int.npy')
        x = self.reg_convs0_1(1, weight_reg_convs0_1, self.bias_int_reg_convs0_1, path1, block=0, inchannel=8)

        weight_reg_preds0 = tensorr('../para_682/head.reg_preds.0.weight.int.npy')
        x = self.reg_preds0(1, weight_reg_preds0, self.bias_int_reg_preds0, path1, block=0, inchannel=32)

        weight_obj_preds0 = tensorr('../para_682/head.obj_preds.0.weight.int.npy')
        x = self.obj_preds0(1, weight_obj_preds0, self.bias_int_obj_preds0, path1, block=0, inchannel=32)

        # ===========================stem_conv============================
        weight_stems1_conv = tensorr('../para_682/head.stems.1.conv.weight.int.npy')
        x = self.stems1_conv(1, weight_stems1_conv, self.bias_int_stems1_conv, path1, block=0, inchannel=32)

        weight_cls_convs1_0 = tensorr('../para_682/head.cls_convs.1.0.conv.weight.int.npy')
        x = self.cls_convs1_0(1, weight_cls_convs1_0, self.bias_int_cls_convs1_0, path1, block=0, inchannel=8)

        weight_cls_convs1_1 = tensorr('../para_682/head.cls_convs.1.1.conv.weight.int.npy')
        x = self.cls_convs1_1(1, weight_cls_convs1_1, self.bias_int_cls_convs1_1, path1, block=0, inchannel=8)

        weight_cls_preds1 = tensorr('../para_682/head.cls_preds.1.weight.int.npy')
        x = self.cls_preds1(1, weight_cls_preds1, self.bias_int_cls_preds1, path1, block=0, inchannel=32)

        weight_reg_convs1_0 = tensorr('../para_682/head.reg_convs.1.0.conv.weight.int.npy')
        x = self.reg_convs1_0(1, weight_reg_convs1_0, self.bias_int_reg_convs1_0, path1, block=0, inchannel=8)

        weight_reg_convs1_1 = tensorr('../para_682/head.reg_convs.1.1.conv.weight.int.npy')
        x = self.reg_convs1_1(1, weight_reg_convs1_1, self.bias_int_reg_convs1_1, path1, block=0, inchannel=8)

        weight_reg_preds1 = tensorr('../para_682/head.reg_preds.1.weight.int.npy')
        x = self.reg_preds1(1, weight_reg_preds1, self.bias_int_reg_preds1, path1, block=0, inchannel=32)

        weight_obj_preds1 = tensorr('../para_682/head.obj_preds.1.weight.int.npy')
        x = self.obj_preds1(1, weight_obj_preds1, self.bias_int_cls_preds1, path1, block=0, inchannel=32)

        # -------------------------------------------------------------
        weight_stems2_conv = tensorr('../para_682/head.stems.2.conv.weight.int.npy')
        x = self.stems2_conv(1, weight_stems2_conv, self.bias_int_stems2_conv, path1, block=0, inchannel=32)

        weight_cls_convs2_0 = tensorr('../para_682/head.cls_convs.2.0.conv.weight.int.npy')
        x = self.cls_convs2_0(1, weight_cls_convs2_0, self.bias_int_cls_convs2_0, path1, block=0, inchannel=8)

        weight_cls_convs2_1 = tensorr('../para_682/head.cls_convs.2.1.conv.weight.int.npy')
        x = self.cls_convs2_1(1, weight_cls_convs2_1, self.bias_int_cls_convs2_1, path1, block=0, inchannel=8)

        weight_cls_preds2 = tensorr('../para_682/head.cls_preds.2.weight.int.npy')
        x = self.cls_preds2(1, weight_cls_preds2, self.bias_int_cls_preds2, path1, block=0, inchannel=32)

        weight_reg_convs2_0 = tensorr('../para_682/head.reg_convs.2.0.conv.weight.int.npy')
        x = self.reg_convs2_0(1, weight_reg_convs2_0, self.bias_int_reg_convs2_0, path1, block=0, inchannel=8)

        weight_reg_convs2_1 = tensorr('../para_682/head.reg_convs.2.1.conv.weight.int.npy')
        x = self.reg_convs2_1(1, weight_reg_convs2_1, self.bias_int_reg_convs2_1, path1, block=0, inchannel=8)

        weight_reg_preds2 = tensorr('../para_682/head.reg_preds.2.weight.int.npy')
        x = self.reg_preds2(1, weight_reg_preds2, self.bias_int_reg_preds2, path1, block=0, inchannel=32)

        weight_obj_preds2 = tensorr('../para_682/head.obj_preds.2.weight.int.npy')
        x = self.obj_preds2(1, weight_obj_preds2, self.bias_int_obj_preds2, path1, block=0, inchannel=32)
        obj_output_P5_address = x


if __name__ == "__main__":
    model = QuantizableYolo_tiny()(1)
