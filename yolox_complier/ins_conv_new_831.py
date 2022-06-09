import torch
from torch.nn.quantized import functional as qF
import numpy as np

'''

leaky_relu y = max(0,x)+leak*min(0,x)
输入激活函数之前的激活值范围 0-255 那么上面的公式就可以简化为
y = x

s3 激活值对应的scale
'''


def reg6_leaky(s3):
    add_data = []
    data1 = torch.ones(16)
    # 将data1中的数据更新为 -index * 10 - 5 范围(-5,-155,stride=-10)
    for index in range(data1.shape[0]):
        data1[index] = -index * 10 - 5

    for index in range(data1.shape[0]):
        # q_feature = round((data1[index]*s3-0)/s3)  q_feature中所有小于-128的值限制为-128，data1中的其他值保持原样
        # r =s(q-z) q = r/s+z
        q_feature = torch.quantize_per_tensor(data1[index] * s3, scale=float(s3),
                                              zero_point=int(0), dtype=torch.qint8)

        a = q_feature.int_repr()
        out_leak = qF.leaky_relu(input=q_feature, negative_slope=0.1, inplace=True)
        # out为量化之后的值进行leaky relu 运算的结果
        out = out_leak.int_repr()

        out2 = data1[index] * 0.1
        # 将量化值*0.1 四舍五入后的值与 原本值*0.1 四舍五入后的值进行对比
        # 对此方法来说超过 int8 负数范围的值 add_data 值为1，超过正数范围的值 add_data 为-1 范围内对应的add_data为0
        if np.round(out) > np.round(out2):
            add_data.append(1)
        elif np.round(out) < np.round(out2):
            add_data.append(-1)
        else:
            add_data.append(0)

    print(add_data)
    out_str = ""
    for index in range(data1.shape[0]):
        if add_data[index] == 0:
            out_str += '00'
        elif add_data[index] == 1:
            out_str += '01'
        elif add_data[index] == -1:
            out_str += '10'

    return out_str


'''
conv33para: 主要对reg4进行操作
在coe文件中存储权重和bias时，一行可写入64bit的值，权重是int8类型的一行可写入8个权重值，bias是 32 一行可写入2个bias值
    --m:输出通道数
    --c:输入通道数
    --dataSizeW:一行想要写入的权重的bit数，本项目应该是固定的64bit
    --dataSizeB:一行想要写入的bias的bit数，应该也是固定的64bit
'''


def conv33para(m, c, dataSizeW, dataSizeB):
    # 64*32*3*3   / 8*9
    # 一行存储8个值，one_conv_line 代表m个卷积核，每个卷积核所有通道上的左上角第一个值会占几行
    one_conv_line = int((m * c * 9) / ((dataSizeW / 8) * 9))
    one_conv_line = '{:04x}'.format(one_conv_line)

    # bias_line是bias需要的行数；64bit一行可以放2个bias，一个bias是32bit；bias的数量是m
    bias_line = int(m / (dataSizeB / 32))
    bias_line = '{:02x}'.format(bias_line)
    all_zero = '00'
    # reg4_para 组成了8位16进制 同样也是32bit
    reg4_para = str(one_conv_line) + str(bias_line) + all_zero
    # print(reg4[4])
    # out_reg4_para = str('')
    # for index in range(len(reg4_para)):
    #     out_reg4_para += reg4_para[index]
    out_reg4_para = reg4_para
    return out_reg4_para


'''
conv33compute: 用于给出reg4--7的值
    --m:输出通道数
    --c:输入通道数
    --inPictureSize: 输入图片尺寸
    --stride:步长
    --padding:padding
    --z1:上层激活的zp
    --z3:本层激活的zp
    --s3:本层激活的s3
'''


def conv33compute(m, c, dataSizeW, dataSizeB, inPictureSize, stride=1, padding=0, z1=0, z3=0, s3=0):
    channel_in = int(c)
    channel_in = '{:010b}'.format(channel_in)
    # stride = 1的情况下 输出feature map的尺寸
    outPictureSize_stride1 = int((inPictureSize - 3 + 2 * padding) / 1 + 1)
    outPictureSize_stride1 = '{:011b}'.format(outPictureSize_stride1)
    zero = '0'
    # 输出通道
    channel_out = int(m)
    channel_out = '{:010b}'.format(channel_out)
    # 输入通道+输出feature map尺寸+0+输出通道
    reg4_compute = str(channel_in) + str(outPictureSize_stride1) + zero + str(channel_out)

    out_reg4_compute = str('')
    for index in range(len(reg4_compute)):
        out_reg4_compute += reg4_compute[index]

    # ================================reg5==============================================
    # compute的reg5
    # 第一位: 是否需要通道补0 (本工程不用，当输入维度是RGB三通道，则需要通道补0)yolo headconv11在软件补0
    # 第二位:1位  是否需要 padding 的信号
    # is_padding:是否padding,如果padding时候等于1,不padding则等于0
    # 第三位:是否需要  stride  的信号,如果stride不等于1则is_stride是1,如果stride=1,则is_stride是0
    # 第四-六位:3 位   padding 添零的圈数 ，本工程为1
    # 最后11位图片输入的宽高数

    is_padding = 0
    is_stride = 0
    if padding != 0:
        is_padding = 1
    elif padding == 0:
        is_padding = 0
    if stride != 1:
        is_stride = 1
    elif stride == 1:
        is_stride = 0
    # 输入图片的尺寸
    inPictureSize = int(inPictureSize)
    inPictureSize = '{:011b}'.format(inPictureSize)
    # 0+padding?+stride?+001(padding 数)+0(15位)+输入图片尺寸(11位) 32位
    reg5_compute = '0' + str(is_padding) + str(is_stride) + '001' + '000000000000000' + str(inPictureSize)

    out_reg5_compute = str('')
    for index in range(len(reg5_compute)):
        out_reg5_compute += reg5_compute[index]

    # ================================reg6==============================================
    # 前16位全部为0,在8位是bias在coe中行数,最后的8位补0(改为leakrelu的reg)
    out_reg6_compute = reg6_leaky(s3)

    # bias_line = int(m / (dataSizeB / 32))
    # bias_line = '{:08b}'.format(bias_line)
    # reg6_compute = '0000000000000000' + str(bias_line) + '00000000'
    # out_reg6_compute = str('')
    # for index in range(len(reg6_compute)):
    #     # print(i)
    #     # exit()
    #     # print(reg4)
    #     out_reg6_compute += reg6_compute[index]
    # if (index + 1) % 4 == 0 and (index + 1) != len(reg6_compute):
    #     out_reg6_compute += '_'

    # ================================reg7==============================================
    # 前8位Padding中填0的值(若Z1不为0，则填0的值就是Z1)
    # 在8位z3的值
    # 最后16位，3 * 3卷积中1个卷积点的所有通道数的行数
    # 量化之后的padding 补零补的就是上层的zp
    if padding == 0:
        z1 = 0
    z1 = int(z1)
    z1 = '{:08b}'.format(z1)
    z3 = int(z3)
    z3 = '{:08b}'.format(z3)
    one_conv_line = int((m * c * 9) / ((dataSizeW / 8) * 9))
    one_conv_line = '{:016b}'.format(one_conv_line)
    reg7_compute = str(z1) + str(z3) + str(one_conv_line)
    out_reg7_compute = str('')
    for index in range(len(reg7_compute)):
        out_reg7_compute += reg7_compute[index]
        # if (index + 1) % 4 == 0 and (index + 1) != len(reg7_compute):
        #     out_reg7_compute += '_'
    return str(int(out_reg4_compute, 2)), str(int(out_reg5_compute, 2)), str(int(out_reg6_compute, 2)), str(
        int(out_reg7_compute, 2))


'''
conv11para:函数的功能和上面的3*3是一样的，只不过在reg4中用于存储权重和bias信息的位数不同
如果bias的行数小于255 则采用8bit存储 否则采用9bit存储
'''


def conv11para(m, c, dataSizeW, dataSizeB):
    # m是输出通道数,c是输入通道数,dataSizeW是权重每行多少bit,dataSizeB是bias每行多少bit
    # reg4：高16:一个卷积点的所有通道数的行数,在9:bias在coe中的行数
    one_conv_line = int((m * c) / (dataSizeW / 8))
    one_conv_line = '{:016b}'.format(one_conv_line)
    bias_line = int(m / (dataSizeB / 32))
    if (bias_line <= 255):
        bias_line = '{:08b}'.format(bias_line)
        print(bias_line)

        out_reg4_para = str(one_conv_line) + str(bias_line) + '00000000'
    else:
        bias_line = '{:09b}'.format(bias_line)
        bias_line = bias_line[1:] + bias_line[0]  # 把最高位换到最低位
        out_reg4_para = str(one_conv_line) + str(bias_line) + '0000000'

    return str(int(out_reg4_para, 2))


'''
conv11compute

'''


def conv11compute(m, c, dataSizeW, dataSizeB, inPictureSize, stride=1, padding=0, z1=0, z3=0, isleakrelu=0, s3=0):
    # ================================reg4==============================================
    # reg4高十位是输入通道数,接着11位是卷积后图片宽高,下一位是0,最后十位是输出通道数
    channel_in = int(c)

    channel_in = '{:010b}'.format(channel_in)
    outPictureSize = int((inPictureSize - 1 + 2 * padding) / stride + 1)
    outPictureSize = '{:011b}'.format(outPictureSize)

    zero = '0'
    channel_out = int(m)
    channel_out = '{:010b}'.format(channel_out)
    # one_conv_line是1个卷积点的所有通道数的行数

    out_reg4_compute = str(channel_in) + str(outPictureSize) + zero + str(channel_out)

    # ================================reg5==============================================
    # compute的reg5
    # 第一位: 是否需要通道补0 (本工程不用，当输入维度是RGB三通道，则需要通道补0)
    # 第二位:1位  是否需要 padding 的信号
    # 第三位:是否需要  stride 的信号
    # 第四-六位:3 位   padding 添零的圈数  (针对 5*5 卷积而设计的) ，本工程为1
    # 第七位:1位   是否需要 leakyrelu 。 1 为不用，0 为用
    # 在往后14位补0
    # 最后11位图片输入的宽高数

    is_padding = 0
    is_stride = 0
    if padding != 0:
        is_padding = 1
    elif padding == 0:
        is_padding = 0
    if stride != 1:
        is_stride = 1
    elif stride == 1:
        is_stride = 0

    inPictureSize = int(inPictureSize)
    inPictureSize = '{:011b}'.format(inPictureSize)
    out_reg5_compute = '0' + str(is_padding) + str(is_stride) + '001' + str(isleakrelu) + '00000000000000' + str(
        inPictureSize)

    # ================================reg6==============================================
    # reg_leakrelu
    out_reg6_compute = reg6_leaky(s3)
    # bias_line = int(m / (dataSizeB / 32))
    # if (bias_line <= 255):
    #     bias_line = '{:08b}'.format(bias_line)
    #     print(bias_line)
    #     # exit()
    #     out_reg6_compute = '0000000000000000' + str(bias_line) + '00000000'
    # else:
    #     bias_line = '{:09b}'.format(bias_line)
    #     bias_line = bias_line[1:] + bias_line[0]
    #     out_reg6_compute = '0000000000000000' + str(bias_line) + '0000000'

    # return str(int(out_reg4_compute, 2)), str(int(out_reg5_compute, 2)), str(int(out_reg6_compute, 2))

    # ================================reg7==============================================
    # 前8位为
    # 在8位z3的值
    # 最后16位， 为0

    z3 = int(z3)
    z3 = '{:08b}'.format(z3)
    all_zero = '0000000000000000'
    # one_conv_line = int((m * c) / (dataSizeW / 8))
    # one_conv_line = '{:016b}'.format(one_conv_line)

    out_reg7_compute = '00000000' + str(z3) + all_zero

    # print(out_reg5_compute)
    return str(int(out_reg4_compute, 2)), str(int(out_reg5_compute, 2)), str(int(out_reg6_compute, 2)), str(
        int(out_reg7_compute, 2))


def Focus(x):
    patch_top_left = x[..., ::2, ::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_right = x[..., 1::2, 1::2]
    x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
