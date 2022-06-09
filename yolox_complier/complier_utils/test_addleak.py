import numpy as np
import torch
import math
from ctypes import c_int32
from torch.nn.quantized import functional as qF


def red_leak_error(s3):
    # leakrelu修正表
    add_data = []
    data1 = torch.ones(16)

    for index in range(data1.shape[0]):
        data1[index] = -index * 10 - 5
    for index in range(data1.shape[0]):
        q_feature = torch.quantize_per_tensor(data1[index] * s3, scale=float(s3),
                                              zero_point=int(0), dtype=torch.qint8)
        out_leak = qF.leaky_relu(input=q_feature, negative_slope=0.1, inplace=True)
        out = out_leak.int_repr()
        out2 = data1[index] * 0.1
        # print(data1[index],np.round(out),np.round(out2))
        if np.round(out) > np.round(out2):
            add_data.append(1)
        elif np.round(out) < np.round(out2):
            add_data.append(-1)
        else:
            add_data.append(0)
    # exit()
    return add_data


def gen_int_bias_float(s1, s2, bias_float):
    aa = bias_float / s1
    bias = torch.div(aa, s2)
    return bias


def gen_int_bias(s1, s2, bias_float):
    aa = bias_float / s1
    bb = torch.div(aa, s2)  # 张量和标量做逐元素除法
    for i, m in enumerate(bb):
        bb[i] = round(m.item())
    bias = bb.int()

    return bias


def gen_B(S1, S2, S3):
    M = (S1 * S2) / S3
    M = M.numpy()
    # print(M, 53)
    # exit()
    daxiao = S2.shape[0]
    SCALE = np.zeros(daxiao, dtype=np.uint32, order='C')
    N_REAL = np.zeros(daxiao, dtype=np.uint32, order='C')
    for i, ii in enumerate(M):

        while not (ii >= 0.5 and ii <= 1.0):
            ii *= 2
        pass
        mmmm = ii * (2 ** 32)

        SCALE[i] = round(mmmm.astype(np.int32))

    for i, ii in enumerate(M):
        N_REAL[i] = round(math.log(SCALE[i] / ii, 2)) - 32 - 1
    # print(N_REAL)
    # exit()
    return N_REAL


def gen_M_N(S1, S2, S3):
    # print(S1,51)
    daxiao = S2.shape[0]
    M = np.zeros(daxiao, dtype=np.uint32, order='C')
    # M = torch.Tensor(S2.shape[0],dtype=torch.float)
    N_REAL = gen_B(S1, S2, S3)
    # print(N_REAL)

    M = np.zeros(S2.shape[0])
    for i, ii in enumerate(M):
        M[i] = (torch.round((S1 * S2[i]) / S3 * (2 ** (32 + N_REAL[i] + 1)))).numpy()
    # print(M.max())

    return M, N_REAL


def new_bias(z1, q2, bias):
    bias1 = z1 * q2
    shape = bias1.shape
    # print(shape) # torch.Size([64, 32, 3, 3])
    n_bias = np.zeros(shape[0], dtype=np.float, order='C')
    out_bias = np.zeros(shape[0], dtype=np.float, order='C')
    out_bias1 = np.zeros(shape[0], dtype=np.float, order='C')
    out_bias2 = np.zeros(shape[0], dtype=np.float, order='C')
    out_final = np.zeros(shape[0], dtype=np.float, order='C')
    for m in range(shape[0]):
        n_bias[m] = bias1[m, :, :, :].sum()
        n_bias[m] = bias[m] - n_bias[m]
    return n_bias


def new_bias_2(z1, q2, bias):
    # 新版bias
    bias1 = z1 * q2
    shape = bias1.shape
    n_bias = np.zeros(shape[0], dtype=np.float, order='C')
    for m in range(shape[0]):
        n_bias[m] = bias1[m, :, :, :].sum()
        n_bias[m] = (bias[m] - n_bias[m])
    # print(n_bias)
    # exit()
    daxiao = shape[0]
    SCALE = np.zeros(daxiao, dtype=np.float64, order='C')
    # N_REAL = np.zeros(daxiao, dtype=np.float32, order='C')
    N_REAL = []
    for i, ii in enumerate(n_bias):
        index = 0
        while not (abs(ii) >= (2 ** 23) and abs(ii) <= (2 ** 24)):

            if index >= 16:  # fpga里面最多移动16位,所有成到16就停止了,这样精度也够了
                print('-------------------------------------------------------')
                break
            else:
                ii *= 2
                index = index + 1
        N_REAL.append(index)
        SCALE[i] = round(ii)

    return SCALE, N_REAL


def new_bias_change(z1, q2, bias, s1, s2, s3, M):
    print(bias)
    bias1 = z1 * q2
    shape = bias1.shape
    n_bias = np.zeros(shape[0], dtype=np.float, order='C')
    for m in range(shape[0]):
        n_bias[m] = bias1[m, :, :, :].sum()
        n_bias[m] = bias[m] - n_bias[m]
        n_bias[m] = n_bias[m] * M[m]
    bias = n_bias.astype(np.int)

    return n_bias


def conv_naive(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):
    # 老板手写卷积,bias老板
    s = s1 * s2 / s3
    i_s = input.shape
    f_s = filter.shape
    M, N_REAL = gen_M_N(s1, s2, s3)
    bias_int = gen_int_bias(s1, s2, bias)
    bias_int = bias_int.numpy()
    q_weight = filter.numpy().astype(np.float)
    q_weight = torch.from_numpy(q_weight)
    bias = new_bias(z1, q_weight, bias_int)

    if compare == 1:
        # print('111111111111111')
        np.save('../compare_coe/SCALE', M)
        np.save('../compare_coe/N_REAL', N_REAL)
        np.save('../compare_coe/bias', bias)
        np.save('../compare_coe/weight', q_weight)
        exit()
    assert len(i_s) == 4, "plz input the correct image shape"
    assert len(f_s) == 4, "plz input the correct filters shape"
    assert i_s[1] == f_s[1], "input image channels are mismatch with filter's input channels"
    assert padding >= 0, "padding ? wrong !! "
    assert stride >= 1, "stride ? wrong !! "
    if padding > 0:
        hw = np.array((i_s[2] + padding * 2, i_s[3] + padding * 2))
        pad_img = np.ones((i_s[0], i_s[1], hw[0], hw[1]), dtype=np.float) * z1

        pad_img[:, :, padding:-padding, padding:-padding] = input
        input = pad_img
        i_s = input.shape
    # stride
    stride_hw = (np.array(i_s[2:]) - np.array(f_s[2:])) // stride + 1
    output_shape = np.concatenate([i_s[:1], f_s[:1], stride_hw])
    output = np.zeros(output_shape)
    batch, in_channel, img_h, img_w = i_s[0], i_s[1], output_shape[2], output_shape[3]
    out_channel, f_h, f_w = f_s[0], f_s[2], f_s[3]
    input = torch.as_tensor(input, dtype=torch.float)
    filter = torch.as_tensor(filter, dtype=torch.float)
    for n in range(batch):  # N
        for imh in range(img_h):  # H
            for imw in range(img_w):  # W
                for ouc in range(out_channel):  # M
                    a = (input[n, :in_channel, imh * stride:f_h + imh * stride,
                         imw * stride:imw * stride + f_w] * filter[ouc, :, :, :]).sum()
                    a = a + bias[ouc]
                    # if (qF_out!+):
                    #     print('bias之后的:',a)
                    a = a * M[ouc]
                    # a = a * s1 * s2[ouc] / s3
                    # print(a)

                    # if (n == 0 and imh == 3 and imw == 4 and ouc == 7):
                    #     print('成s之后的值',a)
                    a = torch.round(a / (2 ** (32 + N_REAL[ouc] + 1)))
                    # if (n == 0 and imh == 3 and imw == 4 and ouc == 7):
                    #     print('加上z3之前的值也就是shift之后的值',a)
                    a = a + z3
                    if a >= 255:
                        a = 255
                    if a <= 0:
                        a = 0
                    # if (n == 0 and imh == 3 and imw ==4 and ouc == 7):
                    #     print('relu的结果',a.type())
                    a = a - z3
                    if a < 0:
                        # print(a)
                        a = torch.round(a * 0.1)

                    # if (n == 0 and imh == 3 and imw == 4 and ouc == 7):
                    #     print('加z3之前结果',a)
                    a = a + z3
                    if a >= 255:
                        a = 255
                    if a <= 0:
                        a = 0
                    # if (n == 0 and imh == 3 and imw == 4 and ouc == 7):
                    #     print('最后结果',a)
                    output[n][ouc][imh][imw] = a

    output_tensor = torch.tensor(output, dtype=torch.uint8)
    # torch.nn.funtional.leaky_relu(input=output, negative_slope=0.1, inplace=True)
    # print(output)
    return output_tensor


def conv_naive_onlyconv(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):
    # 单独卷积结果
    # print(input, filter.shape, 81)
    # exit()
    s = s1 * s2 / s3
    # print(s)
    # exit()
    i_s = input.shape
    # c_out,c_in,kh,kw
    f_s = filter.shape
    bias_int = gen_int_bias(s1, s2, bias)
    #  print(bias_int[0])
    # WWWW
    # print(bias_int)
    # exit()
    bias_int = bias_int.numpy()

    q_weight = filter.numpy().astype(np.float)
    q_weight = torch.from_numpy(q_weight)
    # print(q_weight)
    # exit()
    bias = new_bias(z1, q_weight, bias_int)

    M, N_REAL = gen_M_N(s1, s2, s3)
    # print(M,N_REAL,bias,q_weight)
    # exit()
    print(compare)
    if compare == 1:
        # print('111111111111111')
        np.save('../compare_coe/SCALE', M)
        np.save('../compare_coe/N_REAL', N_REAL)
        np.save('../compare_coe/bias', bias)
        np.save('../compare_coe/weight', q_weight)
        exit()
    assert len(i_s) == 4, "plz input the correct image shape"
    assert len(f_s) == 4, "plz input the correct filters shape"
    assert i_s[1] == f_s[1], "input image channels are mismatch with filter's input channels"
    assert padding >= 0, "padding ? wrong !! "
    assert stride >= 1, "stride ? wrong !! "

    # padding
    if padding > 0:
        hw = np.array((i_s[2] + padding * 2, i_s[3] + padding * 2))
        pad_img = np.ones((i_s[0], i_s[1], hw[0], hw[1]), dtype=np.float) * z1
        # print(input)
        pad_img[:, :, padding:-padding, padding:-padding] = input
        input = pad_img
        i_s = input.shape
    # stride
    stride_hw = (np.array(i_s[2:]) - np.array(f_s[2:])) // stride + 1
    output_shape = np.concatenate([i_s[:1], f_s[:1], stride_hw])
    output = np.zeros(output_shape)
    batch, in_channel, img_h, img_w = i_s[0], i_s[1], output_shape[2], output_shape[3]
    out_channel, f_h, f_w = f_s[0], f_s[2], f_s[3]
    # print(filter)
    input = torch.as_tensor(input, dtype=torch.float)
    filter = torch.as_tensor(filter, dtype=torch.float)

    for n in range(batch):  # N
        for imh in range(img_h):  # H
            for imw in range(img_w):  # W
                for ouc in range(out_channel):  # M
                    a = (input[n, :in_channel, imh * stride:f_h + imh * stride,
                         imw * stride:imw * stride + f_w] * filter[ouc, :, :, :]).sum()

                    output[n][ouc][imh][imw] = a

    output_tensor = torch.tensor(output, dtype=torch.int32)

    return output_tensor


def conv_naive_relu(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):
    s = s1 * s2 / s3
    i_s = input.shape
    f_s = filter.shape
    bias_int = gen_int_bias(s1, s2, bias)
    bias_int = bias_int.numpy()
    q_weight = filter.numpy().astype(np.float)
    q_weight = torch.from_numpy(q_weight)
    bias = new_bias(z1, q_weight, bias_int)
    M, N_REAL = gen_M_N(s1, s2, s3)
    if compare == 1:
        # print('111111111111111')
        np.save('../compare_coe/SCALE', M)
        np.save('../compare_coe/N_REAL', N_REAL)
        np.save('../compare_coe/bias', bias)
        np.save('../compare_coe/weight', q_weight)
        exit()
    assert len(i_s) == 4, "plz input the correct image shape"
    assert len(f_s) == 4, "plz input the correct filters shape"
    assert i_s[1] == f_s[1], "input image channels are mismatch with filter's input channels"
    assert padding >= 0, "padding ? wrong !! "
    assert stride >= 1, "stride ? wrong !! "
    if padding > 0:
        hw = np.array((i_s[2] + padding * 2, i_s[3] + padding * 2))
        pad_img = np.ones((i_s[0], i_s[1], hw[0], hw[1]), dtype=np.float) * z1

        pad_img[:, :, padding:-padding, padding:-padding] = input
        input = pad_img
        i_s = input.shape
    # stride
    stride_hw = (np.array(i_s[2:]) - np.array(f_s[2:])) // stride + 1
    output_shape = np.concatenate([i_s[:1], f_s[:1], stride_hw])
    output = np.zeros(output_shape)
    batch, in_channel, img_h, img_w = i_s[0], i_s[1], output_shape[2], output_shape[3]
    out_channel, f_h, f_w = f_s[0], f_s[2], f_s[3]
    input = torch.as_tensor(input, dtype=torch.float)
    filter = torch.as_tensor(filter, dtype=torch.float)
    for n in range(batch):  # N
        for imh in range(img_h):  # H
            for imw in range(img_w):  # W
                for ouc in range(out_channel):  # M
                    a = (input[n, :in_channel, imh * stride:f_h + imh * stride,
                         imw * stride:imw * stride + f_w] * filter[ouc, :, :, :]).sum()
                    a = a + bias[ouc]
                    a = a * M[ouc]
                    a = torch.round(a / (2 ** (32 + N_REAL[ouc] + 1)))
                    a = a + z3
                    if a >= 255:
                        a = 255
                    if a <= 0:
                        a = 0

                    output[n][ouc][imh][imw] = a

    output_tensor = torch.tensor(output, dtype=torch.uint8)
    # torch.nn.funtional.leaky_relu(input=output, negative_slope=0.1, inplace=True)
    # print(output)
    return output_tensor


def conv_naive_red_float(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):
    add_data = red_leak_error(s3)
    print(add_data)
    s = s1 * s2 / s3
    i_s = input.shape
    f_s = filter.shape
    # bias1
    bias_int = gen_int_bias_float(s1, s2, bias)
    # print(bias_int)
    # exit()
    bias_int = bias_int.numpy()
    q_weight = filter.numpy().astype(np.float)
    q_weight = torch.from_numpy(q_weight)
    M_bias, N_REAL_bias = new_bias_2(z1, q_weight, bias_int)
    # print(M_bias)
    # print(N_REAL_bias)
    # exit()
    # M_bias, N_REAL_bias = new_bias_2(z1, q_weight, bias_int)
    M, N_REAL = gen_M_N(s1, s2, s3)
    if compare == 1:
        # print('111111111111111')
        np.save('../compare_coe/SCALE', M)
        np.save('../compare_coe/N_REAL', N_REAL)
        np.save('../compare_coe/bias', bias)
        np.save('../compare_coe/weight', q_weight)
        exit()
    assert len(i_s) == 4, "plz input the correct image shape"
    assert len(f_s) == 4, "plz input the correct filters shape"
    assert i_s[1] == f_s[1], "input image channels are mismatch with filter's input channels"
    assert padding >= 0, "padding ? wrong !! "
    assert stride >= 1, "stride ? wrong !! "
    if padding > 0:
        hw = np.array((i_s[2] + padding * 2, i_s[3] + padding * 2))
        pad_img = np.ones((i_s[0], i_s[1], hw[0], hw[1]), dtype=np.float) * z1

        pad_img[:, :, padding:-padding, padding:-padding] = input
        input = pad_img
        i_s = input.shape
    bias1 = np.zeros(s2.shape, dtype=np.float, order='C')
    bias2 = np.zeros(s2.shape, dtype=np.float, order='C')
    # stride
    stride_hw = (np.array(i_s[2:]) - np.array(f_s[2:])) // stride + 1
    output_shape = np.concatenate([i_s[:1], f_s[:1], stride_hw])
    output = np.zeros(output_shape)
    batch, in_channel, img_h, img_w = i_s[0], i_s[1], output_shape[2], output_shape[3]
    out_channel, f_h, f_w = f_s[0], f_s[2], f_s[3]
    input = torch.as_tensor(input, dtype=torch.float)
    filter = torch.as_tensor(filter, dtype=torch.float)
    for n in range(batch):  # N
        for imh in range(img_h):  # H
            for imw in range(img_w):  # W
                for ouc in range(out_channel):  # M
                    a = (input[n, :in_channel, imh * stride:f_h + imh * stride,
                         imw * stride:imw * stride + f_w] * filter[ouc, :, :, :]).sum()
                    aaa = torch.tensor(M_bias[ouc])
                    aaa = aaa / (2 ** N_REAL_bias[ouc])
                    a_bias = a + aaa
                    a_m = torch.round(a_bias * M[ouc])
                    a_s = torch.round(a_m / (2 ** (32 + N_REAL[ouc] + 1)))
                    a_z = a_s + z3
                    if a_z >= 255:
                        a_z = 255
                    if a_z <= 0:
                        a_z = 0
                    a_z = a_z - z3
                    if a_z < 0:
                        if (a_z - 5) % (-10) == 0 and a_z != 0:
                            for index in range(16):
                                if index + 1 == (abs(a_z / 5) + 1) / 2:
                                    a_z = torch.round(a_z * 0.1) + add_data[index]
                        else:
                            a_z = torch.round(a_z * 0.1)
                    a_leak_z = a_z + z3
                    if a_leak_z >= 255:
                        a_leak_z = 255
                    if a_leak_z <= 0:
                        a_leak_z = 0
                    # if (qF_out[n][ouc][imh][imw] != a_leak_z):
                    #   with open('error_realseed3_640.txt', "a+") as fp:
                    #     fp.write(str(qF_out[n][ouc][imh][imw].numpy()))
                    #     fp.write('   ')
                    #     fp.write('   ')
                    #     # fp.write('手写卷积结果:')
                    #     fp.write(str(a_leak_z.numpy()))
                    #     fp.write('   ')
                    #     fp.write('   ')
                    #     # fp.write('卷积结果:')
                    #     fp.write(str(a.numpy()))
                    #     fp.write('   ')
                    #     # fp.write('加上bias结果:')
                    #     fp.write(str(a_bias.numpy()))
                    #     fp.write('   ')
                    #     # fp.write('乘scale之后结果:')
                    #     fp.write(str(a_m.numpy()))
                    #     fp.write('   ')
                    #     # fp.write('shift之后结果:')
                    #     fp.write(str(a_s.numpy()))
                    #     fp.write('   ')
                    #     fp.write('   ')
                    #     fp.write(str(a_z.numpy()))
                    #     fp.write('   ')
                    #     fp.write('   ')
                    #     # fp.write('z3的值:')
                    #     fp.write(str(z3.numpy()))
                    #     fp.write('   ')
                    #     fp.write('   ')
                    #     fp.write(str(bias[ouc]))
                    #     fp.write('   ')
                    #     fp.write('   ')
                    #     fp.write(str(s2[ouc].numpy()))
                    #     fp.write('   ')
                    #     fp.write('   ')
                    #     fp.write(str(n))
                    #     fp.write('   ')
                    #     fp.write(str(ouc))
                    #     fp.write('   ')
                    #     fp.write(str(imh))
                    #     fp.write('   ')
                    #     fp.write(str(imw))
                    #     fp.write('\n')
                    output[n][ouc][imh][imw] = a_m

    output_tensor = torch.tensor(output, dtype=torch.int64)
    return output_tensor


def conv_naive_Simulation(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):  # 仿真器,与fpga结果一样
    add_data = red_leak_error(s3)
    s = s1 * s2 / s3
    i_s = input.shape
    f_s = filter.shape
    # bias1
    bias_int = gen_int_bias_float(s1, s2, bias)

    bias_int = bias_int.numpy()
    q_weight = filter.numpy().astype(np.float)
    q_weight = torch.from_numpy(q_weight)
    M_bias, N_REAL_bias = new_bias_2(z1, q_weight, bias_int)
    # print(M_bias)
    # print(N_REAL_bias)
    # exit()
    # M_bias, N_REAL_bias = new_bias_2(z1, q_weight, bias_int)
    M, N_REAL = gen_M_N(s1, s2, s3)
    if compare == 1:
        # print('111111111111111')
        np.save('../compare_coe/SCALE', M)
        np.save('../compare_coe/N_REAL', N_REAL)
        np.save('../compare_coe/bias', bias)
        np.save('../compare_coe/weight', q_weight)
        exit()
    assert len(i_s) == 4, "plz input the correct image shape"
    assert len(f_s) == 4, "plz input the correct filters shape"
    assert i_s[1] == f_s[1], "input image channels are mismatch with filter's input channels"
    assert padding >= 0, "padding ? wrong !! "
    assert stride >= 1, "stride ? wrong !! "
    if padding > 0:
        hw = np.array((i_s[2] + padding * 2, i_s[3] + padding * 2))
        pad_img = np.ones((i_s[0], i_s[1], hw[0], hw[1]), dtype=np.float) * z1

        pad_img[:, :, padding:-padding, padding:-padding] = input
        input = pad_img
        i_s = input.shape
    bias1 = np.zeros(s2.shape, dtype=np.float, order='C')
    bias2 = np.zeros(s2.shape, dtype=np.float, order='C')
    # stride
    stride_hw = (np.array(i_s[2:]) - np.array(f_s[2:])) // stride + 1
    output_shape = np.concatenate([i_s[:1], f_s[:1], stride_hw])
    output = np.zeros(output_shape)
    batch, in_channel, img_h, img_w = i_s[0], i_s[1], output_shape[2], output_shape[3]
    out_channel, f_h, f_w = f_s[0], f_s[2], f_s[3]
    input = torch.as_tensor(input, dtype=torch.float)
    filter = torch.as_tensor(filter, dtype=torch.float)
    for n in range(batch):  # N
        for imh in range(img_h):  # H
            for imw in range(img_w):  # W
                for ouc in range(out_channel):  # M
                    a = (input[n, :in_channel, imh * stride:f_h + imh * stride,
                         imw * stride:imw * stride + f_w] * filter[ouc, :, :, :]).sum()
                    aaa = torch.tensor(M_bias[ouc])
                    aaa = aaa / (2 ** N_REAL_bias[ouc])
                    a_bias = a + aaa
                    a_m = a_bias * M[ouc]
                    a_s = torch.round(a_m / (2 ** (32 + N_REAL[ouc] + 1)))
                    a_z = a_s + z3
                    if a_z >= 255:
                        a_z = 255
                    if a_z <= 0:
                        a_z = 0
                    a_z = a_z - z3
                    if a_z < 0:
                        if (a_z - 5) % (-10) == 0 and a_z != 0:
                            for index in range(16):
                                if index + 1 == (abs(a_z / 5) + 1) / 2:
                                    a_z = torch.round(a_z * 0.1) + add_data[index]
                        else:
                            a_z = torch.round(a_z * 0.1)
                    a_leak_z = a_z + z3
                    if a_leak_z >= 255:
                        a_leak_z = 255
                    if a_leak_z <= 0:
                        a_leak_z = 0

                    output[n][ouc][imh][imw] = a_leak_z

    output_tensor = torch.tensor(output, dtype=torch.uint8)
    return output_tensor
