import numpy as np
import torch
import math
import time
from torch.nn.quantized import functional as qF
from torch.nn.functional import conv2d

import inspect
import ctypes
import threading
import copy


def red_leak_error(s3):
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
        if np.round(out) > np.round(out2):
            add_data.append(1)
        elif np.round(out) < np.round(out2):
            add_data.append(-1)
        else:
            add_data.append(0)
    return add_data


def gen_int_bias_float(s1, s2, bias_float):
    aa = bias_float / s1
    bias = torch.div(aa, s2)
    return bias


def gen_int_bias(s1, s2, bias_float):
    aa = bias_float / s1
    bb = torch.div(aa, s2)
    for i, m in enumerate(bb):
        bb[i] = round(m.item())
    bias = bb.int()
    return bias


def gen_B(S1, S2, S3):
    M = (S1 * S2) / S3
    M = M.numpy()
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
    return N_REAL


def gen_M_N(S1, S2, S3):
    daxiao = S2.shape[0]
    M = np.zeros(daxiao, dtype=np.uint32, order='C')
    N_REAL = gen_B(S1, S2, S3)
    M = np.zeros(S2.shape[0])
    for i, ii in enumerate(M):
        M[i] = (torch.round((S1 * S2[i]) / S3 * (2 ** (32 + N_REAL[i] + 1)))).numpy()
    return M, N_REAL


def new_bias(z1, q2, bias):
    bias1 = z1 * q2
    shape = bias1.shape
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
    bias1 = z1 * q2
    shape = bias1.shape
    n_bias = np.zeros(shape[0], dtype=np.float, order='C')
    for m in range(shape[0]):
        n_bias[m] = bias1[m, :, :, :].sum()
        n_bias[m] = (bias[m] - n_bias[m])
    daxiao = shape[0]
    SCALE = np.zeros(daxiao, dtype=np.float64, order='C')
    N_REAL = []
    for i, ii in enumerate(n_bias):
        index = 0
        while not (abs(ii) >= (2 ** 23) and abs(ii) <= (2 ** 24)):
            if index >= 16:  # fpga里面最多移动16位,所有成到16就停止了,这样精度也够了
                break
            else:
                ii *= 2
                index = index + 1
        N_REAL.append(index)
        SCALE[i] = round(ii)

    return SCALE, N_REAL


def new_bias_change(z1, q2, bias, s1, s2, s3, M):
    bias1 = z1 * q2
    shape = bias1.shape
    n_bias = np.zeros(shape[0], dtype=np.float, order='C')
    for m in range(shape[0]):
        n_bias[m] = bias1[m, :, :, :].sum()
        n_bias[m] = bias[m] - n_bias[m]
        n_bias[m] = n_bias[m] * M[m]

    bias = n_bias.astype(np.int)
    return n_bias


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    # if res == 0:
    #     raise ValueError("invalid thread id")
    if res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def conv_naive_relu(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):
    z3 = z3.numpy()

    s = s1 * s2 / s3
    i_s = input.shape
    f_s = filter.shape
    bias_int = gen_int_bias_float(s1, s2, bias)
    bias_int = bias_int.numpy()
    q_weight = filter.numpy().astype(np.float)
    q_weight = torch.from_numpy(q_weight)
    M_bias, N_REAL_bias = new_bias_2(z1, q_weight, bias_int)
    M, N_REAL = gen_M_N(s1, s2, s3)
    type_a = type(M)
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
    # input=torch.as_tensor(input,dtype=torch.float)
    # filter = torch.as_tensor(filter, dtype=torch.float)
    filter = filter.numpy().astype(np.float)
    #
    for n in range(batch):  # N
        for imh in range(img_h):  # H
            for imw in range(img_w):  # W
                for ouc in range(out_channel):  # M
                    a = (input[n, :in_channel, imh * stride:f_h + imh * stride,
                         imw * stride:imw * stride + f_w] * filter[ouc, :, :, :]).sum()
                    aaa = M_bias[ouc]
                    aaa = aaa / (2 ** N_REAL_bias[ouc])
                    a = a + aaa
                    a = a * M[ouc]

                    a = round(a / (2 ** (32 + N_REAL[ouc] + 1)))
                    a = a + z3
                    if a >= 255:
                        a = 255
                    if a <= 0:
                        a = 0
                    #
                    output[n][ouc][imh][imw] = a

    output_tensor = torch.tensor(output, dtype=torch.uint8)
    return output_tensor


def square(x):
    x = int(x)
    return add_data[x]


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def compute_leak(a_x):
    # print(a_x[((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0))])
    a_x[((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0))] = np.round(
        a_x[((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0) & (a_x != 0))] * 0.1)
    return a_x[((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0))]


def compute_leak_red_error(a_x):
    # print(error_index[0])
    # print(error_index[1])
    add_index = (a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] - 5) / -10 - 1
    a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] = np.round(
        a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] * 0.1) + list(map(square, add_index))
    return 1


def conv_naive_Simulation_speed(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):
    z3 = z3.numpy()
    before_conv = time.time()
    # print("量化之前时间:", time111 - before_conv)
    global add_data
    add_data = red_leak_error(s3)
    # print(add_data)
    s = s1 * s2 / s3
    i_s = input.shape
    f_s = filter.shape
    # bias1
    bias_int = gen_int_bias_float(s1, s2, bias)
    bias_int = bias_int.numpy()
    q_weight = filter.numpy().astype(np.float)
    q_weight = torch.from_numpy(q_weight)
    M_bias, N_REAL_bias = new_bias_2(z1, q_weight, bias_int)
    M, N_REAL = gen_M_N(s1, s2, s3)
    if compare == 1:
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
    conv_out = torch.nn.functional.conv2d(input=input, weight=filter, stride=stride)
    conv_out = conv_out.numpy()
    # print(img_h)
    # exit()
    # out_z = np.ones((1,1,img_h,img_w))
    for n in range(batch):  # N
        for ouc in range(out_channel):  # M
            # for imh in range(img_h):  # H
            #     for imw in range(img_w):  # W
            a = conv_out[n, ouc, :, :]
            aaa = M_bias[ouc]
            aaa = aaa / (2 ** N_REAL_bias[ouc])
            a_bias = a + aaa
            a_m = a_bias * M[ouc]
            a_s = np.round(a_m / (2 ** (32 + N_REAL[ouc] + 1)))
            a_x = a_s + z3
            a_x[a_x >= 255] = 255
            a_x[a_x <= 0] = 0
            a_x = a_x - z3
            # print(a_x)
            # old_x = copy.deepcopy(a_x)

            # np.where(((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0)), np.round(a_x * 0.1), a_x)
            # np.where(((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0)), np.round(a_x * 0.1), a_x)
            error_index = np.where(
                ((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0)) & ((np.round(a_x * 0.1) - 5) % (-10) == 0))
            # print(error_index)
            change_out = []

            # print(a_x)
            for index in range(len(error_index[0])):
                out_data = a_x[int(error_index[0][index])][int(error_index[1][index])]
                change_out.append(out_data)
            # bbb=int(list(error_index)[1].tolist())
            # print(a_x[aaa][bbb])
            # print(old_x[(old_x < 0) & ((old_x - 5) % (-10) == 0) & (old_x != 0)])
            # old_x2 = copy.deepcopy(a_x)
            # print(a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)])
            # print(add_data)
            # if ouc == 19:
            #     print(a_x[64, 5])
            # old_x = copy.deepcopy(a_x[((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0))])
            # map(square,a_x)
            # add_index = (a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] - 5) / -10-1
            # a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] = np.round(a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)]*0.1)+list(map(square, add_index))
            # t1 = threading.Thread(target=compute_leak, args=(a_x,))
            a_x[((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0))] = np.round(
                a_x[((a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0) & (a_x != 0))] * 0.1)
            # print(a_x)
            add_index = (a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] - 5) / -10 - 1
            a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] = np.round(
                a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] * 0.1) + list(map(square, add_index))
            # t2 = threading.Thread(target=compute_leak_red_error, args=(a_x,))
            # print(old_x[(old_x < 0) & ((old_x - 5) % (-10) == 0) & (old_x != 0)])
            # print(a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)])
            # t1.start()
            # t2.start()
            # print(old_x[(old_x < 0) & ((old_x - 5) % (-10) == 0) & (old_x != 0)])

            # a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] = old_x[(old_x < 0) & ((old_x - 5) % (-10) == 0) & (old_x != 0)]

            for index in range(len(error_index[0])):
                # print(change_out[index])
                a_x[int(error_index[0][index])][int(error_index[1][index])] = round(change_out[index] * 0.1)
                # change_out.append(out_data)
            # if ouc==19:
            #     print(1111111111111111111111111111111111111)
            #     print(a_x[0][0])
            #     exit()
            # t1.join()
            # t2.join()
            # stop_thread(t1)
            # stop_thread(t2)
            # _async_raise(thread.ident, SystemExit)
            # a_x[(a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0)] = np.round(
            #     (a_x[(a_x < 0) & ((a_x - 5) % (-10) != 0) & (a_x != 0)]) * 0.1)
            # a_x=list(map(square,a_x))
            # exit()
            # print(a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)])
            # length_change = len((a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] - 5) / -10)
            # add_list = []
            # for index in range(length_change):
            #     add_list.append(int((a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)][index] - 5) / -10))
            # a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] = np.round(
            #     a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] * 0.1) + add_list
            # a_x[(a_x < 0) & ((a_x - 5) % (-10) != 0) ] = np.round(
            #     a_x[(a_x < 0) & ((a_x - 5) % (-10) != 0) ] * 0.1)
            # print(a_x)
            # a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)] = 1
            # print((a_x[(a_x < 0) & ((a_x - 5) % (-10) == 0) & (a_x != 0)]-5)/-10-1)
            # for imh in range(img_h):
            #      for imw in range(img_w):
            #         a_z = a_x_old[imh, imw]
            #         # print(a_z)
            #         if a_z < 0:
            #             if (a_z - 5) % (-10) == 0 and a_z != 0:
            #                 # print(1111111111)
            #                 for index in range(16):
            #                     if index + 1 == (abs(a_z / 5) + 1) / 2:
            #                             a_x[imh, imw]=a_x[imh,imw]+add_data[index]
            a_x = a_x + z3
            a_x[a_x >= 255] = 255
            a_x[a_x <= 0] = 0
            # a_x = a_x-z3
            #         a_x[a_x<=0] = a_x*0.1
            #         for imh in range(img_h):
            #             for imw in range(img_w):
            #                 a_z = a_x[imh, imw]
            #                 if a_z < 0:
            # if (a_z - 5) % (-10) == 0 and a_z != 0:
            #     for index in range(16):
            #         if index + 1 == (abs(a_z / 5) + 1) / 2:
            #             a_z = round(a_z * 0.1) + add_data[index]
            # else:
            #         a_z = round(a_z * 0.1)
            # a_leak_z = a_z + z3
            # if a_leak_z >= 255:
            #     a_leak_z = 255
            # if a_leak_z <= 0:
            #     a_leak_z = 0
            # print(a_x[a_x==67])
            # if ouc == 30:
            #    print(a_x[64,5])
            # if ouc == 61:
            #     print(a_x[24, 52])
            output[n, ouc, :, :] = a_x
    # print("量化时间:", time.time() - before_conv)
    output_tensor = torch.tensor(output, dtype=torch.uint8)
    return output_tensor


def conv_naive_Simulation(input, filter, padding, stride, bias, s1, s2, s3, z1, z2, z3, compare):
    z3 = z3.numpy()
    before_conv = time.time()
    # print("量化之前时间:", time111 - before_conv)
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
    M, N_REAL = gen_M_N(s1, s2, s3)
    if compare == 1:
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
    filter = filter.numpy().astype(np.float)
    for n in range(batch):  # b
        for imh in range(img_h):  # H
            for imw in range(img_w):  # W
                for ouc in range(out_channel):  # M
                    a = (input[n, :in_channel, imh * stride:f_h + imh * stride,
                         imw * stride:imw * stride + f_w] * filter[ouc, :, :, :]).sum()
                    aaa = M_bias[ouc]
                    aaa = aaa / (2 ** N_REAL_bias[ouc])
                    a_bias = a + aaa
                    a_m = round(a_bias * M[ouc])
                    a_s = round(a_m / (2 ** (32 + N_REAL[ouc] + 1)))
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
                                    a_z = round(a_z * 0.1) + add_data[index]
                        else:
                            a_z = round(a_z * 0.1)
                    a_leak_z = a_z + z3
                    if a_leak_z >= 255:
                        a_leak_z = 255
                    if a_leak_z <= 0:
                        a_leak_z = 0
                    output[n][ouc][imh][imw] = a_leak_z
    # print("量化时间:",time.time()-before_conv)
    output_tensor = torch.tensor(output, dtype=torch.uint8)
    return output_tensor
