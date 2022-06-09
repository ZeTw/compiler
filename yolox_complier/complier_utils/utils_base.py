import time

import numpy as np
import torch


def gen_coe(coe_name, input):
    '''

    :param coe_name: 写入的coe文件名称
    :param input: 想要写入的结果，api结果需要调用int_repr()方法，手写卷积则不需要
    :return:
    '''
    shape = input.shape
    out = []
    print('start gen coe file:{}'.format(coe_name))
    with open(coe_name, "w+") as fp:
        for r in range(shape[2]):  # hang
            for c in range(shape[3]):  # lie
                for ch in range(shape[1]):  # channel
                    for n in range(shape[0]):  # image_num
                        out.append(input[n][ch][r][c])
                        if len(out) == 8:  # 8
                            out.reverse()
                            for m in out:
                                m = m.item()
                                fp.write('%02x' % m)
                            fp.write(',\n')
                            out = []


def focus(x):
    patch_top_left = x[..., ::2, ::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_right = x[..., 1::2, 1::2]
    return patch_top_left, patch_bot_left, patch_top_right, patch_bot_right


def np2tensor(x):
    tensor_py = torch.from_numpy(np.load(x))
    return tensor_py


def quant_cat(cat1, cat2, cat1_scale, cat2_scale, cat3_scale, cat1_zero_point, cat2_zero_point, cat3_zero_point):
    '''

    :param cat1: 需要 cat的feature1
    :param cat2: 需要cat的feature2
    :param cat1_scale: feature1的scale
    :param cat2_scale: feature2的scale
    :param cat3_scale: cat之后feature的scale
    :param cat1_zero_point: feature1的zero_point
    :param cat2_zero_point: feature2的zero_point
    :param cat3_zero_point: cat之后feature的zero_point
    :return:  cat之后的feature
    '''
    cat_start = time.time()
    cat1 = cat1.type(torch.float).numpy()
    cat2 = cat2.type(torch.float).numpy()
    zero_point_one = (cat3_scale / cat1_scale) * cat3_zero_point - cat1_zero_point
    zero_point_one = (torch.round(zero_point_one * (2 ** 16))).numpy()
    zero_point_one = cat1 * (2 ** 16) + zero_point_one
    zero_point_one = torch.as_tensor(zero_point_one, dtype=torch.int32).numpy()

    M1 = (torch.round((cat1_scale / cat3_scale) * (2 ** 16)))  # float
    M1 = M1.numpy()
    cat1 = zero_point_one * M1
    # cat1 = cat1 / (2 ** 32)
    # cat1 = np.round(cat1)
    # cat1 = cat1.astype(np.int32)
    # cat1[cat1 >= 255] = 255
    # cat1[cat1 <= 0] = 0
    zero_point_two = ((cat3_scale / cat2_scale) * cat3_zero_point - cat2_zero_point).numpy()
    zero_point_two = (np.round(zero_point_two * (2 ** 16)))
    zero_point_two = zero_point_two.astype(np.int32)
    zero_point_two = cat2 * (2 ** 16) + zero_point_two
    zero_point_two = (torch.as_tensor(zero_point_two, dtype=torch.int32)).numpy()

    M2 = (torch.round((cat2_scale / cat3_scale) * (2 ** 16))).numpy()  # float
    cat2 = zero_point_two * M2
    # cat2 = cat2 / (2 ** 32)
    # cat2 = np.round(cat2)
    # cat2 = cat2.astype(np.int32)
    # cat2[cat2 >= 255] = 255
    # cat2[cat2 <= 0] = 0
    cat1 = torch.from_numpy(cat1)
    cat2 = torch.from_numpy(cat2)
    out = torch.cat([cat1, cat2], 1)
    out /= (2 ** 32)
    out = torch.round(out)
    out[out >= 255] = 255
    out[out <= 0] = 0
    # out = out.type(torch.uint8)
    # out = torch.tensor(out, dtype=torch.uint8)
    out.to(torch.uint8)
    cat_end = time.time()
    print("cat的时间:", cat_end - cat_start)
    return out


#  add5 先相加，再右移，然后round
def quant_add(feature1, feature2, add_scale1, add_scale2, add_scale3, add_zp1, add_zp2,
              add_zp3):
    feature1 = feature1.type(torch.float).numpy()
    feature2 = feature2.type(torch.float).numpy()
    zero_point_one = (add_scale3 / add_scale1) * add_zp3 - add_zp1
    zero_point_one = (torch.round(zero_point_one * (2 ** 16))).numpy()
    result1 = feature1 * (2 ** 16) + zero_point_one

    result1 = torch.as_tensor(result1, dtype=torch.int32).numpy()
    M1 = ((add_scale1 / add_scale3) * (2 ** 16)).numpy()
    result1 = M1 * result1

    zero_point_two = (-add_zp2 * (2 ** 16)).numpy()
    result2 = feature2 * (2 ** 16) + zero_point_two
    result2 = torch.as_tensor(result2, dtype=torch.int32).numpy()
    M2 = ((add_scale2 / add_scale3) * (2 ** 16)).numpy()
    result2 = M2 * result2

    result = np.round(np.add(result1, result2))
    result /= (2 ** 32)
    result = np.round(result)
    result = result.astype(np.int32)
    result = torch.from_numpy(result)

    # result = result / (2 ** 32)
    result[result >= 255] = 255
    result[result <= 0] = 0
    # result = torch.tensor(result, dtype=torch.uint8)
    result.to(torch.uint8)
    return result


def reg_add(add_scale1, add_scale2, add_scale3, add_zp1, add_zp2, add_zp3):
    zero_point_one = (add_scale3 / add_scale1) * add_zp3 - add_zp1
    zero_point_one = (torch.round(zero_point_one * (2 ** 16)))
    zero_point_one = zero_point_one.numpy().astype(np.uint32)
    M1 = (torch.round((add_scale1 / add_scale3) * (2 ** 16)))
    M1 = M1.numpy().astype(np.uint32)

    zero_point_two = torch.round(-add_zp2 * (2 ** 16))
    zero_point_two = zero_point_two.numpy().astype(np.uint32)
    M2 = (torch.round((add_scale2 / add_scale3)) * (2 ** 16))
    M2 = M2.numpy().astype(np.uint32)
    return M1, M2, zero_point_one, zero_point_two


def ins64to32(ins_64path, ins_32path):
    '''

    :param ins_64path: 输入64位指令文件
    :param ins_32path: 生成的32位指令文件
    :return:
    '''
    torch.set_printoptions(profile="full")
    out_api = open(ins_64path)
    outapi = out_api.read().splitlines()
    with open(ins_32path, 'a+') as fp:
        for m in range(0, len(outapi)):
            data = outapi[m].rsplit(',')
            data = data[0]
            fp.write('0x')
            fp.write(data[8:])
            fp.write('\n')
            fp.write('0x')
            fp.write(data[:8])
            fp.write('\n')
    out_api.close()


def add_channel(old, quant_zero_point3):
    '''
    主要的作用是将yolo head cat之前和cat之后的结果进行补零操作
    :param old:需要通道补零的输入
    :param quant_zero_point3: 补零的零点，
    :return: 返回通道补零的结果
    '''

    z3 = quant_zero_point3.numpy()
    shape = old.shape
    out_data = old.int_repr()
    if shape[1] == 6:
        new_shape_kernel = 24
        xxx = np.ones((shape[0], new_shape_kernel, shape[2], shape[3]),
                      dtype=np.uint8) * z3
        xxx[:, :4, :, :] = out_data[:, :4, :, :]
        xxx[:, 8, :, :] = out_data[:, 4, :, :]
        xxx[:, 16, :, :] = out_data[:, 5, :, :]
        final_output = xxx
        return final_output
    elif shape[1] < 6:
        new_shape_kernel = shape[1] + 8 - shape[1] % 8
        xxx = np.ones((shape[0], new_shape_kernel, shape[2], shape[3]),
                      dtype=np.uint8) * z3
        xxx[:, :shape[1], :, :] = out_data
        final_output = xxx
        return final_output
