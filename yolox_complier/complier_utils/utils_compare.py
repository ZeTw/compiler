import os
import time
import numpy as np
import torch

'''
dat2coe:
      --dat_path:dat文件路径
      --is_weight:dat文件是否为权重文件
      --no_head:是否含有头数据，这里将与后面数据格式不一致的情况定义为头数据
功能说明：
    该函数的主要功能为实现dat和coe文件格式的转换
    权重dat文件的每一行数据都带有'0x'在转换的时候是不需要的，因此切片就需要从2开始 slice_start=2
    是否有head数据主要用于区别第一行数据是否需要转化 如果不需要 索引就需要从1开始  index_start=1

'''


def dat2coe(dat_path, is_weight, no_head):
    torch.set_printoptions(profile="full")
    dat_file = open(dat_path)
    slice_start = index_start = 0
    dat_lines = dat_file.read().splitlines()
    coe_path = dat_path.replace('dat', 'coe')
    if is_weight:
        slice_start = 2
    if not no_head:
        index_start = 1

    with open(coe_path, 'w+') as f:
        for m in range(index_start, len(dat_lines), 2):
            data = dat_lines[m].rsplit(',')
            data = data[0][slice_start:]
            data2 = dat_lines[m + 1].rsplit(',')
            data2 = data2[0][slice_start:]
            f.write(data2)
            f.write(data)
            # f.write(',')
            f.write('\n')
    dat_file.close()


def coe_comparison(coe_file1_path, coe_file2_path):
    # if need_dat2coe:
    #     # 遍历dat文件夹下的所有dat文件，进行dat到coe的转换，不过此时需要保证该文件夹下的dat格式一致
    #     dat_file_list = [x for x in os.listdir(dat_dir_path) if '.dat' in x]
    #     for dat_file in dat_file_list:
    #         dat2coe(dat_file, is_weight, no_head)

    coe_file1 = open(coe_file1_path)
    coe_file1_lines = coe_file1.read().splitlines()
    coe_file2 = open(coe_file2_path)
    coe_file2_lines = coe_file2.read().splitlines()
    coe_file1_out1 = []
    coe_file1_out2 = []
    # 将该coe_file1文件下的每两个16进制值转化为对应的10进制 保存在 coe_file1_out2中
    for index1 in range(len(coe_file2_lines)):
        data1 = coe_file1_lines[index1].rsplit(',')[0]
        tmp = ''
        for data1_index, data1_item in enumerate(data1):
            tmp += data1_item
            if data1_index % 2 != 0:
                tmp = int(tmp, 16)
                coe_file1_out2.append(tmp)
                tmp = ''
        coe_file1_out1.append(data1)

    coe_file2_out1 = []
    coe_file2_out2 = []
    # 将该coe_file2文件下的每两个16进制值转化为对应的10进制 保存在 coe_file2_out2中
    for index2 in range(len(coe_file2_lines)):
        data2 = coe_file2_lines[index2].rsplit(',')[0]
        tmp = ''
        for data2_index, data2_item in enumerate(data2):
            tmp += data2_item
            if data2_index % 2 != 0:
                tmp = int(tmp, 16)
                coe_file2_out2.append(tmp)
                tmp = ''
        coe_file2_out1.append(data2)
    assert len(coe_file1_out1) == len(coe_file2_out1)
    # 将coe_file1_out2中的10进制值转换为np数组保存在final_out1中
    final_out1 = np.array(coe_file1_out2[:len(coe_file2_out2)])
    # 将coe_file2_out2中的10进制值转换为np数组保存在final_out2中
    # 按理来说进行对比的两个文件内的数据量是一致的，不需要使用第二个文件的数据量作为切分值
    final_out2 = np.array(coe_file2_out2)
    dif = final_out2 - final_out1
    result_dict = {}
    list_array = list(dif)
    set_array = set(list_array)
    for item in set_array:
        result_dict.update({item: list_array.count(item)})
    for key in result_dict.keys():
        val = result_dict[key]
        val = val / len(coe_file2_out2) * 100
        result_dict[key] = val

    print('result_dict:', result_dict)


def fpga_dat2coe(fpga_coe_file_path):
    file_list = os.listdir(fpga_coe_file_path)
    for i in file_list:
        dat_path = os.path.join(fpga_coe_file_path, i)
        coe_path = dat_path.replace('dat', 'coe')
        if not os.path.exists(coe_path):
            dat2coe(os.path.join(fpga_coe_file_path, i), 0, 1)


if __name__ == "__main__":
    # dat2coe('../weight/yolox_weight682.dat', 1, 1)

    # fpga_dat2coe('../fpga_coe_68')
    # time.sleep(1)
    #
    # fpga_coe_list = [x for x in os.listdir('../hand_coe') if x.endswith('coe')]
    #
    # for i in fpga_coe_list:
    #
    #     if 'output' in i:
    #         add_channel = i.split('.')[0] + '_addchannel' + '.coe'
    #     else:
    #         add_channel = i
    #     # coe_file_name = add_channel.split('_', 1)[1]
    #     # print(coe_file_name)
    #     file_name = os.path.join('../coe_file', add_channel)
    #     fpga_coe_name = os.path.join('../hand_coe', i)
    #     print('当前对比coe名为:', i)
    #     coe_comparison(file_name, fpga_coe_name)
    #     print('\n')
    # coe_comparison('../api_coe/P3_obj_output_addchannel.coe', '../hand_coe/P3_obj_output.coe')
    # coe_comparison('../fpga_coe/stage', '../api_coe/P5_output_final.coe')
    # coe_comparison('../fpga_coe/stage100_cat4.coe', '../coe_file/P4_output2.oce')
    # coe_comparison('../coe_file/', '../hand_coe/p4_output.coe')
    # coe_comparison('../fpga_coe/stage47_lateral_conv0.coe','../coe_file/lateral_conv0.coe')
    # coe_comparison('../fpga_coe/stage90_cat2.coe', '../api_coe/P3_output_final.coe')
    # print('==============fpga api 对比结果====================\n')
    # coe_comparison('../fpga_coe/stage100_cat4.coe', '../api_coe/P4_output_final.coe')
    # coe_comparison('../fpga_coe/stage110_cat6.coe', '../api_coe/P5_output_final.coe')
    # print('==============fpga 手写卷积 对比结果====================\n')
    # coe_comparison('../fpga_coe/stage100_cat4.coe', '../hand_coe/P4_output.coe')
    # coe_comparison('../fpga_coe/stage110_cat6.coe', '../hand_coe/P5_output.coe')
    #
    # print('==============api 手写卷积 对比结果====================\n')
    # coe_comparison('../hand_coe/P4_output.coe', '../api_coe/P4_output_final.coe')
    # coe_comparison('../hand_coe/P5_output.coe', '../api_coe/P5_output_final.coe')
    coe_comparison('../api_coe/P5_output_final.coe', '../test_coe/P5_output.coe')
    coe_comparison('../api_coe/P4_output_final.coe', '../test_coe/P4_output.coe')
    coe_comparison('../api_coe/P3_output_final.coe', '../test_coe/P3_output.coe')

