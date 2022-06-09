import os

from test_addleak import *


def coe2bin(coepath, binpath):
    torch.set_printoptions(profile="full")
    out_api = open(coepath)
    outapi = out_api.read().splitlines()
    data_size = int(len(outapi) * len(outapi[0]) / 2)
    p = 0
    # data_size = 5939200    #B的个数
    bin_feature = np.zeros(data_size, dtype=np.uint8, order='C')
    for index in range(len(outapi)):
        data = outapi[index].rsplit(',')
        data = data[0]
        tmp = ''
        out = []
        for index2 in range(len(data)):
            tmp += data[index2]
            if len(tmp) % 2 == 0:
                out.append(tmp)
                tmp = ''
            out.reverse()
        for index3 in range(int(len(data) / 2)):
            bin_feature[p] = int(out[index3], 16)
            p += 1
    write_path1 = binpath
    fp1 = open(write_path1, "ab+")  # 打开fp1 ab+追加写入
    bin_feature.tofile(fp1)
    fp1.close()


weight_coe_path = "../coe_file_24_001/P5_output.coe"
weight_bin_path = "../coe_file_24_001/P5_output.bin"
coe2bin(weight_coe_path, weight_bin_path)

# coe_file_path = '../bbox2_coe'
# coe_list = os.listdir(coe_file_path)
# for coe in coe_list:
#     coe2bin(os.path.join(coe_file_path, coe),)
# ins_coe_path = "../coe_file/P5_output_2bbox.coe"
# ins_bin_path = "../coe_file/P5_output_2bbox.bin"
# coe2bin(ins_coe_path, ins_bin_path)
