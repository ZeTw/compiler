import os


#
# cat_file = open('fpga1/stage110_cat6.coe')
#
# cat_lines = cat_file.readlines()
# content_list = []
# for index, line in enumerate(cat_lines):
#     line = line.strip()
#     if index % 3 == 0:
#         content = line[8:]
#     else:
#         content = line[14:]
#     content_list.append(content)
# cat_file.close()
# # print(content_list)
#
# for i in range(len(content_list)):
#     if (i + 1) % 3 == 0:
#         new_content = content_list[i] + content_list[i - 1] + content_list[i - 2]
#         with open('fpga1/new_stage110_cat6.coe', 'a') as f:
#             f.write(new_content)
#             f.write('\n')

def output_reshape(head_file_path):
    head_file_list = os.listdir(head_file_path)
    for head_file in head_file_list:
        head_file_item = open(os.path.join(head_file_path, head_file))
        head_file_lines = head_file_item.readlines()
        content_list = []
        for index, line in enumerate(head_file_lines):
            line = line.strip()
            if index % 3 == 0:
                content = line[8:]
            else:
                content = line[14:]
            content_list.append(content)
        head_file_item.close()

        for i in range(len(content_list)):
            if (i + 1) % 3 == 0:
                new_content = content_list[i] + content_list[i - 1] + content_list[i - 2]
                with open(os.path.join(head_file_path, 'new_' + head_file), 'a') as f:
                    f.write(new_content)
                    f.write('\n')
