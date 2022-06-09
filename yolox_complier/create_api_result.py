import os.path

import torch
from complier_utils.utils_picture import picture_load, detect_img
from complier_utils.utils_base import focus, gen_coe, np2tensor, add_channel
from complier_utils.utils_bbox import decode_outputs


def get_api_result(quant_pth_path, img_path, show_result, para_path, layer_name_list=None, api_coe_path=''):
    '''
    :param quant_pth_path: 量化后权重的路径
    :param img_path: 需要预测的图片路径
    :param layer_name_list: 想要生成api结果的层数列表
    :param api_coe_path: api_coe 存储路径
    :param show_result: 是否展示画框结果
    '''

    model = torch.jit.load(quant_pth_path)
    model.eval()

    # =============yolox模型定义================
    #  外层 backbone 为YOLOPAFPN 内层backone为CSPDarknet

    YOLOPAFPN = model.backbone
    CSPDarknet = model.backbone.backbone
    stem = CSPDarknet.stem
    stem_conv = stem.conv
    focus_csp0 = stem.csp0

    # ==============start dark2===================
    dark2 = CSPDarknet.dark2
    dark2_list = list(dark2.children())
    dark2_BaseConv = dark2_list[0]
    dark2_CSP = dark2_list[1]
    dark2_CSP_Conv1 = dark2_CSP.conv1
    dark2_CSP_Conv2 = dark2_CSP.conv2
    dark2_CSP_Conv3 = dark2_CSP.conv3
    dark2_CSP_csp_cat1 = dark2_CSP.csp1
    dark2_CSP_m = dark2_CSP.m
    dark2_CSP_m_Bottleneck = list(dark2_CSP_m.children())[0]
    dark2_CSP_m_conv1 = dark2_CSP_m_Bottleneck.conv1
    dark2_CSP_m_conv2 = dark2_CSP_m_Bottleneck.conv2
    dark2_CSP_m_Bottleneck_add = dark2_CSP_m_Bottleneck.csp

    #  ===========start dark3===========
    dark3 = CSPDarknet.dark3
    dark3_list = list(dark3.children())
    dark3_BaseConv = dark3_list[0]
    dark3_CSP = dark3_list[1]
    dark3_CSP_Conv1 = dark3_CSP.conv1
    dark3_CSP_Conv2 = dark3_CSP.conv2
    dark3_CSP_Conv3 = dark3_CSP.conv3
    dark3_CSP_csp_cat1 = dark3_CSP.csp1
    dark3_CSP_m = dark3_CSP.m
    dark3_CSP_m_Bottleneck_list = list(dark3_CSP_m.children())
    dark3_CSP_m_Bottleneck_item0 = dark3_CSP_m_Bottleneck_list[0]
    dark3_Bottleneck_item0_conv1 = dark3_CSP_m_Bottleneck_item0.conv1
    dark3_Bottleneck_item0_conv2 = dark3_CSP_m_Bottleneck_item0.conv2
    dark3_Bottleneck_item0_add = dark3_CSP_m_Bottleneck_item0.csp

    dark3_CSP_m_Bottleneck_item1 = dark3_CSP_m_Bottleneck_list[1]
    dark3_Bottleneck_item1_conv1 = dark3_CSP_m_Bottleneck_item1.conv1
    dark3_Bottleneck_item1_conv2 = dark3_CSP_m_Bottleneck_item1.conv2
    dark3_Bottleneck_item1_add = dark3_CSP_m_Bottleneck_item1.csp

    dark3_CSP_m_Bottleneck_item2 = dark3_CSP_m_Bottleneck_list[2]
    dark3_Bottleneck_item2_conv1 = dark3_CSP_m_Bottleneck_item2.conv1
    dark3_Bottleneck_item2_conv2 = dark3_CSP_m_Bottleneck_item2.conv2
    dark3_Bottleneck_item2_add = dark3_CSP_m_Bottleneck_item2.csp

    #  ===========start dark4===========
    dark4 = CSPDarknet.dark4
    dark4_list = list(dark4.children())
    dark4_BaseConv = dark4_list[0]
    dark4_CSP = dark4_list[1]
    dark4_CSP_Conv1 = dark4_CSP.conv1
    dark4_CSP_Conv2 = dark4_CSP.conv2
    dark4_CSP_Conv3 = dark4_CSP.conv3
    dark4_CSP_csp_cat1 = dark4_CSP.csp1
    dark4_CSP_m = dark4_CSP.m
    dark4_CSP_m_Bottleneck_list = list(dark4_CSP_m.children())
    dark4_CSP_m_Bottleneck_item0 = dark4_CSP_m_Bottleneck_list[0]
    dark4_Bottleneck_item0_conv1 = dark4_CSP_m_Bottleneck_item0.conv1
    dark4_Bottleneck_item0_conv2 = dark4_CSP_m_Bottleneck_item0.conv2
    dark4_Bottleneck_item0_add = dark4_CSP_m_Bottleneck_item0.csp

    dark4_CSP_m_Bottleneck_item1 = dark4_CSP_m_Bottleneck_list[1]
    dark4_Bottleneck_item1_conv1 = dark4_CSP_m_Bottleneck_item1.conv1
    dark4_Bottleneck_item1_conv2 = dark4_CSP_m_Bottleneck_item1.conv2
    dark4_Bottleneck_item1_add = dark4_CSP_m_Bottleneck_item1.csp

    dark4_CSP_m_Bottleneck_item2 = dark4_CSP_m_Bottleneck_list[2]
    dark4_Bottleneck_item2_conv1 = dark4_CSP_m_Bottleneck_item2.conv1
    dark4_Bottleneck_item2_conv2 = dark4_CSP_m_Bottleneck_item2.conv2
    dark4_Bottleneck_item2_add = dark4_CSP_m_Bottleneck_item2.csp

    # ===========start dark5===========
    dark5 = CSPDarknet.dark5
    dark5_list = list(dark5.children())
    dark5_BaseConv = dark5_list[0]
    dark5_CSP = dark5_list[1]
    dark5_CSP_Conv1 = dark5_CSP.conv1
    dark5_CSP_Conv2 = dark5_CSP.conv2
    dark5_CSP_Conv3 = dark5_CSP.conv3
    dark5_CSP_csp_cat1 = dark5_CSP.csp1
    dark5_CSP_m = dark5_CSP.m
    dark5_CSP_m_Bottleneck_list = list(dark5_CSP_m.children())
    dark5_CSP_m_Bottleneck_item0 = dark5_CSP_m_Bottleneck_list[0]
    dark5_Bottleneck_item0_conv1 = dark5_CSP_m_Bottleneck_item0.conv1
    dark5_Bottleneck_item0_conv2 = dark5_CSP_m_Bottleneck_item0.conv2
    dark5_Bottleneck_item0_add = dark5_CSP_m_Bottleneck_item0.csp

    # ===========start upsample===========

    upsample = YOLOPAFPN.upsample
    lateral_conv0 = YOLOPAFPN.lateral_conv0

    # ===========start C3_P4===========
    C3_P4 = YOLOPAFPN.C3_p4
    C3_P4_CSP_Conv1 = C3_P4.conv1
    C3_P4_CSP_Conv2 = C3_P4.conv2
    C3_P4_CSP_Conv3 = C3_P4.conv3
    C3_P4_CSP_cat = C3_P4.csp1
    C3_P4_m = C3_P4.m
    C3_P4_m_Bottleneck_list = list(C3_P4_m.children())
    C3_P4_m_Bottleneck_item0 = C3_P4_m_Bottleneck_list[0]
    C3_P4_m_Bottleneck_item0_conv1 = C3_P4_m_Bottleneck_item0.conv1
    C3_P4_m_Bottleneck_item0_conv2 = C3_P4_m_Bottleneck_item0.conv2
    C3_P4_m_Bottleneck_item0_add = C3_P4_m_Bottleneck_item0.csp

    # ===========start reduce_conv1===========
    reduce_conv1 = YOLOPAFPN.reduce_conv1

    # ===========start C3_P3===========
    C3_P3 = YOLOPAFPN.C3_p3
    C3_P3_CSP_Conv1 = C3_P3.conv1
    C3_P3_CSP_Conv2 = C3_P3.conv2
    C3_P3_CSP_Conv3 = C3_P3.conv3
    C3_P3_CSP_cat = C3_P3.csp1
    C3_P3_m = C3_P3.m
    C3_P3_m_Bottleneck_list = list(C3_P3_m.children())
    C3_P3_m_Bottleneck_item0 = C3_P3_m_Bottleneck_list[0]
    C3_P3_m_Bottleneck_item0_conv1 = C3_P3_m_Bottleneck_item0.conv1
    C3_P3_m_Bottleneck_item0_conv2 = C3_P3_m_Bottleneck_item0.conv2
    C3_P3_m_Bottleneck_item0_add = C3_P3_m_Bottleneck_item0.csp

    # ===========start bu_conv2===========
    bu_conv2 = YOLOPAFPN.bu_conv2

    # ===========start C3_n3===========
    C3_n3 = YOLOPAFPN.C3_n3
    C3_n3_CSP_Conv1 = C3_n3.conv1
    C3_n3_CSP_Conv2 = C3_n3.conv2
    C3_n3_CSP_Conv3 = C3_n3.conv3
    C3_n3_CSP_cat = C3_n3.csp1
    C3_n3_m = C3_n3.m
    C3_n3_m_Bottleneck_list = list(C3_n3_m.children())
    C3_n3_m_Bottleneck_item0 = C3_n3_m_Bottleneck_list[0]
    C3_n3_m_Bottleneck_item0_conv1 = C3_n3_m_Bottleneck_item0.conv1
    C3_n3_m_Bottleneck_item0_conv2 = C3_n3_m_Bottleneck_item0.conv2
    C3_n3_m_Bottleneck_item0_add = C3_n3_m_Bottleneck_item0.csp

    # ===========start bu_conv1===========
    bu_conv1 = YOLOPAFPN.bu_conv1

    # ===========start C3_n4===========
    C3_n4 = YOLOPAFPN.C3_n4
    C3_n4_CSP_Conv1 = C3_n4.conv1
    C3_n4_CSP_Conv2 = C3_n4.conv2
    C3_n4_CSP_Conv3 = C3_n4.conv3
    C3_n4_CSP_cat = C3_n4.csp1
    C3_n4_m = C3_n4.m
    C3_n4_m_Bottleneck_list = list(C3_n4_m.children())
    C3_n4_m_Bottleneck_item0 = C3_n4_m_Bottleneck_list[0]
    C3_n4_m_Bottleneck_item0_conv1 = C3_n4_m_Bottleneck_item0.conv1
    C3_n4_m_Bottleneck_item0_conv2 = C3_n4_m_Bottleneck_item0.conv2
    C3_n4_m_Bottleneck_item0_add = C3_n4_m_Bottleneck_item0.csp

    # ===========start p4_w1_relu===========
    # p4_w1_relu = YOLOPAFPN.p4_w1_relu

    YOLOPAFPN_csp2_cat = YOLOPAFPN.csp2
    YOLOPAFPN_csp3_cat = YOLOPAFPN.csp3
    YOLOPAFPN_csp4_cat = YOLOPAFPN.csp4
    YOLOPAFPN_csp5_cat = YOLOPAFPN.csp5

    # ==========start head ===========

    head = model.head

    cls_convs = head.cls_convs
    cls_convs_item_list = list(cls_convs.children())

    cls_convs_item0 = cls_convs_item_list[0]
    cls_convs_item0_BaseConv_list = list(cls_convs_item0.children())
    cls_convs_item0_BaseConv0 = cls_convs_item0_BaseConv_list[0]
    cls_convs_item0_BaseConv1 = cls_convs_item0_BaseConv_list[1]

    cls_convs_item1 = cls_convs_item_list[1]
    cls_convs_item1_BaseConv_list = list(cls_convs_item1.children())
    cls_convs_item1_BaseConv0 = cls_convs_item1_BaseConv_list[0]
    cls_convs_item1_BaseConv1 = cls_convs_item1_BaseConv_list[1]

    cls_convs_item2 = cls_convs_item_list[2]
    cls_convs_item2_BaseConv_list = list(cls_convs_item2.children())
    cls_convs_item2_BaseConv0 = cls_convs_item2_BaseConv_list[0]
    cls_convs_item2_BaseConv1 = cls_convs_item2_BaseConv_list[1]

    # ===============start reg_convs================
    reg_convs = head.reg_convs

    reg_convs_item_list = list(reg_convs.children())
    reg_convs_item0 = reg_convs_item_list[0]
    reg_convs_item0_BaseConv_list = list(reg_convs_item0.children())

    reg_convs_item0_BaseConv0 = reg_convs_item0_BaseConv_list[0]
    reg_convs_item0_BaseConv1 = reg_convs_item0_BaseConv_list[1]

    reg_convs_item1 = reg_convs_item_list[1]
    reg_convs_item1_BaseConv_list = list(reg_convs_item1.children())
    reg_convs_item1_BaseConv0 = reg_convs_item1_BaseConv_list[0]
    reg_convs_item1_BaseConv1 = reg_convs_item1_BaseConv_list[1]

    reg_convs_item2 = reg_convs_item_list[2]
    reg_convs_item2_BaseConv_list = list(reg_convs_item2.children())
    reg_convs_item2_BaseConv0 = reg_convs_item2_BaseConv_list[0]
    reg_convs_item2_BaseConv1 = reg_convs_item2_BaseConv_list[1]

    # =================start cls_preds================
    cls_preds = head.cls_preds
    cls_preds_item_list = list(cls_preds.children())
    cls_preds_Conv2d0 = cls_preds_item_list[0]
    cls_preds_Conv2d1 = cls_preds_item_list[1]
    cls_preds_Conv2d2 = cls_preds_item_list[2]

    # =================start reg_preds================
    reg_preds = head.reg_preds
    reg_preds_item_list = list(reg_preds.children())
    reg_preds_Conv2d0 = reg_preds_item_list[0]
    reg_preds_Conv2d1 = reg_preds_item_list[1]
    reg_preds_Conv2d2 = reg_preds_item_list[2]

    # =================start obj_preds================
    obj_preds = head.obj_preds
    obj_preds_item_list = list(obj_preds.children())
    obj_preds_Conv2d0 = obj_preds_item_list[0]
    obj_preds_Conv2d1 = obj_preds_item_list[1]
    obj_preds_Conv2d2 = obj_preds_item_list[2]

    # =================start stems================
    stems = head.stems
    stems_item_list = list(stems.children())
    stems_BaseConv0 = stems_item_list[0]
    stems_BaseConv1 = stems_item_list[1]
    stems_BaseConv2 = stems_item_list[2]

    head_cat6 = head.csp6
    # head_cat7 = head.csp7
    quant = model.quant
    dequant = model.dequant
    result_list = []
    # 模型定义完成，开始进行推断
    torch.set_printoptions(profile="full")
    with torch.no_grad():
        img = picture_load(img_path)

        quant_feature = quant(img)

        # =================start CSPDarknet=============
        patch_top_left, patch_bot_left, patch_top_right, patch_bot_right = focus(quant_feature)

        x = focus_csp0.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        focus_cat_feature = x
        focus_conv_feature = stem_conv(x)
        result_list.append(focus_cat_feature.int_repr())

        '''
        dark 所参与的运算都是类似的，
        Baseconv(x)
        x1 =conv1(x)
        x2 = conv2(x)
        x_m = m(x1)
        x_cat = cat(x_m,x2)
        conv3(x_cat)
    
        m中的运算也是类似的
        m_conv1(x1)
        y=m_conv2(m_conv1(x1))
        csp.add(x1,y)
        '''
        # ==================start dark2=================
        # stem输出作为dark2 的输入
        dark2_BaseConv_feature = dark2_BaseConv(focus_conv_feature)
        result_list.append(dark2_BaseConv_feature.int_repr())

        dark2_x1 = dark2_CSP_Conv1(dark2_BaseConv_feature)
        result_list.append(dark2_x1.int_repr())

        dark2_x2 = dark2_CSP_Conv2(dark2_BaseConv_feature)
        result_list.append(dark2_x2.int_repr())

        # x1输入到m中
        dark2_CSP_m_conv1_feature = dark2_CSP_m_conv1(dark2_x1)
        result_list.append(dark2_CSP_m_conv1_feature.int_repr())

        dark2_CSP_m_conv2_feature = dark2_CSP_m_conv2(dark2_CSP_m_conv1_feature)
        result_list.append(dark2_CSP_m_conv2_feature.int_repr())

        dark2_CSP_m_feature = dark2_CSP_m_Bottleneck_add.add(dark2_x1, dark2_CSP_m_conv2_feature)
        result_list.append(dark2_CSP_m_feature.int_repr())

        # m结束，x2和m的输出进行cat

        dark2_CSP_cat_feature = dark2_CSP_csp_cat1.cat((dark2_CSP_m_feature, dark2_x2), dim=1)
        result_list.append(dark2_CSP_cat_feature.int_repr())

        dark2_CSP_Conv3_feature = dark2_CSP_Conv3(dark2_CSP_cat_feature)
        result_list.append(dark2_CSP_Conv3_feature.int_repr())

        # outputs['dark2']

        # ==================start dark3=================
        dark3_BaseConv_feature = dark3_BaseConv(dark2_CSP_Conv3_feature)
        result_list.append(dark3_BaseConv_feature.int_repr())

        dark3_x1 = dark3_CSP_Conv1(dark3_BaseConv_feature)
        result_list.append(dark3_x1.int_repr())

        dark3_x2 = dark3_CSP_Conv2(dark3_BaseConv_feature)
        result_list.append(dark3_x2.int_repr())

        # ============start dark3_m=============
        # m0
        dark3_Bottleneck_item0_conv1_feature = dark3_Bottleneck_item0_conv1(dark3_x1)
        result_list.append(dark3_Bottleneck_item0_conv1_feature.int_repr())

        dark3_Bottleneck_item0_conv2_feature = dark3_Bottleneck_item0_conv2(dark3_Bottleneck_item0_conv1_feature)
        result_list.append(dark3_Bottleneck_item0_conv2_feature.int_repr())

        dark3_Bottleneck_item0_feature = dark3_Bottleneck_item0_add.add(dark3_x1, dark3_Bottleneck_item0_conv2_feature)
        result_list.append(dark3_Bottleneck_item0_feature.int_repr())

        # m1
        dark3_Bottleneck_item1_conv1_feature = dark3_Bottleneck_item1_conv1(dark3_Bottleneck_item0_feature)
        result_list.append(dark3_Bottleneck_item1_conv1_feature.int_repr())

        dark3_Bottleneck_item1_conv2_feature = dark3_Bottleneck_item1_conv2(dark3_Bottleneck_item1_conv1_feature)
        result_list.append(dark3_Bottleneck_item1_conv2_feature.int_repr())

        dark3_Bottleneck_item1_feature = dark3_Bottleneck_item1_add.add(
            dark3_Bottleneck_item0_feature, dark3_Bottleneck_item1_conv2_feature)

        result_list.append(dark3_Bottleneck_item1_feature.int_repr())

        # m2
        dark3_Bottleneck_item2_conv1_feature = dark3_Bottleneck_item2_conv1(dark3_Bottleneck_item1_feature)
        result_list.append(dark3_Bottleneck_item2_conv1_feature.int_repr())
        dark3_Bottleneck_item2_conv2_feature = dark3_Bottleneck_item2_conv2(dark3_Bottleneck_item2_conv1_feature)
        result_list.append(dark3_Bottleneck_item2_conv2_feature.int_repr())

        dark3_Bottleneck_item2_feature = dark3_Bottleneck_item2_add.add(
            dark3_Bottleneck_item1_feature, dark3_Bottleneck_item2_conv2_feature)
        result_list.append(dark3_Bottleneck_item2_feature.int_repr())

        # dark3 csp
        dark3_CSP_cat_feature = dark3_CSP_csp_cat1.cat((dark3_Bottleneck_item2_feature, dark3_x2), dim=1)
        result_list.append(dark3_CSP_cat_feature.int_repr())

        dark3_CSP_Conv3_feature = dark3_CSP_Conv3(dark3_CSP_cat_feature)
        result_list.append(dark3_CSP_Conv3_feature.int_repr())

        # outputs['dark3']

        # ==================start dark4=================
        dark4_BaseConv_feature = dark4_BaseConv(dark3_CSP_Conv3_feature)
        result_list.append(dark4_BaseConv_feature.int_repr())

        dark4_x1 = dark4_CSP_Conv1(dark4_BaseConv_feature)
        result_list.append(dark4_x1.int_repr())

        dark4_x2 = dark4_CSP_Conv2(dark4_BaseConv_feature)
        result_list.append(dark4_x2.int_repr())
        # ============start dark4_m=============
        # m0
        dark4_Bottleneck_item0_conv1_feature = dark4_Bottleneck_item0_conv1(dark4_x1)
        result_list.append(dark4_Bottleneck_item0_conv1_feature.int_repr())

        dark4_Bottleneck_item0_conv2_feature = dark4_Bottleneck_item0_conv2(dark4_Bottleneck_item0_conv1_feature)
        result_list.append(dark4_Bottleneck_item0_conv2_feature.int_repr())

        dark4_Bottleneck_item0_feature = dark4_Bottleneck_item0_add.add(dark4_x1, dark4_Bottleneck_item0_conv2_feature)
        result_list.append(dark4_Bottleneck_item0_feature.int_repr())

        # m1
        dark4_Bottleneck_item1_conv1_feature = dark4_Bottleneck_item1_conv1(dark4_Bottleneck_item0_feature)
        result_list.append(dark4_Bottleneck_item1_conv1_feature.int_repr())

        dark4_Bottleneck_item1_conv2_feature = dark4_Bottleneck_item1_conv2(dark4_Bottleneck_item1_conv1_feature)
        result_list.append(dark4_Bottleneck_item1_conv2_feature.int_repr())

        dark4_Bottleneck_item1_feature = dark4_Bottleneck_item1_add.add(dark4_Bottleneck_item0_feature,
                                                                        dark4_Bottleneck_item1_conv2_feature)
        result_list.append(dark4_Bottleneck_item1_feature.int_repr())

        # m2
        dark4_Bottleneck_item2_conv1_feature = dark4_Bottleneck_item2_conv1(dark4_Bottleneck_item1_feature)
        result_list.append(dark4_Bottleneck_item2_conv1_feature.int_repr())

        dark4_Bottleneck_item2_conv2_feature = dark4_Bottleneck_item2_conv2(dark4_Bottleneck_item2_conv1_feature)
        result_list.append(dark4_Bottleneck_item2_conv2_feature.int_repr())

        dark4_Bottleneck_item2_feature = dark4_Bottleneck_item2_add.add(dark4_Bottleneck_item1_feature,
                                                                        dark4_Bottleneck_item2_conv2_feature)

        result_list.append(dark4_Bottleneck_item2_feature.int_repr())
        # dark4 csp
        dark4_CSP_cat_feature = dark4_CSP_csp_cat1.cat((dark4_Bottleneck_item2_feature, dark4_x2), dim=1)
        result_list.append(dark4_CSP_cat_feature.int_repr())

        dark4_CSP_Conv3_feature = dark4_CSP_Conv3(dark4_CSP_cat_feature)
        result_list.append(dark4_CSP_Conv3_feature.int_repr())

        # outpus['dark4']

        # ==================start dark5=================
        dark5_BaseConv_feature = dark5_BaseConv(dark4_CSP_Conv3_feature)
        result_list.append(dark5_BaseConv_feature.int_repr())

        dark5_x1 = dark5_CSP_Conv1(dark5_BaseConv_feature)
        result_list.append(dark5_x1.int_repr())

        dark5_x2 = dark5_CSP_Conv2(dark5_BaseConv_feature)
        result_list.append(dark5_x2.int_repr())

        # ============start dark5_m=============
        # m0
        dark5_Bottleneck_item0_conv1_feature = dark5_Bottleneck_item0_conv1(dark5_x1)
        result_list.append(dark5_Bottleneck_item0_conv1_feature.int_repr())

        dark5_Bottleneck_item0_conv2_feature = dark5_Bottleneck_item0_conv2(dark5_Bottleneck_item0_conv1_feature)
        result_list.append(dark5_Bottleneck_item0_conv2_feature.int_repr())

        # dark5 csp
        dark5_CSP_cat_feature = dark5_CSP_csp_cat1.cat((dark5_Bottleneck_item0_conv2_feature, dark5_x2), dim=1)
        result_list.append(dark5_CSP_cat_feature.int_repr())

        dark5_CSP_Conv3_feature = dark5_CSP_Conv3(dark5_CSP_cat_feature)
        result_list.append(dark5_CSP_Conv3_feature.int_repr())

        # ===============CSPDarknet 结束，开始YOLOPAFPN的其他部分===============
        feat1, feat2, feat3 = dark3_CSP_Conv3_feature, dark4_CSP_Conv3_feature, dark5_CSP_Conv3_feature
        P5 = lateral_conv0(feat3)
        result_list.append(P5.int_repr())

        P5_upsample = upsample(P5)
        result_list.append(P5_upsample.int_repr())

        P5_upsample_cat = YOLOPAFPN_csp2_cat.cat([P5_upsample, feat2], 1)
        result_list.append(P5_upsample_cat.int_repr())

        P5_upsample_cat_feature = P5_upsample_cat

        # ==========start C3_p4================
        C3_P4_x1 = C3_P4_CSP_Conv1(P5_upsample_cat_feature)
        result_list.append(C3_P4_x1.int_repr())

        C3_P4_x2 = C3_P4_CSP_Conv2(P5_upsample_cat_feature)
        result_list.append(C3_P4_x2.int_repr())

        # ==========start C3_p4_m =============
        C3_P4_Bottleneck_item0_conv1_feature = C3_P4_m_Bottleneck_item0_conv1(C3_P4_x1)
        result_list.append(C3_P4_Bottleneck_item0_conv1_feature.int_repr())

        C3_P4_Bottleneck_item0_conv2_feature = C3_P4_m_Bottleneck_item0_conv2(C3_P4_Bottleneck_item0_conv1_feature)
        result_list.append(C3_P4_Bottleneck_item0_conv2_feature.int_repr())

        C3_P4_CSP_cat_feature = C3_P4_CSP_cat.cat((C3_P4_Bottleneck_item0_conv2_feature, C3_P4_x2), dim=1)
        result_list.append(C3_P4_CSP_cat_feature.int_repr())

        C3_P4_CSP_Conv3_feature = C3_P4_CSP_Conv3(C3_P4_CSP_cat_feature)
        result_list.append(C3_P4_CSP_Conv3_feature.int_repr())

        P4 = reduce_conv1(C3_P4_CSP_Conv3_feature)
        result_list.append(P4.int_repr())

        P4_upsample = upsample(P4)
        result_list.append(P4_upsample.int_repr())

        P4_upsample_cat_feature = YOLOPAFPN_csp3_cat.cat([P4_upsample, feat1], 1)
        result_list.append(P4_upsample_cat_feature.int_repr())

        # ===============start C3_p3===============
        C3_P3_x1 = C3_P3_CSP_Conv1(P4_upsample_cat_feature)
        result_list.append(C3_P3_x1.int_repr())

        C3_P3_x2 = C3_P3_CSP_Conv2(P4_upsample_cat_feature)
        result_list.append(C3_P3_x2.int_repr())

        # ==========start C3_p3_m =============
        C3_P3_Bottleneck_item0_conv1_feature = C3_P3_m_Bottleneck_item0_conv1(C3_P3_x1)
        result_list.append(C3_P3_Bottleneck_item0_conv1_feature.int_repr())

        C3_P3_Bottleneck_item0_conv2_feature = C3_P3_m_Bottleneck_item0_conv2(C3_P3_Bottleneck_item0_conv1_feature)
        result_list.append(C3_P3_Bottleneck_item0_conv2_feature.int_repr())

        C3_P3_CSP_cat_feature = C3_P3_CSP_cat.cat((C3_P3_Bottleneck_item0_conv2_feature, C3_P3_x2), dim=1)
        result_list.append(C3_P3_CSP_cat_feature.int_repr())

        C3_P3_CSP_Conv3_feature = C3_P3_CSP_Conv3(C3_P3_CSP_cat_feature)
        result_list.append(C3_P3_CSP_Conv3_feature.int_repr())

        P3_downsample = bu_conv2(C3_P3_CSP_Conv3_feature)
        result_list.append(P3_downsample.int_repr())

        P3_downsample_cat = YOLOPAFPN_csp4_cat.cat([P3_downsample, P4], dim=1)
        result_list.append(P3_downsample_cat.int_repr())

        # ===============start C3_n3==============
        C3_n3_x1 = C3_n3_CSP_Conv1(P3_downsample_cat)
        result_list.append(C3_n3_x1.int_repr())

        C3_n3_x2 = C3_n3_CSP_Conv2(P3_downsample_cat)
        result_list.append(C3_n3_x2.int_repr())  # 66

        # ==========start C3_n3_m =============
        C3_n3_Bottleneck_item0_conv1_feature = C3_n3_m_Bottleneck_item0_conv1(C3_n3_x1)
        result_list.append(C3_n3_Bottleneck_item0_conv1_feature.int_repr())  # 67

        C3_n3_Bottleneck_item0_conv2_feature = C3_n3_m_Bottleneck_item0_conv2(C3_n3_Bottleneck_item0_conv1_feature)
        result_list.append(C3_n3_Bottleneck_item0_conv2_feature.int_repr())  # 68

        C3_n3_CSP_cat_feature = C3_n3_CSP_cat.cat((C3_n3_Bottleneck_item0_conv2_feature, C3_n3_x2), dim=1)
        result_list.append(C3_n3_CSP_cat_feature.int_repr())  # 69

        C3_n3_CSP_Conv3_feature = C3_n3_CSP_Conv3(C3_n3_CSP_cat_feature)
        result_list.append(C3_n3_CSP_Conv3_feature.int_repr())  # 70

        P4_downsample = bu_conv1(C3_n3_CSP_Conv3_feature)
        result_list.append(P4_downsample.int_repr())  # 71

        P4_downsample_cat = YOLOPAFPN_csp5_cat.cat([P4_downsample, P5], 1)
        result_list.append(P4_downsample_cat.int_repr())  # 72

        # ===============start C3_n4==============
        C3_n4_x1 = C3_n4_CSP_Conv1(P4_downsample_cat)
        result_list.append(C3_n4_x1.int_repr())  # 73

        C3_n4_x2 = C3_n4_CSP_Conv2(P4_downsample_cat)
        result_list.append(C3_n4_x2.int_repr())  # 74
        # ==========start C3_n4_m =============
        C3_n4_Bottleneck_item0_conv1_feature = C3_n4_m_Bottleneck_item0_conv1(C3_n4_x1)
        result_list.append(C3_n4_Bottleneck_item0_conv1_feature.int_repr())  # 75
        C3_n4_Bottleneck_item0_conv2_feature = C3_n4_m_Bottleneck_item0_conv2(C3_n4_Bottleneck_item0_conv1_feature)
        result_list.append(C3_n4_Bottleneck_item0_conv2_feature.int_repr())  # 76

        C3_n4_CSP_cat_feature = C3_n4_CSP_cat.cat((C3_n4_Bottleneck_item0_conv2_feature, C3_n4_x2), dim=1)
        result_list.append(C3_n4_CSP_cat_feature.int_repr())  # 77
        C3_n4_CSP_Conv3_feature = C3_n4_CSP_Conv3(C3_n4_CSP_cat_feature)
        result_list.append(C3_n4_CSP_Conv3_feature.int_repr())  # 78

        #  三个yolo head 分别对应 P3_out, P4_out, P5_out
        P3_out = C3_P3_CSP_Conv3_feature
        P4_out = C3_n3_CSP_Conv3_feature
        P5_out = C3_n4_CSP_Conv3_feature

        # =========start yolo head ============
        # print('P3_out', P3_out.shape)
        P3_stem = stems_BaseConv0(P3_out)
        result_list.append(P3_stem.int_repr())  # 79

        cls_convs_item0_BaseConv0_feature = cls_convs_item0_BaseConv0(P3_stem)
        result_list.append(cls_convs_item0_BaseConv0_feature.int_repr())  # 80

        cls_convs_item0_BaseConv1_feature = cls_convs_item0_BaseConv1(cls_convs_item0_BaseConv0_feature)
        P3_cls_feat = cls_convs_item0_BaseConv1_feature
        result_list.append(P3_cls_feat.int_repr())  # 81

        P3_cls_output = cls_preds_Conv2d0(P3_cls_feat)
        p3_cls_preds_act_zp = np2tensor(os.path.join(para_path, 'head.cls_preds.0.zero_point.npy'))
        P3_cls_output_addchannel = add_channel(P3_cls_output, p3_cls_preds_act_zp)
        result_list.append(P3_cls_output_addchannel)  # 82

        reg_convs_item0_BaseConv0_feature = reg_convs_item0_BaseConv0(P3_stem)
        result_list.append(reg_convs_item0_BaseConv0_feature.int_repr())  # 83

        reg_convs_item0_BaseConv1_feature = reg_convs_item0_BaseConv1(reg_convs_item0_BaseConv0_feature)

        P3_reg_feat = reg_convs_item0_BaseConv1_feature
        result_list.append(P3_reg_feat.int_repr())  # 84

        head_cat_zp = np2tensor(os.path.join(para_path, 'head.csp6.zero_point.npy'))
        P3_reg_output = reg_preds_Conv2d0(P3_reg_feat)
        p3_reg_preds_act_zp = np2tensor(os.path.join(para_path, 'head.reg_preds.0.zero_point.npy'))
        P3_reg_output_addchannel = add_channel(P3_reg_output, p3_reg_preds_act_zp)
        result_list.append(P3_reg_output_addchannel)  # 85

        P3_obj_output = obj_preds_Conv2d0(P3_reg_feat)
        p3_obj_preds_act_zp = np2tensor(os.path.join(para_path, 'head.obj_preds.0.zero_point.npy'))
        P3_obj_output_addchannel = add_channel(P3_obj_output, p3_obj_preds_act_zp)
        result_list.append(P3_obj_output_addchannel)  # 86

        P3_output = head_cat6.cat([P3_reg_output, P3_obj_output, P3_cls_output], 1)
        # gen_coe('bbox2/P3_output.coe', P3_output.int_repr())
        P3_output_final = add_channel(P3_output, head_cat_zp)
        # gen_coe('api_coe/P3_output_final.coe', P3_output_final)
        result_list.append(P3_output_final)  # 87

        P4_stem = stems_BaseConv1(P4_out)
        result_list.append(P4_stem.int_repr())  # 88

        cls_convs_item1_BaseConv0_feature = cls_convs_item1_BaseConv0(P4_stem)
        result_list.append(cls_convs_item1_BaseConv0_feature.int_repr())  # 89

        cls_convs_item1_BaseConv1_feature = cls_convs_item1_BaseConv1(cls_convs_item1_BaseConv0_feature)

        P4_cls_feat = cls_convs_item1_BaseConv1_feature
        result_list.append(P4_cls_feat.int_repr())  # 90

        P4_cls_output = cls_preds_Conv2d1(P4_cls_feat)
        p4_cls_preds_act_zp = np2tensor(os.path.join(para_path, 'head.cls_preds.1.zero_point.npy'))
        P4_cls_output_addchannel = add_channel(P4_cls_output, p4_cls_preds_act_zp)
        result_list.append(P4_cls_output_addchannel)  # 91

        reg_convs_item1_BaseConv0_feature = reg_convs_item1_BaseConv0(P4_stem)
        result_list.append(reg_convs_item1_BaseConv0_feature.int_repr())  # 92
        reg_convs_item1_BaseConv1_feature = reg_convs_item1_BaseConv1(reg_convs_item1_BaseConv0_feature)
        P4_reg_feat = reg_convs_item1_BaseConv1_feature
        result_list.append(P4_reg_feat.int_repr())  # 93

        P4_reg_output = reg_preds_Conv2d1(P4_reg_feat)
        p4_reg_preds_act_zp = np2tensor(os.path.join(para_path, 'head.reg_preds.1.zero_point.npy'))
        P4_reg_output_addchannel = add_channel(P4_reg_output, p4_reg_preds_act_zp)
        result_list.append(P4_reg_output_addchannel)  # 94

        P4_obj_output = obj_preds_Conv2d1(P4_reg_feat)
        p4_obj_preds_act_zp = np2tensor(os.path.join(para_path, 'head.obj_preds.1.zero_point.npy'))
        P4_obj_output_addchannel = add_channel(P4_obj_output, p4_obj_preds_act_zp)
        result_list.append(P4_obj_output_addchannel)  # 95

        P4_output = head_cat6.cat([P4_reg_output, P4_obj_output, P4_cls_output], 1)
        # gen_coe('bbox2/P4_output.coe', P4_output.int_repr())

        P4_output_final = add_channel(P4_output, head_cat_zp)
        # gen_coe('api_coe/P4_output_final.coe', P4_output_final)
        result_list.append(P4_output_final)  # 96

        P5_stem = stems_BaseConv2(P5_out)
        result_list.append(P5_stem.int_repr())  # 97

        cls_convs_item2_BaseConv0_feature = cls_convs_item2_BaseConv0(P5_stem)
        result_list.append(cls_convs_item2_BaseConv0_feature.int_repr())  # 98
        cls_convs_item2_BaseConv1_feature = cls_convs_item2_BaseConv1(cls_convs_item2_BaseConv0_feature)

        P5_cls_feat = cls_convs_item2_BaseConv1_feature
        result_list.append(P5_cls_feat.int_repr())  # 99

        P5_cls_output = cls_preds_Conv2d2(P5_cls_feat)

        p5_cls_preds_act_zp = np2tensor(os.path.join(para_path, 'head.cls_preds.2.zero_point.npy'))
        P5_cls_output_addchannel = add_channel(P5_cls_output, p5_cls_preds_act_zp)
        result_list.append(P5_cls_output_addchannel)  # 100

        reg_convs_item2_BaseConv0_feature = reg_convs_item2_BaseConv0(P5_stem)
        result_list.append(reg_convs_item2_BaseConv0_feature.int_repr())  # 101

        reg_convs_item2_BaseConv1_feature = reg_convs_item2_BaseConv1(reg_convs_item2_BaseConv0_feature)
        P5_reg_feat = reg_convs_item2_BaseConv1_feature
        result_list.append(P5_reg_feat.int_repr())  # 102

        P5_reg_output = reg_preds_Conv2d2(P5_reg_feat)
        p5_reg_preds_act_zp = np2tensor(os.path.join(para_path, 'head.reg_preds.2.zero_point.npy'))
        P5_reg_output_addchannel = add_channel(P5_reg_output, p5_reg_preds_act_zp)
        result_list.append(P5_reg_output_addchannel)  # 103

        P5_obj_output = obj_preds_Conv2d2(P5_reg_feat)
        p5_obj_preds_act_zp = np2tensor(os.path.join(para_path, 'head.obj_preds.2.zero_point.npy'))
        P5_obj_output_addchannel = add_channel(P5_obj_output, p5_obj_preds_act_zp)
        result_list.append(P5_obj_output_addchannel)  # 104

        P5_output = head_cat6.cat([P5_reg_output, P5_obj_output, P5_cls_output], 1)
        # gen_coe('bbox2/P5_output.coe', P5_output.int_repr())

        P5_output_final = add_channel(P5_output, head_cat_zp)
        # gen_coe('api_coe/P5_output_final.coe', P5_output_final)
        result_list.append(P5_output_final)  # 105

        if show_result:
            out_list = []
            head_outputs = [P3_output, P4_output, P5_output]

            for i in head_outputs:
                out_list.append(dequant(i))
            final_output = decode_outputs(out_list, [640, 640])

            # out = model(img)
            # out = decode_outputs(out, [640, 640])

            img_result = detect_img(img_path, outputs=final_output)
            img_result.show()

        if api_coe_path:
            if layer_name_list:

                file_path = 'model_structure/layer_name.txt'
                name_file = open(file_path)
                name_lines = name_file.readlines()
                name_file.close()
                name_dict = {}

                for index, name_line in enumerate(name_lines):
                    name_dict[index + 1] = name_line.strip()
                #
                for layer_index in layer_name_list:
                    gen_coe(os.path.join(api_coe_path, name_dict[layer_index] + '.coe'), result_list[layer_index - 1])

        # print('===============网络推断已完成，开始生成中间层结果===================')
        #
        # file_path = 'model_structure/layer_name.txt'
        # name_file = open(file_path)
        # name_lines = name_file.readlines()
        # name_dict = {}
        #
        # for index, name_line in enumerate(name_lines):
        #     name_dict[index + 1] = name_line.strip()
        #
        # name_list = [name_dict[x] for x in layer_name_list]
        # print('要生成coe文件的层为：', name_list)
        # for name in name_list:
        #     gen_coe(os.path.join(api_coe_path, name + '.coe'), )


if __name__ == "__main__":
    quant_pth = 'yolox_quant_pth523/Epoch2-yolox_quantization_post.pth'
    img_path = 'img/001.jpg'
    para_path = 'para'
    api_coe_path = 'coe_file_24_001'
    # layer_name_list 输入想要写入coe文件的层数，可以查看model_structure 下面的 layer_name.txt
    # 常用的层数为 87 96 105  这三层为最终的3个yolo head 且为补完通道之后的结果，可直接和fpga的结果进行对比
    # 分别对应
    layer_name_list = [87, 96, 105]
    get_api_result(quant_pth, img_path, True, para_path)
