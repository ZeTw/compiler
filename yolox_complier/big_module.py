import torch
from picture_load import picture_load, detect_img
from complier_utils.utils_bbox import decode_outputs
from complier_utils.utils_base import gen_coe


def focus(x):
    patch_top_left = x[..., ::2, ::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_right = x[..., 1::2, 1::2]
    return patch_top_left, patch_bot_left, patch_top_right, patch_bot_right


quant_pth_path = 'yolox_quant_pth523/Epoch2-yolox_quantization_post.pth'
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
# print(str(dark5_CSP))

dark5_CSP_Conv1 = dark5_CSP.conv1
dark5_CSP_Conv2 = dark5_CSP.conv2
dark5_CSP_Conv3 = dark5_CSP.conv3
dark5_CSP_csp_cat1 = dark5_CSP.csp1
dark5_CSP_m = dark5_CSP.m
# print(str(dark5_CSP_m))
# exit()
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

head_cat = head.csp6
quant = model.quant
dequant = model.dequant

# 模型定义完成，开始进行推断

with torch.no_grad():
    img = picture_load('img/001.jpg')
    quant_feature = model.quant(img)
    # =================start CSPDarknet=============
    patch_top_left, patch_bot_left, patch_top_right, patch_bot_right = focus(quant_feature)
    x = focus_csp0.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
    focus_cat_feature = x
    focus_conv_feature = stem_conv(x)
    # gen_coe('coe_file/focus_conv.coe', focus_conv_feature.int_repr())
    # exit()

    dark2_out = dark2(focus_conv_feature)
    # gen_coe('big_module_coe/dark2_output.coe',dark2_out.int_repr())
    # exit()
    dark3_out = dark3(dark2_out)
    # gen_coe('big_module_coe/dark3_output.coe',dark3_out.int_repr())
    dark4_out = dark4(dark3_out)
    # gen_coe('big_module_coe/dark4_output.coe', dark4_out.int_repr())
    dark5_out = dark5(dark4_out)

    feat1 = dark3_out
    feat2 = dark4_out
    feat3 = dark5_out

    P5 = lateral_conv0(feat3)
    P5_upsample = upsample(P5)
    P5_upsample_cat = YOLOPAFPN_csp2_cat.cat([P5_upsample, feat2], 1)
    C3_p4_out = C3_P4(P5_upsample_cat)
    # gen_coe('big_module_coe/C3_p4_out.coe', C3_p4_out.int_repr())
    P4 = reduce_conv1(C3_p4_out)
    P4_upsample = upsample(P4)
    P4_upsample_cat = YOLOPAFPN_csp3_cat.cat([P4_upsample, feat1], 1)
    P3_out = C3_P3(P4_upsample_cat)

    P3_downsample = bu_conv2(P3_out)
    # gen_coe('coe_file/bu_conv2.coe', P3_downsample.int_repr())

    P3_downsample_cat = YOLOPAFPN_csp4_cat.cat([P3_downsample, P4], dim=1)
    P4_out = C3_n3(P3_downsample_cat)

    P4_downsample = bu_conv1(P4_out)
    P4_downsample_cat = YOLOPAFPN_csp5_cat.cat([P4_downsample, P5], 1)
    P5_out = C3_n4(P4_downsample_cat)

    outputs = (P3_out, P4_out, P5_out)
    final_output = head(outputs)

    gen_coe('big_module_coe/P3_output.coe', final_output[0].int_repr())
    gen_coe('big_module_coe/P4_output.coe', final_output[1].int_repr())
    gen_coe('big_module_coe/P5_output.coe', final_output[2].int_repr())
    exit()

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
    # gen_coe('coe_file/dark2_BaseConv_feature.coe', dark2_BaseConv_feature.int_repr())
    # exit()
    dark2_x1 = dark2_CSP_Conv1(dark2_BaseConv_feature)
    # gen_coe('coe_file/dark2_CSP_Conv1.coe', dark2_x1.int_repr())
    # exit()
    dark2_x2 = dark2_CSP_Conv2(dark2_BaseConv_feature)
    # gen_coe('coe_file/dark2_CSP_Conv2.coe', dark2_x2.int_repr())
    # exit()

    # x1输入到m中
    dark2_CSP_m_conv1_feature = dark2_CSP_m_conv1(dark2_x1)
    # gen_coe('coe_file/dark2_CSP_m_Conv1.coe', dark2_CSP_m_conv1_feature.int_repr())
    dark2_CSP_m_conv2_feature = dark2_CSP_m_conv2(dark2_CSP_m_conv1_feature)
    # gen_coe('coe_file/dark2_CSP_m_Conv2.coe', dark2_CSP_m_conv2_feature.int_repr())
    dark2_CSP_m_feature = dark2_CSP_m_Bottleneck_add.add(dark2_x1, dark2_CSP_m_conv2_feature)
    # gen_coe('coe_file/dark2_CSP_m_add_feature.coe', dark2_CSP_m_feature.int_repr())
    # m结束，x2和m的输出进行cat
    dark2_CSP_cat_feature = dark2_CSP_csp_cat1.cat((dark2_CSP_m_feature, dark2_x2), dim=1)
    # gen_coe('coe_file/dark2_CSP_cat_feature.coe', dark2_CSP_cat_feature.int_repr())
    dark2_CSP_Conv3_feature = dark2_CSP_Conv3(dark2_CSP_cat_feature)
    # gen_coe('coe_file/dark2_CSP_Conv3_feature.coe', dark2_CSP_Conv3_feature.int_repr())
    # exit()
    # outputs['dark2']

    # ==================start dark3=================
    dark3_BaseConv_feature = dark3_BaseConv(dark2_CSP_Conv3_feature)
    # gen_coe('coe_file/dark3_BaseConv_feature.coe', dark3_BaseConv_feature.int_repr())
    # exit()
    dark3_x1 = dark3_CSP_Conv1(dark3_BaseConv_feature)
    # gen_coe('coe_file/dark3_CSP_Conv1.coe', dark3_x1.int_repr())

    dark3_x2 = dark3_CSP_Conv2(dark3_BaseConv_feature)
    # gen_coe('coe_file/dark3_CSP_Conv2.coe', dark3_x2.int_repr())
    # exit()

    # ============start dark3_m=============
    # m0
    dark3_Bottleneck_item0_conv1_feature = dark3_Bottleneck_item0_conv1(dark3_x1)
    # gen_coe('coe_file/dark3_CSP_m0_Conv1.coe', dark3_Bottleneck_item0_conv1_feature.int_repr())
    dark3_Bottleneck_item0_conv2_feature = dark3_Bottleneck_item0_conv2(dark3_Bottleneck_item0_conv1_feature)
    # gen_coe('coe_file/dark3_CSP_m0_Conv2.coe', dark3_Bottleneck_item0_conv2_feature.int_repr())
    # exit()
    dark3_Bottleneck_item0_feature = dark3_Bottleneck_item0_add.add(dark3_x1, dark3_Bottleneck_item0_conv2_feature)
    # gen_coe('coe_file/dark3_CSP_m0_add.coe', dark3_Bottleneck_item0_feature.int_repr())
    # m1
    dark3_Bottleneck_item1_conv1_feature = dark3_Bottleneck_item1_conv1(dark3_Bottleneck_item0_feature)
    # gen_coe('coe_file/dark3_CSP_m1_Conv1.coe', dark3_Bottleneck_item1_conv1_feature.int_repr())
    # exit()
    dark3_Bottleneck_item1_conv2_feature = dark3_Bottleneck_item1_conv2(dark3_Bottleneck_item1_conv1_feature)
    # gen_coe('coe_file/dark3_CSP_m1_Conv2.coe', dark3_Bottleneck_item1_conv2_feature.int_repr())
    dark3_Bottleneck_item1_feature = dark3_Bottleneck_item1_add.add(
        dark3_Bottleneck_item0_feature, dark3_Bottleneck_item1_conv2_feature
    )
    # gen_coe('coe_file/dark3_CSP_m1_add.coe', dark3_Bottleneck_item1_feature.int_repr())
    # exit()

    # m2
    dark3_Bottleneck_item2_conv1_feature = dark3_Bottleneck_item2_conv1(dark3_Bottleneck_item1_feature)
    # gen_coe('coe_file/dark3_CSP_m2_Conv1.coe', dark3_Bottleneck_item2_conv1_feature.int_repr())
    dark3_Bottleneck_item2_conv2_feature = dark3_Bottleneck_item2_conv2(dark3_Bottleneck_item2_conv1_feature)
    # gen_coe('coe_file/dark3_CSP_m2_Conv2.coe', dark3_Bottleneck_item2_conv2_feature.int_repr())
    dark3_Bottleneck_item2_feature = dark3_Bottleneck_item2_add.add(
        dark3_Bottleneck_item1_feature, dark3_Bottleneck_item2_conv2_feature)
    # gen_coe('coe_file/dark3_CSP_m2_add.coe', dark3_Bottleneck_item2_feature.int_repr())
    # exit()

    # dark3 csp
    dark3_CSP_cat_feature = dark3_CSP_csp_cat1.cat((dark3_Bottleneck_item2_feature, dark3_x2), dim=1)
    # gen_coe('coe_file/dark3_CSP_cat_feature.coe', dark3_CSP_cat_feature.int_repr())
    dark3_CSP_Conv3_feature = dark3_CSP_Conv3(dark3_CSP_cat_feature)
    # gen_coe('coe_file/dark3_CSP_Conv3_feature.coe', dark3_CSP_Conv3_feature.int_repr())
    # exit()
    # gen_coe('coe_file/dark3.coe',dark3_CSP_Conv3_feature.int_repr())
    # exit()
    # outputs['dark3']

    # ==================start dark4=================
    dark4_BaseConv_feature = dark4_BaseConv(dark3_CSP_Conv3_feature)
    # gen_coe('coe_file/dark4_BaseConv.coe', dark4_BaseConv_feature.int_repr())
    dark4_x1 = dark4_CSP_Conv1(dark4_BaseConv_feature)
    # gen_coe('coe_file/dark4_CSP_Conv1.coe', dark4_x1.int_repr())
    dark4_x2 = dark4_CSP_Conv2(dark4_BaseConv_feature)
    # gen_coe('coe_file/dark4_CSP_Conv2.coe', dark4_x2.int_repr())
    # ============start dark4_m=============
    # m0
    dark4_Bottleneck_item0_conv1_feature = dark4_Bottleneck_item0_conv1(dark4_x1)
    # gen_coe('coe_file/dark4_CSP_m0_Conv1.coe', dark4_Bottleneck_item0_conv1_feature.int_repr())
    dark4_Bottleneck_item0_conv2_feature = dark4_Bottleneck_item0_conv2(dark4_Bottleneck_item0_conv1_feature)
    # gen_coe('coe_file/dark4_CSP_m0_Conv2.coe', dark4_Bottleneck_item0_conv2_feature.int_repr())
    dark4_Bottleneck_item0_feature = dark4_Bottleneck_item0_add.add(dark4_x1, dark4_Bottleneck_item0_conv2_feature)
    # gen_coe('coe_file/dark4_CSP_m0_add.coe', dark4_Bottleneck_item0_feature.int_repr())
    # m1
    dark4_Bottleneck_item1_conv1_feature = dark4_Bottleneck_item1_conv1(dark4_Bottleneck_item0_feature)
    # gen_coe('coe_file/dark4_CSP_m1_Conv1.coe', dark4_Bottleneck_item1_conv1_feature.int_repr())
    dark4_Bottleneck_item1_conv2_feature = dark4_Bottleneck_item1_conv2(dark4_Bottleneck_item1_conv1_feature)
    # gen_coe('coe_file/dark4_CSP_m1_Conv2.coe', dark4_Bottleneck_item1_conv2_feature.int_repr())
    dark4_Bottleneck_item1_feature = dark4_Bottleneck_item1_add.add(dark4_Bottleneck_item0_feature,
                                                                    dark4_Bottleneck_item1_conv2_feature)
    # gen_coe('coe_file/dark4_CSP_m1_add.coe', dark4_Bottleneck_item1_feature.int_repr())

    # m2
    dark4_Bottleneck_item2_conv1_feature = dark4_Bottleneck_item2_conv1(dark4_Bottleneck_item1_feature)
    # gen_coe('coe_file/dark4_CSP_m2_Conv1.coe', dark4_Bottleneck_item2_conv1_feature.int_repr())
    dark4_Bottleneck_item2_conv2_feature = dark4_Bottleneck_item2_conv2(dark4_Bottleneck_item2_conv1_feature)
    # gen_coe('coe_file/dark4_CSP_m2_Conv2.coe', dark4_Bottleneck_item2_conv2_feature.int_repr())
    dark4_Bottleneck_item2_feature = dark4_Bottleneck_item2_add.add(dark4_Bottleneck_item1_feature,
                                                                    dark4_Bottleneck_item2_conv2_feature)
    # gen_coe('coe_file/dark4_CSP_m2_add.coe', dark4_Bottleneck_item2_feature.int_repr())

    # dark4 csp
    dark4_CSP_cat_feature = dark4_CSP_csp_cat1.cat((dark4_Bottleneck_item2_feature, dark4_x2), dim=1)
    # gen_coe('coe_file/dark4_CSP_cat.coe', dark4_CSP_cat_feature.int_repr())
    dark4_CSP_Conv3_feature = dark4_CSP_Conv3(dark4_CSP_cat_feature)
    # gen_coe('coe_file/dark4_CSP_Conv3.coe', dark4_CSP_Conv3_feature.int_repr())
    # exit()
    # outpus['dark4']

    # ==================start dark5=================
    dark5_BaseConv_feature = dark5_BaseConv(dark4_CSP_Conv3_feature)
    # gen_coe('coe_file/dark5_BaseConv.coe', dark5_BaseConv_feature.int_repr())
    dark5_x1 = dark5_CSP_Conv1(dark5_BaseConv_feature)
    # gen_coe('coe_file/dark5_CSP_Conv1.coe', dark5_x1.int_repr())
    dark5_x2 = dark5_CSP_Conv2(dark5_BaseConv_feature)
    # gen_coe('coe_file/dark5_CSP_Conv2.coe', dark5_x2.int_repr())

    # ============start dark5_m=============
    # m0
    dark5_Bottleneck_item0_conv1_feature = dark5_Bottleneck_item0_conv1(dark5_x1)
    # gen_coe('coe_file/dark5_CSP_m0_Conv1.coe', dark5_Bottleneck_item0_conv1_feature.int_repr())

    dark5_Bottleneck_item0_conv2_feature = dark5_Bottleneck_item0_conv2(dark5_Bottleneck_item0_conv1_feature)
    # gen_coe('coe_file/dark5_CSP_m0_Conv2.coe', dark5_Bottleneck_item0_conv2_feature.int_repr())

    dark5_Bottleneck_item0_feature = dark5_Bottleneck_item0_add.add(dark5_x1, dark5_Bottleneck_item0_conv2_feature)
    # gen_coe('coe_file/dark5_CSP_m0_add.coe', dark5_Bottleneck_item0_feature.int_repr())

    # dark5 csp
    dark5_CSP_cat_feature = dark5_CSP_csp_cat1.cat((dark5_Bottleneck_item0_feature, dark5_x2), dim=1)
    # gen_coe('coe_file/dark5_CSP_cat.coe', dark5_CSP_cat_feature.int_repr())
    dark5_CSP_Conv3_feature = dark5_CSP_Conv3(dark5_CSP_cat_feature)
    # gen_coe('coe_file/dark5_CSP_Conv3.coe', dark5_CSP_Conv3_feature.int_repr())

    # exit()
    # outputs['dark5']

    # ===============CSPDarknet 结束，开始YOLOPAFPN的其他部分===============
    feat1, feat2, feat3 = dark3_CSP_Conv3_feature, dark4_CSP_Conv3_feature, dark5_CSP_Conv3_feature
    P5 = lateral_conv0(feat3)
    # gen_coe('coe_file/P5.coe', P5.int_repr())
    P5_feature = P5
    P5_upsample = upsample(P5)
    # gen_coe('coe_file/P5_upsample.coe', P5_upsample.int_repr())
    P5_upsample_feature = P5_upsample
    P5_upsample_cat = YOLOPAFPN_csp2_cat.cat([P5_upsample, feat2], 1)
    # gen_coe('coe_file/P5_upsample_cat.coe', P5_upsample_cat.int_repr())

    P5_upsample_cat_feature = P5_upsample_cat

    # ==========start C3_p4================
    C3_P4_x1 = C3_P4_CSP_Conv1(P5_upsample_cat_feature)
    # gen_coe('coe_file/C3_P4_CSP_Conv1.coe', C3_P4_x1.int_repr())
    C3_P4_x2 = C3_P4_CSP_Conv2(P5_upsample_cat_feature)
    # gen_coe('coe_file/C3_P4_CSP_Conv2.coe', C3_P4_x2.int_repr())
    # ==========start C3_p4_m =============
    C3_P4_Bottleneck_item0_conv1_feature = C3_P4_m_Bottleneck_item0_conv1(C3_P4_x1)
    # gen_coe('coe_file/C3_P4_CSP_m0_Conv1.coe', C3_P4_Bottleneck_item0_conv1_feature.int_repr())
    C3_P4_Bottleneck_item0_conv2_feature = C3_P4_m_Bottleneck_item0_conv2(C3_P4_Bottleneck_item0_conv1_feature)
    # gen_coe('coe_file/C3_P4_CSP_m0_Conv2.coe', C3_P4_Bottleneck_item0_conv2_feature.int_repr())

    C3_P4_Bottleneck_item0_feature = C3_P4_m_Bottleneck_item0_add.add(C3_P4_x1, C3_P4_Bottleneck_item0_conv2_feature)
    # gen_coe('coe_file/C3_P4_CSP_m0_add.coe', C3_P4_Bottleneck_item0_feature.int_repr())

    C3_P4_CSP_cat_feature = C3_P4_CSP_cat.cat((C3_P4_Bottleneck_item0_feature, C3_P4_x2), dim=1)
    # gen_coe('coe_file/C3_P4_CSP_cat.coe', C3_P4_CSP_cat_feature.int_repr())
    C3_P4_CSP_Conv3_feature = C3_P4_CSP_Conv3(C3_P4_CSP_cat_feature)
    # gen_coe('coe_file/C3_P4_CSP_Conv3.coe', C3_P4_CSP_Conv3_feature.int_repr())

    # gen_coe('coe_file/c3_p4.coe', C3_P4_CSP_Conv3_feature.int_repr())
    # exit()
    P4 = reduce_conv1(C3_P4_CSP_Conv3_feature)
    # gen_coe('coe_file/reduce_conv1.coe', P4.int_repr())

    P4_upsample = upsample(P4)
    # gen_coe('coe_file/P4_upsample.coe', P4_upsample.int_repr())
    P4_upsample_cat_feature = YOLOPAFPN_csp3_cat.cat([P4_upsample, feat1], 1)
    # gen_coe('coe_file/P4_upsample_cat.coe', P4_upsample_cat_feature.int_repr())

    # ===============start C3_p3===============
    C3_P3_x1 = C3_P3_CSP_Conv1(P4_upsample_cat_feature)
    # gen_coe('coe_file/C3_P3_CSP_Conv1.coe', C3_P3_x1.int_repr())
    C3_P3_x2 = C3_P3_CSP_Conv2(P4_upsample_cat_feature)
    # gen_coe('coe_file/C3_P3_CSP_Conv2.coe', C3_P3_x2.int_repr())
    # ==========start C3_p3_m =============
    C3_P3_Bottleneck_item0_conv1_feature = C3_P3_m_Bottleneck_item0_conv1(C3_P3_x1)
    # gen_coe('coe_file/C3_P3_CSP_m0_Conv1.coe', C3_P3_Bottleneck_item0_conv1_feature.int_repr())
    C3_P3_Bottleneck_item0_conv2_feature = C3_P3_m_Bottleneck_item0_conv2(C3_P3_Bottleneck_item0_conv1_feature)
    # gen_coe('coe_file/C3_P3_CSP_m0_Conv2.coe', C3_P3_Bottleneck_item0_conv2_feature.int_repr())

    C3_P3_Bottleneck_item0_feature = C3_P3_m_Bottleneck_item0_add.add(C3_P3_x1, C3_P3_Bottleneck_item0_conv2_feature)
    # gen_coe('coe_file/C3_P3_CSP_m0_add.coe', C3_P3_Bottleneck_item0_feature.int_repr())

    C3_P3_CSP_cat_feature = C3_P3_CSP_cat.cat((C3_P3_Bottleneck_item0_feature, C3_P3_x2), dim=1)
    # gen_coe('coe_file/C3_P3_CSP_cat.coe', C3_P3_CSP_cat_feature.int_repr())
    C3_P3_CSP_Conv3_feature = C3_P3_CSP_Conv3(C3_P3_CSP_cat_feature)
    # gen_coe('coe_file/C3_P3_CSP_Conv3.coe', C3_P3_CSP_Conv3_feature.int_repr())
    # exit()

    P3_downsample = bu_conv2(C3_P3_CSP_Conv3_feature)
    # gen_coe('coe_file/bu_conv2.coe', P3_downsample.int_repr())

    P3_downsample_cat = YOLOPAFPN_csp4_cat.cat([P3_downsample, P4], dim=1)
    # gen_coe('coe_file/P3_downsample_cat.coe', P3_downsample_cat.int_repr())
    # exit()

    # ===============start C3_n3==============
    C3_n3_x1 = C3_n3_CSP_Conv1(P3_downsample_cat)
    # gen_coe('coe_file/C3_n3_CSP_Conv1.coe', C3_n3_x1.int_repr())
    C3_n3_x2 = C3_n3_CSP_Conv2(P3_downsample_cat)
    # gen_coe('coe_file/C3_n3_CSP_Conv2.coe', C3_n3_x2.int_repr())

    # ==========start C3_n3_m =============
    C3_n3_Bottleneck_item0_conv1_feature = C3_n3_m_Bottleneck_item0_conv1(C3_n3_x1)
    # gen_coe('coe_file/C3_n3_CSP_m0_conv1.coe', C3_n3_Bottleneck_item0_conv1_feature.int_repr())
    C3_n3_Bottleneck_item0_conv2_feature = C3_n3_m_Bottleneck_item0_conv2(C3_n3_Bottleneck_item0_conv1_feature)
    # gen_coe('coe_file/C3_n3_CSP_m0_conv2.coe', C3_n3_Bottleneck_item0_conv2_feature.int_repr())
    C3_n3_Bottleneck_item0_feature = C3_n3_m_Bottleneck_item0_add.add(C3_n3_x1, C3_n3_Bottleneck_item0_conv2_feature)
    # gen_coe('coe_file/C3_n3_CSP_m0_add.coe', C3_n3_Bottleneck_item0_feature.int_repr())

    C3_n3_CSP_cat_feature = C3_n3_CSP_cat.cat((C3_n3_Bottleneck_item0_feature, C3_n3_x2), dim=1)
    # gen_coe('coe_file/C3_n3_CSP_cat.coe', C3_n3_CSP_cat_feature.int_repr())
    C3_n3_CSP_Conv3_feature = C3_n3_CSP_Conv3(C3_n3_CSP_cat_feature)
    # gen_coe('coe_file/C3_n3_CSP_Conv3.coe', C3_n3_CSP_Conv3_feature.int_repr())

    P4_downsample = bu_conv1(C3_n3_CSP_Conv3_feature)
    # gen_coe('coe_file/P4_downsample.coe', P4_downsample.int_repr())
    P4_downsample_cat = YOLOPAFPN_csp5_cat.cat([P4_downsample, P5], 1)
    # gen_coe('coe_file/P4_downsample_cat.coe', P4_downsample_cat.int_repr())
    # exit()
    # ===============start C3_n4==============
    C3_n4_x1 = C3_n4_CSP_Conv1(P4_downsample_cat)
    # gen_coe('coe_file/C3_n4_CSP_Conv1.coe', C3_n4_x1.int_repr())
    C3_n4_x2 = C3_n4_CSP_Conv2(P4_downsample_cat)
    # gen_coe('coe_file/C3_n4_CSP_Conv2.coe', C3_n4_x2.int_repr())

    # ==========start C3_n4_m =============
    C3_n4_Bottleneck_item0_conv1_feature = C3_n4_m_Bottleneck_item0_conv1(C3_n4_x1)
    # gen_coe('coe_file/C3_n4_CSP_m0_conv1.coe', C3_n4_Bottleneck_item0_conv1_feature.int_repr())
    C3_n4_Bottleneck_item0_conv2_feature = C3_n4_m_Bottleneck_item0_conv2(C3_n4_Bottleneck_item0_conv1_feature)
    # gen_coe('coe_file/C3_n4_CSP_m0_conv2.coe', C3_n4_Bottleneck_item0_conv2_feature.int_repr())

    C3_n4_Bottleneck_item0_feature = C3_n4_m_Bottleneck_item0_add.add(C3_n4_x1, C3_n4_Bottleneck_item0_conv2_feature)
    # gen_coe('coe_file/C3_n4_CSP_m0_add.coe', C3_n4_Bottleneck_item0_feature.int_repr())

    C3_n4_CSP_cat_feature = C3_n4_CSP_cat.cat((C3_n4_Bottleneck_item0_feature, C3_n4_x2), dim=1)
    # gen_coe('coe_file/C3_n4_CSP_cat.coe', C3_n4_CSP_cat_feature.int_repr())

    C3_n4_CSP_Conv3_feature = C3_n4_CSP_Conv3(C3_n4_CSP_cat_feature)

    # gen_coe('coe_file/C3_n4_CSP_Conv3_feature.coe', C3_n4_CSP_Conv3_feature.int_repr())
    # exit()
    #  三个yolo head 分别对应 P3_out, P4_out, P5_out
    P3_out = C3_P3_CSP_Conv3_feature
    P4_out = C3_n3_CSP_Conv3_feature
    # gen_coe('coe_file/P4_out.coe', P4_out.int_repr())
    # exit()
    P5_out = C3_n4_CSP_Conv3_feature
    # for i in outputs:
    #     print(i.shape)
    # =========start yolo head ============

    P3_stem = stems_BaseConv0(P3_out)
    # gen_coe('coe_file/P3_stem.coe', P3_stem.int_repr())
    # exit()

    cls_convs_item0_BaseConv0_feature = cls_convs_item0_BaseConv0(P3_stem)
    # gen_coe('coe_file/P3_cls_conv0.coe', cls_convs_item0_BaseConv0_feature.int_repr())
    cls_convs_item0_BaseConv1_feature = cls_convs_item0_BaseConv1(cls_convs_item0_BaseConv0_feature)
    # gen_coe('coe_file/P3_cls_conv1.coe', cls_convs_item0_BaseConv1_feature.int_repr())
    P3_cls_feat = cls_convs_item0_BaseConv1_feature
    # gen_coe('coe_file/P3_cls_feat.coe', P3_cls_feat.int_repr())
    # exit()
    P3_cls_output = cls_preds_Conv2d0(P3_cls_feat)
    # gen_coe('coe_file/P3_cls_preds.coe', P3_cls_output.int_repr())
    # exit()
    reg_convs_item0_BaseConv0_feature = reg_convs_item0_BaseConv0(P3_stem)
    # gen_coe('coe_file/P3_reg_conv0.coe', reg_convs_item0_BaseConv0_feature.int_repr())
    reg_convs_item0_BaseConv1_feature = reg_convs_item0_BaseConv1(reg_convs_item0_BaseConv0_feature)
    # gen_coe('coe_file/P3_reg_conv1.coe', reg_convs_item0_BaseConv1_feature.int_repr())
    P3_reg_feat = reg_convs_item0_BaseConv1_feature

    P3_reg_output = reg_preds_Conv2d0(P3_reg_feat)
    # gen_coe('coe_file/P3_reg_preds.coe', P3_reg_output.int_repr())
    # exit()
    P3_obj_output = obj_preds_Conv2d0(P3_reg_feat)
    # gen_coe('coe_file/P3_obj_preds.coe', P3_obj_output.int_repr())
    # exit()
    P3_output = head_cat.cat([P3_reg_output, P3_obj_output, P3_cls_output], 1)
    # gen_coe('coe_file/P3_output.coe', P3_output.int_repr())
    # exit()

    P4_stem = stems_BaseConv1(P4_out)
    # gen_coe('coe_file/P4_stem.coe', P4_stem.int_repr())
    # exit()
    cls_convs_item1_BaseConv0_feature = cls_convs_item1_BaseConv0(P4_stem)
    cls_convs_item1_BaseConv1_feature = cls_convs_item1_BaseConv1(cls_convs_item1_BaseConv0_feature)
    P4_cls_feat = cls_convs_item1_BaseConv1_feature
    # gen_coe('coe_file/P4_cls_feat.coe', P4_cls_feat.int_repr())
    # exit()
    P4_cls_output = cls_preds_Conv2d1(P4_cls_feat)

    reg_convs_item1_BaseConv0_feature = reg_convs_item1_BaseConv0(P4_stem)
    reg_convs_item1_BaseConv1_feature = reg_convs_item1_BaseConv1(reg_convs_item1_BaseConv0_feature)
    P4_reg_feat = reg_convs_item1_BaseConv1_feature
    P4_reg_output = reg_preds_Conv2d1(P4_reg_feat)
    P4_obj_output = obj_preds_Conv2d1(P4_reg_feat)
    P4_output = head_cat.cat([P4_reg_output, P4_obj_output, P4_cls_output], 1)
    # gen_coe('coe_file/P4_output.coe', P4_output.int_repr())
    # exit()

    P5_stem = stems_BaseConv2(P5_out)  # 100
    # gen_coe('coe_file/P5_stem.coe', P5_stem.int_repr())
    # exit()
    cls_convs_item2_BaseConv0_feature = cls_convs_item2_BaseConv0(P5_stem)  # 100
    # gen_coe('coe_file/P5_cls_conv0.coe', cls_convs_item2_BaseConv0_feature.int_repr())
    # exit()
    cls_convs_item2_BaseConv1_feature = cls_convs_item2_BaseConv1(cls_convs_item2_BaseConv0_feature)
    # gen_coe('coe_file/P5_cls_conv1.coe', cls_convs_item2_BaseConv1_feature.int_repr())
    # exit()
    P5_cls_feat = cls_convs_item2_BaseConv1_feature
    P5_cls_output = cls_preds_Conv2d2(P5_cls_feat)  # 100
    # gen_coe('coe_file/P5_cls_feat.coe', P5_cls_output.int_repr())
    # exit()

    reg_convs_item2_BaseConv0_feature = reg_convs_item2_BaseConv0(P5_stem)
    reg_convs_item2_BaseConv1_feature = reg_convs_item2_BaseConv1(reg_convs_item2_BaseConv0_feature)
    P5_reg_feat = reg_convs_item2_BaseConv1_feature
    P5_reg_output = reg_preds_Conv2d2(P5_reg_feat)
    # gen_coe('coe_file/P5_reg_output.coe', P5_reg_output.int_repr())
    # exit()
    P5_obj_output = obj_preds_Conv2d2(P5_reg_feat)
    # gen_coe('coe_file/P5_obj_output.coe', P5_obj_output.int_repr())
    # exit()

    P5_output = head_cat.cat([P5_reg_output, P5_obj_output, P5_cls_output], 1)
    # gen_coe('coe_file/P5_output.coe', P5_output.int_repr())
    # exit()
    out_list = []
    head_outputs = [P3_output, P4_output, P5_output]
    for i in head_outputs:
        out_list.append(dequant(i))

    final_output = decode_outputs(out_list, [640, 640])
    # print(final_output)
    # print(final_output.shape)
    P3_final = dequant(P3_output)
    P4_final = dequant(P4_output)
    P5_final = dequant(P5_output)

    out = model(img)
    # print(P4_final == out[1])
    # a = out[0] == P3_final
    # nump_a = a.numpy()
    # mask = np.unique(nump_a)
    # tmp = {}
    # for v in mask:
    #     tmp[v] = np.sum(nump_a == v)
    # print(tmp)
    # print(nump_a == out[0])
    # print()
    out = decode_outputs(out, [640, 640])
    img_result = detect_img('img/001.jpg', outputs=out)
    img_result.show()
