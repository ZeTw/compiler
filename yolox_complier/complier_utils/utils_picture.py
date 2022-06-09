import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from complier_utils.utils_bbox import non_max_suppression


def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485])
    image /= np.array([0.229])
    return image


def focus(x):
    patch_top_left = x[..., ::2, ::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_right = x[..., 1::2, 1::2]
    return patch_top_left, patch_bot_left, patch_top_right, patch_bot_right


def focus_add_zero(x):
    new_focus_feature = torch.zeros([1, 8, 320, 320])
    origin_shape = x.shape
    new_shape = new_focus_feature.shape

    for batch in range(1):
        for channel in range(4):
            for height in range(320):
                for width in range(320):
                    new_focus_feature[batch][channel][height][width] = x[batch][channel][height][width]
    with open('focus_add_channel.coe', 'w') as f:
        for batch in range(new_shape[0]):
            for height in range(new_shape[2]):
                for width in range(new_shape[3]):
                    for channel in range(new_shape[1]):

                        f.write('%02x' % int(new_focus_feature[batch][channel][height][width].item()))
                        if (channel + 1) % 4 == 0:
                            f.write('\n')

    return new_focus_feature


def picture_load(img_path):
    image = Image.open(img_path)

    crop_image = image.convert('L')
    crop_image = crop_image.resize((640, 640), Image.BICUBIC)
    photo = preprocess_input(np.array(crop_image, dtype=np.float32))
    image_data = np.expand_dims(photo, axis=0)
    image_data = np.expand_dims(image_data, axis=0)
    images = torch.from_numpy(np.asarray(image_data))
    return images


def detect_img(img_path, outputs):
    class_names = ['drone']
    image = Image.open(img_path)
    image_shape = np.array(np.shape(image)[0:2])
    results = non_max_suppression(outputs, 1, [640, 640],
                                  image_shape, True, conf_thres=0.5,
                                  nms_thres=0.3)

    if results[0] is None:
        return

    top_label = np.array(results[0][:, 6], dtype='int32')
    top_conf = results[0][:, 4] * results[0][:, 5]
    top_boxes = results[0][:, :4]
    # ---------------------------------------------------------#
    #   设置字体与边框厚度
    # ---------------------------------------------------------#

    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean([640, 640]), 1))

    # ---------------------------------------------------------#
    #   图像绘制
    # ---------------------------------------------------------#

    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline='red')
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill='red')
        draw.text(text_origin, str(label, 'UTF-8'), fill=("blue"), font=font)
        del draw

    return image
