import random
import torch
import cv2
import numpy as np
from config import config
from PIL import Image
from torchvision import transforms
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData
def char_split(img):
    # 输出图像尺寸和通道信息
    sp = img.shape
    # print("图像信息：", sp)
    sz1 = sp[0]  # height(rows) of image
    sz2 = sp[1]  # width(columns) of image
    sz3 = sp[2]  # the pixels value is made up of three primary colors
    # print('width: %d \n height: %d \n number: %d' % (sz2, sz1, sz3))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("threshold_img", threshold_img)

    # 水平投影分割图像
    gray_value_x = []
    for i in range(sz1):
        white_value = 0
        for j in range(sz2):
            if threshold_img[i, j] == 255:
                white_value += 1
        gray_value_x.append(white_value)
    # print("", gray_value_x)
    # 创建图像显示水平投影分割图像结果
    hori_projection_img = np.zeros((sp[0], sp[1], 1), np.uint8)
    for i in range(sz1):
        for j in range(gray_value_x[i]):
            hori_projection_img[i, j] = 255
    # cv2.imshow("hori_projection_img", hori_projection_img)
    text_rect = []
    # 根据水平投影分割识别行
    inline_x = 0
    start_x = 0
    text_rect_x = []
    for i in range(len(gray_value_x)):
        if inline_x == 0 and gray_value_x[i] > 10:
            inline_x = 1
            start_x = i
        elif inline_x == 1 and gray_value_x[i] < 10 and (i - start_x) > 5:
            inline_x = 0
            if i - start_x > 10:
                rect = [start_x - 1, i + 1]
                text_rect_x.append(rect)
    # 每行数据分段
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    dilate_img = cv2.dilate(threshold_img, kernel)
    # dilate_img = threshold_img
    # cv2.imshow("dilate_img", dilate_img)
    for rect in text_rect_x:
        cropImg = dilate_img[rect[0]:rect[1], 0:sp[1]]  # 裁剪图像y-start:y-end,x-start:x-end
        sp_y = cropImg.shape
        # 垂直投影分割图像
        gray_value_y = []
        for i in range(sp_y[1]):
            white_value = 0
            for j in range(sp_y[0]):
                if cropImg[j, i] == 255:
                    white_value += 1
            gray_value_y.append(white_value)
        # 创建图像显示水平投影分割图像结果
        veri_projection_img = np.zeros((sp_y[0], sp_y[1], 1), np.uint8)
        for i in range(sp_y[1]):
            for j in range(gray_value_y[i]):
                veri_projection_img[j, i] = 255
        # 根据垂直投影分割识别行
        inline_y = 0
        start_y = 0
        text_rect_y = []
        for i in range(len(gray_value_y)):
            if inline_y == 0 and gray_value_y[i] > 2:
                inline_y = 1
                start_y = i
            elif inline_y == 1 and gray_value_y[i] < 2 and (i - start_y) > 5:
                inline_y = 0
                if i - start_y > 10:
                    rect_y = [start_y - 1, i + 1]
                    text_rect_y.append(rect_y)
                    text_rect.append([rect[0], rect[1], start_y - 1, i + 1])
                    cropImg_rect = threshold_img[rect[0]:rect[1], start_y - 1:i + 1]  # 裁剪图像

    return text_rect

def img2charsbyrect(img,text_rect):
    print('>> 将图片切分成字符的图片')
    text_border = config['test_pic']['text_border']
    rec_imgs = []
    for rect_roi in text_rect:
        height = rect_roi[1] - rect_roi[0]
        width = rect_roi[3] - rect_roi[2]
        if height > text_border[0] and height < text_border[1] and width > text_border[2] and width < text_border[3]:
            rectangle_img = img[rect_roi[0]:rect_roi[1], rect_roi[2]:rect_roi[3], :]
            print(
                'org_img:{},{}->[{},{},{},{}]'.format(img.shape[0], img.shape[1], rect_roi[0], rect_roi[1], rect_roi[2],
                                                      rect_roi[3]))
            rec_imgs.append(rectangle_img)
    return random.sample(rec_imgs,config['test_pic']['num_cut'])

def img_process(img_path):
    img = cv2.imread(img_path)
    rec_imgs = img2charsbyrect(img,char_split(img))
    return rec_imgs
def char_feature_ext(img):
    size = 64
    feature_list = []

    height = img.shape[0]
    width = img.shape[1]
    # print(img.shape)
    org_img = np.zeros((size,size))
    left = int((size - width) / 2)
    up = int((size - height) / 2)
    for i in range(up, up + height):
        for j in range(left, left + width):
            if img[i - up][j - left] == 255:
                org_img[i][j] = 255
    # cv2.imshow('test', org_img)
    # cv2.waitKey(0)


    #1) 提取文字的宽度
    left_most = 65
    right_most = -1
    for j in range(size):
        for i in range(size):
            if org_img[i][j] == 255 and i<left_most:
                left_most = i
            if org_img[i][j] == 255 and i>right_most:
                right_most = i
    width_fea = right_most-left_most
    # print("宽度：",width_fea)

    #2) 提取文字的高度
    up_most = 65
    down_most = -1
    for i in range(size):
        for j in range(size):
            if org_img[i][j] == 255 and j<up_most:
                up_most = j
            if org_img[i][j] == 255 and j>down_most:
                down_most = j
    height_fea = down_most-up_most
    # print("高度：",height_fea)

    #3) 文字宽高比
    if height_fea == 0:
        print(org_img.shape)
        print(height,width)
        print(left_most,right_most,down_most, up_most)
        cv2.imshow('test', org_img)
        cv2.waitKey(0)

    ratio_fea = width_fea/height_fea

    # print("宽高比：",ratio_fea)

    #4) 文字面积
    area_fea = width_fea*height_fea
    # print("面积：",area_fea)

    #5) 倾角
    slant_fea = -1
    for i in range(size):
        for j in range(size):
            if org_img[i][j] == 255 and (j/i) >slant_fea:
                slant_fea = (j/i)
    # print("倾角：",slant_fea)

    #6) 上下左右边距
    left_fea = left_most
    right_fea = size - right_most
    up_fea = up_most
    down_fea = size-down_most
    # print("上下左右边距：",up_fea,down_fea,left_fea,right_fea)

    feature_list.append(width_fea)
    feature_list.append(height_fea)
    feature_list.append(ratio_fea)
    feature_list.append(area_fea)
    feature_list.append(slant_fea)
    feature_list.append(up_fea)
    feature_list.append(down_fea)
    feature_list.append(left_fea)
    feature_list.append(right_fea)

    # print("文字特征向量为：",feature_list)
    return feature_list
def img_feature_ext(CNN_model,pic_list,device):
    person_list = []
    person_writing_list = []
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor()])
    model = CNN_model
    model.tanh = Identity()
    model.relu6 = Identity()
    model.fc4 = Identity()
    model.sigmoid = Identity()

    for pic in pic_list:
        # 将三通道的图片pic转化为单通道二值图
        gray_image = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        _, char_pic = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        char_feature = char_feature_ext(char_pic)
        person_writing_list.append(char_feature)

        # CNN特征提取
        pic = cv2.resize(char_pic, (64, 64))
        cnn_pic = Image.fromarray(pic)
        now_pic = transform(cnn_pic)
        now_pic = now_pic.to(device)
        person_list.append(now_pic)
    person_writing_arr = noramlization(np.array(person_writing_list))
    person_writing_tensor = torch.tensor(person_writing_arr)
    person_pic_tensor = torch.stack([pic_temp for pic_temp in person_list])
    person_pic_tensor = model(person_pic_tensor.float()).detach()
    person_pic_tensor = person_pic_tensor.to('cpu')

    person_tensor = torch.cat([person_pic_tensor, person_writing_tensor], 1)
    print(person_tensor.shape)
    s_numpy = [x.numpy() for x in person_tensor]  # 步骤1

    return s_numpy

def txt_embed(txt, token):

    result = token.batch_encode_plus(batch_text_or_text_pairs=txt, truncation=True, padding='max_length', max_length=380,
                                     return_tensors='pt')
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    return input_ids, attention_mask