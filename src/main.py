import yaml
import torch
import cv2
import easyocr
import numpy as np
import transformers
from model import Bert_LSTM_cat,Model
from util import char_split,img_process,img_feature_ext,txt_embed
from transformers import BertTokenizer
from config import config

if __name__ == '__main__':
    # 设置device为cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(transformers.__version__)

    # 加载并处理测试图片
    print('>> 正在对测试图片进行处理...')
    test_pic_path = config['test_pic']['path']
    test_pic = cv2.imread(test_pic_path)
    chars = img_process(test_pic_path)
    print('>> 测试图片处理完成')

    # 使用pretrainedCNN网络和笔迹特征提取器提取特征
    print('>> 正在提取图片特征')
    CNN_model = torch.load(config['checkpoint']['CNN_path'],map_location=device)
    test_x = img_feature_ext(CNN_model,chars,device)

    # 加载并提取文字特征
    if config['test_pic']['content'] == 'ocr':
        pic_txt = ''.join([i[-2] for i in easyocr.Reader(['ch_sim']).readtext(test_pic_path)])
    else:
        pic_txt = config['test_pic']['content']
    token = BertTokenizer.from_pretrained(config['checkpoint']['tokenizer_path'])
    idx,attention_mask = txt_embed([pic_txt],token)

    # 加载模型
    model = torch.load(config['checkpoint']['fusion_path'],map_location=device)
    model.eval()
    # 将test_x这个tensor向后升维
    test_x=torch.tensor(np.array(test_x)).unsqueeze(0).float()
    test_y = model(idx.to(device),attention_mask.to(device),test_x.to(device))
    # 选取最大的预测值作为其标签
    test_y = int(torch.argmax(test_y,dim=1).detach())
    print('>> 其标签是',test_y)


