import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel

class Identity(nn.Module):
    """
    相等映射层
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Bert_LSTM_cat(nn.Module):
    """
    1.使用CNN提取切分后笔迹图片的图片特征(64维)+笔迹特征(9维)=73维；
    2.使用双向LSTM提取上述图片特征的序列特征(128维)；
    3.使用BERT提取笔迹图片内容的文本特征(768维)；
    4.拼接2.3.两个128维特征，通过一个fc层分类
    """
    def get_model(self,path):
        model = torch.load(path)
        model.fc = Identity()
        return model
    def __init__(self,input_size):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-chinese')
        self.bilstm = nn.LSTM(input_size, 64, 1, bidirectional=True,batch_first=True)
        # print(self.img_model)
        self.t_linear = nn.Linear(768, 128)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.2, inplace=True)

    def forward(self, input_ids, attention_mask, image):
        output, (hidden, c_n) = self.bilstm(image)
        # print("input:{} -> output:{}".format(input.shape,output.shape))
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), 1)
        img_out = self.relu(hidden)
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:,0,:]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.dropout(txt_out)
        txt_out = self.t_linear(txt_out)
        txt_out = self.relu(txt_out)
        last_out = torch.cat([txt_out,img_out],dim=1)
        last_out = self.fc(last_out)
        return last_out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # in:1x64x64 -> 6x60x60
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)     # in:6x60x60 -> 6x30x30
        self.conv2 = nn.Conv2d(6, 16, 5) # in:6x30x30 -> 16x26x26
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)     # in:16x26x26 -> 16x13x13
        self.fc1 = nn.Linear(2704, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(512, 64)
        self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(64, 3)
        self.relu6 = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.conv1(x)
        # print("in:",x.shape,"-> ",y.shape)
        y = self.relu1(y)
        y = self.tanh(y)
        y = self.pool1(y)
        # print(y.shape)
        y = self.conv2(y)
        # print(y.shape)
        y = self.relu2(y)
        y = self.tanh(y)
        y = self.pool2(y)
        # print(y.shape)
        y = y.view(y.shape[0], -1)
        # print(y.shape)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.tanh(y)
        y = self.fc2(y)
        # y = self.relu4(y)
        y = self.tanh(y)
        y = self.fc3(y)
        # y = self.relu5(y)
        y = self.tanh(y)
        y = self.fc4(y)
        # y = self.relu6(y)
        y = self.tanh(y)
        return y