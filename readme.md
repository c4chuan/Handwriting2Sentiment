# Handwriting2Sentiment

## 项目简介

Handwriting2Sentiment是一个通过用户的手写笔迹分析用户情感的端到端框架

## 项目结构

```
├─bert-base-chinese                                  # BERT Tokenizer
├─checkpoints                                        # 预训练模型
│      pretained_CNN_acc_0.978.pkl                   # 预训练CNN分类器
│      txt_pic_pic_writing_valid_acc_0.786.pkl       # 预训练多模态分类模型
│      
├─data                                               # 训练数据不公开
│      test_pic.jpg
│      
└─src                                                # 源代码
    │  config.py                                     # 设置全局config
    │  config.yaml                                   # 配置文件
    │  main.py                                       # 主要运行文件
    │  model.py                                      # 模型结构定义
    │  util.py                                       # 工具函数                            
            
```

## 配置项

```yaml
model:
  # 每个字的图片特征(64)+笔迹特征(9)，不建议修改，除非使用其他预训练模型
  input_size: 73
checkpoint:
  # 多模态模型的预训练模型
  fusion_path: ../checkpoints/txt_pic_pic_writing_valid_acc_0.786.pkl
  # 手写数字识别预训练出来的CNN特征提取器
  CNN_path: ../checkpoints/models/pretained_CNN_acc_0.978.pkl
  # 预训练的Bert模型(可以去huggingface或者镜像网站下载)
  tokenizer_path: ../bert-base-chinese
test_pic:
  # 笔迹图片切字后的采样数目（若手写的字比较少的话可能需要设置小一点）
  num_cut: 50
  # 测试图片路径
  path: ../data/test_pic.jpg
  # 切字的时候字的边界，默认为[上，下，左，右]，默认为[13，55，10，55]，可根据字大小调整，单位是px
  text_border: [13,55,10,55]
  # 测试图片的文字信息，若不知道可以设置为"ocr"，程序会自动识别（目前效果不太好）
  content: "我最近过的还不错"
```

## 虚拟环境配置

需要注意的是transformers的版本不对可能会引起bug，建议安装4.35版本

## 使用方法

在配置中配置好图片地址、切字数目、border和文字信息然后运行main.py即可

返回值是一个三分类（0,1,2分别对应焦虑程度低，中等，高）

## 数据获取

仅开源模型参数

[百度云](https://pan.baidu.com/s/1tEN_ssxbF1psbc4dyAA6Gw?pwd=v6jk )
vx:w1449870568