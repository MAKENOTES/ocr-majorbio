"""
用于文本识别的 CRNN 模型 (修复版: ResNet-34 + LogSoftmax + 智能初始化)。
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetFeatureExtractor(nn.Module):
    """
    基于 ResNet-34 修改的特征提取器，适配 32px 高度的输入。
    """
    def __init__(self, input_channel=1, output_channel=512):
        super(ResNetFeatureExtractor, self).__init__()

        # 加载 ResNet-34 预训练权重
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # --- [优化] 第一层卷积适配 ---
        # 原始权重形状: [64, 3, 7, 7]
        if input_channel == 1:
            # 如果是灰度图，我们将 RGB 权重的均值/和作为初始化
            # 这样可以保留预训练模型提取边缘、纹理的能力，避免冷启动
            self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                # sum(dim=1) 将 3通道权重合并为 1通道，keepdim=True 保持形状 [64, 1, 7, 7]
                self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
        else:
            # 如果是 RGB，直接使用原始层
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 截取 Layer1 - Layer4
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # 修改 Layer3 和 Layer4 的 stride (ResNet34 结构与 18 一致)
        # 目标: 高度下采样到 1 (32->1), 宽度下采样到 1/8 左右保留特征
        self.layer3 = resnet.layer3
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)

        self.layer4 = resnet.layer4
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

        # 最后的输出通道数 (ResNet34 layer4 也是 512)
        self.out_channels = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class CRNN(nn.Module):
    """
    ResNet34 + BiLSTM + CTC
    """
    def __init__(self, num_classes: int, img_height: int, num_channels: int, hidden_size: int, num_layers: int):
        super(CRNN, self).__init__()

        # 1. CNN Backbone
        self.cnn = ResNetFeatureExtractor(input_channel=num_channels, output_channel=512)

        # 2. RNN (Bi-LSTM)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        # 3. FC Head
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [batch, c, 32, 100]

        # CNN 特征提取
        conv = self.cnn(x)
        # conv: [batch, 512, 1, w_feat]

        # 维度调整
        conv = conv.squeeze(2) # [batch, 512, w_feat]
        conv = conv.permute(0, 2, 1) # [Batch, SeqLen, InputSize]

        # RNN 序列建模
        rnn_out, _ = self.rnn(conv)

        # FC 分类
        output = self.fc(rnn_out) # [Batch, SeqLen, NumClasses]

        # --- [关键修复] ---
        # CTC Loss 需要 Log Probabilities
        # 必须对 dim=2 (类别维度) 做 LogSoftmax
        output = F.log_softmax(output, dim=2)

        # 调整为 CTCLoss 要求的形状: [SeqLen, Batch, NumClasses]
        return output.permute(1, 0, 2)