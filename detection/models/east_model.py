import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EASTModel(nn.Module):
    """
    EAST (Efficient and Accurate Scene Text Detector) 模型。
    该模型由一个特征提取主干、一个特征融合部分和一个输出层组成，
    用于预测文本分数图和几何图。
    """
    def __init__(self, pretrained=True):
        """
        初始化 EAST 模型。
        :param pretrained: bool, 是否使用在 ImageNet 上预训练的 ResNet-50 权重。
        """
        super(EASTModel, self).__init__()
        # 加载预训练的 ResNet-50，并移除全连接层
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
        resnet = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 特征金字塔网络 (FPN) 风格的上采样
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        
        # 输出层
        # 分数图 (Score map): 1个通道，代表像素点是文本的概率
        self.score_conv = nn.Conv2d(256, 1, kernel_size=1)
        # 几何图 (Geometry map): 4个通道，代表像素点到上、右、下、左边界的距离 (RBOX)
        self.geo_conv = nn.Conv2d(256, 4, kernel_size=1)

    def forward(self, x):
        """
        模型的前向传播过程。
        :param x: 输入的图像张量，形状为 (批次大小, 3, 高, 宽)。 images 这个变量内部实实在在地携带着每一个像素的颜色信息, 通过类似 images[0, :, 100, 200]  去读取。
        :return: 一个元组，包含预测的分数图和几何图。
                 - 分数图形状: (批次大小, 1, 高/4, 宽/4)
                 - 几何图形状: (批次大小, 5, 高/4, 宽/4)
        """
        # 特征提取
        f = self.features(x)
        
        # 类似FPN的上采样
        # ResNet50特征提取器的输出是 (批次, 2048, H/32, W/32)
        # 我们希望恢复到 (批次, C, H/4, W/4)
        y = self.upconv1(f)    # -> (批次, 1024, H/16, W/16)
        y = self.upconv2(y)    # -> (批次, 512, H/8, W/8)
        y = self.upconv3(y)    # -> (批次, 256, H/4, W/4)
        
        # 预测
        score = self.score_conv(y)
        score = torch.sigmoid(score) # 归一化到 [0, 1] 区间
        
        geo = self.geo_conv(y)
        # 既然是纯水平，geo 的 4 个通道全是距离。
        # 直接用 ReLU 保证非负，且无上限（解决长文本截断问题）。
        # 不需要 sigmoid，不需要 * 512
        geo = torch.relu(geo)

        
        return score, geo

if __name__ == '__main__':
    # 一个快速测试，用来检查模型各维度是否正确
    model = EASTModel()
    # 一个 512x512 图像的输入张量
    test_input = torch.randn(1, 3, 512, 512)
    score_map, geo_map = model(test_input)
    
    print("模型初始化成功!")
    print(f"输入形状: {test_input.shape}")
    # 输出的特征图尺寸是输入的 1/4
    print(f"分数图形状: {score_map.shape}") # 应该是 (1, 1, 128, 128)
    print(f"几何图形状: {geo_map.shape}")   # 应该是 (1, 5, 128, 128)
