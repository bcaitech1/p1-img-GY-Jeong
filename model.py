import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from facenet_pytorch import MTCNN, InceptionResnetV1
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class ResnextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnext50 = models.resnext50_32x4d(pretrained=True)
        self.resnext50.fc = nn.Linear(2048, num_classes)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnext50(x)
        return x

class FacenetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2', num_classes=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

class EfficientnetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
        # self.enet.fc = nn.Linear(2034, 32)
        # print(self.enet)
    
    def forward(self, x):
        x = self.enet(x)
        return x

class Efficientnet4Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
        # self.enet._fc = nn.Sequential(
        #     nn.Linear(2304, 512),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x):
        x = self.enet(x)
        return x

class VITModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vnet = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.vnet(x)