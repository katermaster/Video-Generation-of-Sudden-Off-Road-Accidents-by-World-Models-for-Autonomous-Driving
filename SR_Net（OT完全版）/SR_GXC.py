import torch
import torch.nn as nn
from torchsummary import summary

class Type1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Type1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.type1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        ans = self.type1(inputs)
        # print('ans shape: ', ans.shape)
        return ans

class Type2(nn.Module):
    def __init__(self):
        super(Type2, self).__init__()
        self.type2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
        )

    def forward(self, inputs):
        ans = torch.add(inputs, self.type2(inputs))
        # print('ans shape: ', ans.shape)
        return ans

class Type3(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Type3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

        self.type3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self,inputs):
        ans = torch.add(self.shortcut(inputs), self.type3(inputs))
        # print('ans shape: ', ans.shape)
        return ans

class Type4(nn.Module):
    def __init__(self):
        super(Type4, self).__init__()
        self.type4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self,inputs):
        ans0 = self.type4(inputs)
        ans = ans0.view(-1, 512)
        return  ans


class SR_Net(nn.Module):
    def __init__(self,init_weights=True):
        super(SR_Net, self).__init__()
        # 第一种结构类型
        self.layer1 = Type1(1, 64)
        self.layer2 = Type1(64, 16)

        # 第二种结构类型
        self.layer3 = Type2()
        self.layer4 = Type2()
        self.layer5 = Type2()
        self.layer6 = Type2()
        self.layer7 = Type2()

        # 第三种类型
        self.layer8 = Type3(16, 16)
        self.layer9 = Type3(16, 64)
        self.layer10 = Type3(64, 128)
        self.layer11 = Type3(128, 256)

        # 第四种类型
        self.layer12 = Type4()
        #self.layer12 = Type4(256, 512)

        # 最后一层，全连接层
        self.layer13 = nn.Linear(512, 2)

        if init_weights:
            self.initialize_weights()



    def forward(self, inputs):
        # 第一种结构类型
        x = self.layer1(inputs)
        x = self.layer2(x)

        # 第二种结构类型
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # 第三种类型
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        # 第四种类型
        x = self.layer12(x)
        features = x  # 保存特征
        #print(x.shape)
        # 最后一层全连接
        outputs = self.layer13(x)
        #print('self.outputs.shape: ', self.outputs.shape)
        return outputs, features

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.2)
                #nn.init.normal_(m.weight, mean=0, std=0.01)  # L2 正则化

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

if __name__ == '__main__':
    x = torch.randn(size=(1, 1, 256, 256)) .cuda()
    print(x.shape)
    net = SR_Net()
    print(summary(net.cuda(), (1, 256, 256)))