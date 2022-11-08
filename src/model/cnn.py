import torch.nn as nn
import torch.nn.functional as F

class ConvNormRelu(nn.Module):
    """A stack of a convolutional net, a batch norm, and a ReLu"""
    def __init__(self, in_channels, out_channels, kernel_size, dropout=False,p=0.2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.dropout = dropout
        self.p = p
    
    def forward(self,batch):
        if self.dropout:
            batch = nn.Dropout(self.p)(batch)
        return F.relu(self.batchNorm(self.conv(batch)))


class CNN(nn.Module):
    """CNN for feature extraction"""
    def __init__(self) -> None:
        super().__init__()
        
        self.layer1 = nn.Sequential(
            ConvNormRelu(1,100, (3,3)),
            ConvNormRelu(100,100, (3,3)),
            ConvNormRelu(100,100, (3,3)),
            nn.MaxPool2d((2,2))
        )
        
        self.layer2 = nn.Sequential(
            ConvNormRelu(100, 200, (3,3)),
            ConvNormRelu(200, 200, (3,3)),
            ConvNormRelu(200, 200, (3,3)),
            nn.MaxPool2d((2,2))
        )

        self.layer3 = nn.Sequential(
            ConvNormRelu(200,300,(3,3), True, 0.2),
            ConvNormRelu(300,300,(3,3), True, 0.2),
            ConvNormRelu(300,300,(3,3), True, 0.2),
            nn.MaxPool2d((1,2))
        )

        self.layer4 = nn.Sequential(
            ConvNormRelu(300,400,(3,3), True, 0.2),
            ConvNormRelu(400,400,(3,3), True, 0.2),
            ConvNormRelu(400,400,(3,3), True, 0.2),
            nn.MaxPool2d((2,1)),
            ConvNormRelu(400,512, (3,3), True, 0.2)
        )

    def forward(self,batch):
        out = self.layer1(batch)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

