import torch
import torch.nn.functional as F


class ConvNormRelu(torch.nn.Module):
    """A stack of a convolutional net, a batch norm, and a ReLu"""

    def __init__(
        self, in_channels, out_channels, kernel_size, dropout=False, p=0.2
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.batchNorm = torch.nn.BatchNorm2d(out_channels)
        self.dropout = dropout
        self.p = p

    def forward(self, batch):
        if self.dropout:
            batch = torch.nn.Dropout(self.p)(batch)
        return F.relu(self.batchNorm(self.conv(batch)))


class CNN(torch.nn.Module):
    """CNN for feature extraction"""

    def __init__(self, device) -> None:
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            ConvNormRelu(1,64,(3,3)),
            torch.nn.MaxPool2d((2,2))
        )

        self.layer2 = torch.nn.Sequential(
            ConvNormRelu(64,128,(3,3)),
            torch.nn.MaxPool2d((2,2))
        )

        self.layer3 = torch.nn.Sequential(
            ConvNormRelu(128,256,(3,3)),
            ConvNormRelu(256,256,(3,3)),
            torch.nn.MaxPool2d((1,2)),
            ConvNormRelu(256,512,(3,3)),
            torch.nn.MaxPool2d((2,1))
        )


        self.layer4 = ConvNormRelu(512,512,(3,3)).to(device)

    def forward(self, batch):
        out = self.layer1(batch)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
