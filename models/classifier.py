from models.blocks import *


class Classifier(nn.Module):

    def __init__(self, input_channels, hidden_channels=64):
        super(Classifier, self).__init__()
        self.upfeature = ContractingBlock(input_channels,
                                          hidden_channels,
                                          use_dropout=0.0,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=1,
                                          padding=0)
        self.contract1 = ContractingBlock(hidden_channels,
                                          2 * hidden_channels,
                                          use_dropout=0.5,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=1,
                                          padding=0)
        self.contract2 = ContractingBlock(2 * hidden_channels,
                                          4 * hidden_channels,
                                          use_dropout=0.5,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=1,
                                          padding=0)
        self.contract3 = ContractingBlock(4 * hidden_channels,
                                          8 * hidden_channels,
                                          use_dropout=0.0,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=1,
                                          padding=0)
        self.final = nn.Conv2d(8 * hidden_channels, 1, kernel_size=4, stride=1, padding=0)
        self.flt = nn.Flatten()

    def forward(self, x):
        x = interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.final(x3)
        x5 = self.flt(x4)
        return x5
