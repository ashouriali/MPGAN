from models.blocks import *


class UNet(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(UNet, self).__init__()
        self.upfeature0 = ContractingBlock(input_channels,
                                           hidden_channels,
                                           use_leaky_relu=0.2,
                                           use_dropout=0.0,
                                           use_bn=True,
                                           use_maxp=False,
                                           kernel_size=4,
                                           stride=1,
                                           padding=0)
        self.upfeature1 = ContractingBlock(input_channels,
                                           hidden_channels,
                                           use_leaky_relu=0.2,
                                           use_dropout=0.0,
                                           use_bn=True,
                                           use_maxp=False,
                                           kernel_size=4,
                                           stride=1,
                                           padding=0)
        self.contract1 = ContractingBlock(hidden_channels,
                                          2 * hidden_channels,
                                          use_leaky_relu=0.2,
                                          use_dropout=0.0,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1
                                          )
        self.contract2 = ContractingBlock(2 * hidden_channels,
                                          4 * hidden_channels,
                                          use_leaky_relu=0.2,
                                          use_dropout=0.0,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1
                                          )
        self.contract3 = ContractingBlock(4 * hidden_channels,
                                          8 * hidden_channels,
                                          use_leaky_relu=0.2,
                                          use_dropout=0.0,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.contract4 = ContractingBlock(8 * hidden_channels,
                                          8 * hidden_channels,
                                          use_leaky_relu=0.2,
                                          use_dropout=0.0,
                                          use_bn=True,
                                          use_maxp=False,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.expand2 = ExpandingBlock(8 * hidden_channels,
                                      8 * hidden_channels,
                                      use_leaky_relu=0.0,
                                      use_dropout=0.5,
                                      use_bn=True,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1)
        self.expand3 = ExpandingBlock(16 * hidden_channels,
                                      4 * hidden_channels,
                                      use_leaky_relu=0.0,
                                      use_dropout=0.5,
                                      use_bn=True,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1)
        self.expand4 = ExpandingBlock(8 * hidden_channels,
                                      2 * hidden_channels,
                                      use_leaky_relu=0.0,
                                      use_dropout=0.0,
                                      use_bn=True,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1
                                      )
        self.expand5 = ExpandingBlock(4 * hidden_channels,
                                      hidden_channels,
                                      use_leaky_relu=0.0,
                                      use_dropout=0.0,
                                      use_bn=True,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1)
        self.downfeature = nn.Conv2d(2 * hidden_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.th = torch.nn.Tanh()
        self.switch = 0

    def forward(self, x):
        x = interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        if self.switch == 0:
            x0 = self.upfeature0(x)
        elif self.switch == 1:
            x0 = self.upfeature1(x)

        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand2(x4, x3)
        x6 = self.expand3(x5, x2)
        x7 = self.expand4(x6, x1)
        x8 = self.expand5(x7, x0)
        xn = self.downfeature(x8)
        return self.th(xn)
