from models.blocks import *

class Classifier(nn.Module):

    def __init__(self, input_channels, hidden_channels=64):
        super(Classifier, self).__init__()
        self.upfeature = ContractingBlock(input_channels, hidden_channels, use_dropout=0.0, use_bn=True ,use_maxp = False, kernel_size = 4 ,stride=1, padding=0)
        self.contract1 = ContractingBlock(hidden_channels, 2 * hidden_channels, use_dropout=0.5, use_bn=True,
                                          use_maxp=False, kernel_size=4, stride=1, padding=0)
        self.contract2 = ContractingBlock(2 * hidden_channels, 4 * hidden_channels, use_dropout=0.5, use_bn=True,
                                          use_maxp=False, kernel_size=4, stride=1, padding=0)
        self.contract3 = ContractingBlock(4 * hidden_channels, 8 * hidden_channels, use_dropout=0.0, use_bn=True,
                                          use_maxp=False, kernel_size=4, stride=1, padding=0)
        self.final = nn.Conv2d(8 * hidden_channels, 1, kernel_size=4, stride=1, padding=0)
        # self.final1 = nn.Conv2d(hidden_channels * 16, 4, kernel_size=1)
        self.fc = nn.Flatten()
        # self.btn1 = nn.BatchNorm1d(400, affine=False)
        # self.sig = nn.Sigmoid()
        # self.rl = nn.ReLU()

    def forward(self, x):
        # print("disc forward:","x shape",x.shape,"y shape",y.shape)
        # x = torch.cat([y, x], axis=1) #ja ba jayi jaye x va y
        # print("disc forward:","new x shape",x.shape,"y shape",y.shape)
        x = interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.final(x3)
        # print("x4",x4.shape)
        x5 = self.fc(x4)
        # print("in classification",x6.shape)
        # print("x5", x5.shape)
        # x6 = self.btn1(x5)
        # print("x6", x6.shape)
        # x7 = self.sig(x6)#self.rl(x7)
        # print("x7", x7.shape)
        return x5