from models.blocks import *


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake.
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = ContractingBlock(input_channels, hidden_channels, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 1, padding = 0)
        self.contract1 = ContractingBlock(hidden_channels, 2*hidden_channels, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 1, padding = 0)
        self.contract2 = ContractingBlock(2*hidden_channels, 4*hidden_channels, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 1, padding = 0)
        self.contract3 = ContractingBlock(4*hidden_channels, 8*hidden_channels, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 1, padding = 0)
        self.final = nn.Conv2d( 8*hidden_channels, 1, kernel_size=4, stride=1, padding=0)
        #self.sig = nn.Sigmoid()
        #self.rl = nn.ReLU()

    def forward(self, x):
        #print("disc forward:","x shape",x.shape,"y shape",y.shape)
        #x = torch.cat([y, x], axis=1) #ja ba jayi jaye x va y
        #print("disc forward:","new x shape",x.shape,"y shape",y.shape)
        #print("in disc",x.shape)
        x = interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        x0 = self.upfeature(x)
        #print("in disc",x0.shape)
        x1 = self.contract1(x0)
        #print("in disc",x1.shape)
        x2 = self.contract2(x1)
        #print("in disc",x2.shape)
        x3 = self.contract3(x2)
        x4 = self.final(x3)
        #x5 = self.sig(x4)
        x5 = x4.view(x4.size()[0], -1)
        return x5