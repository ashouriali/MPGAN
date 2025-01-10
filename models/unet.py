from models.blocks import *

class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(UNet, self).__init__()
        self.upfeature0 = ContractingBlock(input_channels, hidden_channels, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 1, padding = 0)
        self.upfeature1 = ContractingBlock(input_channels, hidden_channels, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 1, padding = 0)
        self.contract1 = ContractingBlock(hidden_channels, 2*hidden_channels, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 2, padding = 1)
        self.contract2 = ContractingBlock(2*hidden_channels, 4*hidden_channels, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 2, padding = 1)
        self.contract3 = ContractingBlock(4*hidden_channels, 8*hidden_channels, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 2, padding = 1)
        self.contract4 = ContractingBlock(8*hidden_channels, 8*hidden_channels, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 2, padding = 1)
        #self.contract5 = ContractingBlock(input_ch, out_ch, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 1, padding = 0)
        #self.contract6 = ContractingBlock(hidden_channels * 32,use_maxp=False)
        #self.expand0 = ExpandingBlock(hidden_channels * 64)
        #self.expand1 = ExpandingBlock(hidden_channels * 32)
        self.expand2 = ExpandingBlock( 8*hidden_channels, 8*hidden_channels, use_leakyRelu = 0.0, use_dropout=0.5, use_bn=True, kernel_size = 4 ,stride = 2, padding = 1)
        self.expand3 = ExpandingBlock( 16*hidden_channels,4*hidden_channels, use_leakyRelu = 0.0, use_dropout=0.5, use_bn=True, kernel_size = 4 ,stride = 2, padding = 1)
        self.expand4 = ExpandingBlock( 8*hidden_channels, 2*hidden_channels, use_leakyRelu = 0.0, use_dropout=0.0, use_bn=True, kernel_size = 4 ,stride = 2, padding = 1)
        self.expand5 = ExpandingBlock( 4*hidden_channels, hidden_channels, use_leakyRelu = 0.0, use_dropout=0.0, use_bn=True, kernel_size = 4 ,stride = 2, padding = 1)
        self.downfeature = nn.Conv2d( 2*hidden_channels, output_channels, kernel_size= 1, stride=1, padding=0)
        #self.sigmoid = torch.nn.Sigmoid()
        self.th = torch.nn.Tanh()
        self.switch = 0

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet:
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        #a[0][((a[1] == 0)[:,0]==1).tolist()].shape
        #print(x.shape)
        #xt0 = self.upfeature1(x[0][indi==0])
        #xt1 = self.upfeature2(x[0][indi==1])
        #print(x.shape,"###",xt0.shape,xt1.shape)
        #print(x.shape,"x")
        x = interpolate(x, size=(35, 35), mode='bilinear', align_corners=True)
        #print(x.shape,"x")
        if(self.switch == 0):
            x0 = self.upfeature0(x)
        elif(self.switch == 1):
            x0 = self.upfeature1(x)
        #elif(self.switch == 2):
        #    x0 = self.upfeature2(x)
        #elif(self.switch == 3):
        #    x0 = self.upfeature3(x)

        x1 = self.contract1(x0)
        #print(x1.shape,"x1_c")
        x2 = self.contract2(x1)
        #print(x2.shape,"x2_c")
        x3 = self.contract3(x2)
        #print(x3.shape,"x3_c")
        x4 = self.contract4(x3)
        #print(x4.shape,"x4_c")
        #x5 = self.contract5(x4)
        #print(x5.shape)
        #x6 = self.contract6(x5)
        #print(x6.shape)
        #x7 = self.expand0(x6, x5,1)
        #x8 = self.expand1(x5, x4,2)
        x9 = self.expand2(x4, x3)
        #print(x9.shape,"x9_e1")
        x10 = self.expand3(x9, x2)
        #print(x10.shape,"x10_e1")
        x11 = self.expand4(x10, x1)
        #print(x11.shape,"x11_e1")
        x12 = self.expand5(x11, x0)
        #print(x12.shape,"x12_e1")
        xn = self.downfeature(x12)
        #print(xn.shape,"f")
        #return self.sigmoid(xn)
        return self.th(xn)