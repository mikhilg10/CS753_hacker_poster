class DoubleConv(nn.Module):
  def __init__(self,in_chan,out_chan):
    super(DoubleConv,self).__init__()
    self.conv  =  nn.Sequential(
        nn.Conv2d(in_chan,out_chan,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_chan),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chan,out_chan,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_chan),
        nn.ReLU(inplace=True)
    )
  def forward(self,x):
    x = self.conv(x)
    return x
    
class UNetDepth(nn.Module):
  def __init__(self,feature = 16*[32,64,128,256,512]):
    super(UNetDepth,self).__init__()
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.enc1 = DoubleConv(3,feature[0])
    self.enc2 = DoubleConv(feature[0],feature[1])
    self.enc3 = DoubleConv(feature[1],feature[2])
    self.enc4 = DoubleConv(feature[2],feature[3])
    self.enc5 = DoubleConv(feature[3],feature[4])
    self.upsample1 = nn.ConvTranspose2d(in_channels=feature[4],out_channels=feature[3],kernel_size=2,stride=2)
    self.upsample2 = nn.ConvTranspose2d(in_channels=feature[3],out_channels=feature[2],kernel_size=2,stride=2)
    self.upsample3 = nn.ConvTranspose2d(in_channels=feature[2],out_channels=feature[1],kernel_size=2,stride=2)
    self.upsample4 = nn.ConvTranspose2d(in_channels=feature[1],out_channels=feature[0],kernel_size=2,stride=2)
    self.dec1 = DoubleConv(feature[4],feature[3])
    self.dec2 = DoubleConv(feature[3],feature[2])
    self.dec3 = DoubleConv(feature[2],feature[1])
    self.dec4 = DoubleConv(feature[1],feature[0])
    self.out  = nn.Conv2d(feature[0],1,kernel_size=1)
    self.drop = nn.Dropout2d(p=0.2)

  def forward(self,x):
    #encoder
                          #inchannel 3 in_image = 256 256
    x1  = self.enc1(x)    #inchannel=3 outchannel=32
    x1  = self.drop(x1)
    x2  = self.pool(x1)   # 32 128 128
    x2  = self.drop(x2)   ##########$################
    x3  = self.enc2(x2)   # inchannel=32 outchannel=64
    x4  = self.pool(x3)   # 64 64 64 
    x4  = self.drop(x4)   #######################
    x5  = self.enc3(x4)   # inchannel=64 outchannel=128
    x6  = self.pool(x5)   #128 32 32
    x7  = self.enc4(x6)   #inchannel=128 outchannel=256
    x8  = self.pool(x7)   #256 16 16
    x9  = self.enc5(x8)   #inchannel=256 outchannel=512
    #print("x9",x9.shape)
    #decoder
    x10 = torch.cat([self.upsample1(x9),x7],dim=1)
    #print("x10", x10.shape)
    x11 = self.dec1(x10)
    #print("x11", x11.shape)
    x12 = torch.cat([self.upsample2(x11),x5],dim=1)
    #print("x12", x12.shape)
    x13 = self.dec2(x12)
    x13 = self.drop(x13)  ###############
    #print("x13", x13.shape)
    x14 = torch.cat([self.upsample3(x13),x3],dim=1)
    x15 = self.dec3(x14)
    #print("x15", x15.shape)
    x16 = torch.cat([self.upsample4(x15),x1],dim=1)
#     x16 = self.drop(x16)   ################
    x17 = self.dec4(x16)
    #print("x17", x17.shape)
    out = self.out(x17)
    #print("out", out.shape)
    return F.relu(out)

net = UNetDepth().to(device)
model = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
