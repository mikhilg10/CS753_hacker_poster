
class ChanAttn(nn.Module):
  def __init__(self,num_chan,reduction=2):
    super(ChanAttn,self).__init__()
    self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
    self.fc1 = nn.Sequential(nn.Linear(in_features=num_chan, out_features=num_chan//2),
                             nn.BatchNorm1d(num_chan//2),
                             nn.ReLU())
    self.fc2 = nn.Sequential(nn.Linear(in_features=num_chan//2, out_features=num_chan),
                             nn.BatchNorm1d(num_chan),
                             nn.ReLU())
  def forward(self,x):
    x = self.globalAvgPool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    x = x.view(x.size(0), x.size(1), 1, 1)
    return x

class SpatAttn(nn.Module):
  def __init__(self,num_chan,reduction=2):
    super(SpatAttn,self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(num_chan, num_chan//reduction, kernel_size=1, padding=0),
                               nn.BatchNorm2d(num_chan//reduction),
                               nn.ReLU()
                               )
    self.conv2 = nn.Sequential(nn.Conv2d(num_chan//reduction,num_chan//reduction, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                               nn.BatchNorm2d(num_chan//reduction),
                               nn.ReLU()
                               )
    self.conv3 = nn.Sequential(nn.Conv2d(num_chan//reduction, 1, kernel_size=1, padding=0),
                               nn.BatchNorm2d(1),
                               nn.ReLU()
                               )
  def forward(self,x):
    x = self.conv3(self.conv2(self.conv1(x)))
    return x

class BAM(nn.Module):
  def __init__(self,num_chan,reduction=2):
    super(BAM,self).__init__()
    self.M_s = SpatAttn(num_chan)
    self.M_c = ChanAttn(num_chan)
    # self.activation = nn.Sigmoid()
  def forward(self,x):
    att = 1 + F.sigmoid(self.M_s(x)+self.M_c(x))
    return x*att

class ResNet(nn.Module):

    def __init__(self, block, layers, input_channel=3, num_classes=1000, features=64):
        self.inplanes = features
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, features, layers[0])
        self.at1 = BAM(64) #BAM1 attention_layer_1
        self.layer2 = self._make_layer(block, features*2, layers[1], stride=2)
        self.at2 = BAM(128) #BAM2 attention_layer_2
        self.layer3 = self._make_layer(block, features*4, layers[2], stride=2)
        self.at3 = BAM(256) ##BAM3 attention_layer_3
        self.layer4 = self._make_layer(block, features*8, layers[3], stride=2)
        self.at4 = BAM(512) ##BAM4 attention_layer_4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features*8*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.at1(x)
        x = self.at2(self.layer2(x))
        x = self.at3(self.layer3(x))
        x = self.at4(self.layer4(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
