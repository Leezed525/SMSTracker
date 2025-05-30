import torch.nn as nn


def auto_pad(k, p=None, dilation=1):
    """
    Auto padding
    自动填充卷积层的padding
    :param k:(int) kernel size
    :param p:(int) padding
    :param dilation:(int) dilation
    :return:
    """
    if dilation > 1:
        k = dilation * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding=None, dilation=1, groups=1, bias=False,
                 activation=nn.SiLU()):
        """
        Convolution 卷积层，不指定padding时自动padding(保持图像大小不变)
        :param cin: (int) 输入通道
        :param cout: (int) 输出通道
        :param kernel_size: (int) 卷积核大小
        :param stride: (int) 步长
        :param padding: (int) 填充
        :param dilation: (int) dilation
        :param groups: (int) groups
        :param bias: (bool) bias
        :param activation: (nn.Module) 激活函数
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, auto_pad(kernel_size, padding, dilation) if padding is None else padding, dilation,
                              groups, bias)
        self.bn = nn.BatchNorm2d(cout)

        # bn层方差减半有助于收敛
        nn.init.constant_(self.bn.weight, 0.5)
        nn.init.zeros_(self.bn.bias)

        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, cin, expansion=1, bias=False):
        """
        ResBlock
        :param cin: (int) 输入通道
        :param expansion: (float) 隐藏层膨胀系数
        """
        super(ResBlock, self).__init__()
        c_ = int(cin * expansion)
        self.cv1 = Conv(cin, c_, 1, 1, bias=bias)
        self.cv2 = Conv(c_, cin, 3, 1, bias=bias)

        # 初始化卷积层参数
        nn.init.kaiming_normal(self.cv1.weight)
        nn.init.kaiming_normal(self.cv2.weight)

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        return x + out


class ResN(nn.Module):
    def __init__(self, cin, n=1):
        """
        ResN N个ResBlock
        :param cin: 输入通道
        :param n:  ResBlock的数量
        """
        super(ResN, self).__init__()
        self.conv = Conv(cin, cin, 3, 1)
        self.m = nn.Sequential(*[ResBlock(cin) for _ in range(n)])

    def forward(self, x):
        out = self.conv(x)
        out = self.m(out)
        return out
