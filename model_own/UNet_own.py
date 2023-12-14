import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=in_channels // out_channels, mode='bilinear')
        self.up_conv = nn.Conv2d(in_channels, out_channels, 2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_cat, x):
        x = self.up_conv(self.upsample(x))
        # Concatenation
        x_cat_reshaped = torch.zeros(x.size())
        x_cat_reshaped[:,:,:,:] = x_cat[:,:,
                                 (x_cat.size(dim=2)//2 - x.size(dim=2)//2):(x_cat.size(dim=2)//2 + x.size(dim=2)//2) + 1,
                                 (x_cat.size(dim=3)//2 - x.size(dim=3)//2):(x_cat.size(dim=3)//2 + x.size(dim=3)//2) + 1]
        x = torch.cat((x_cat_reshaped, x), 1)
        x = self.double_conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x_no_pool = self.double_conv(x)
        x = self.pool(x_no_pool)
        return x_no_pool, x

class UNet(nn.Module):
    def __init__(self, input_dim=1):
        super(UNet, self).__init__()
        self.down1 = Down(input_dim, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.bottom_conv = DoubleConv(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x_down1_np, x_down1 = self.down1(x)
        x_down2_np, x_down2 = self.down2(x_down1)
        x_down3_np, x_down3 = self.down3(x_down2)
        x_down4_np, x_down4 = self.down4(x_down3)

        x_bottom = self.bottom_conv(x_down4)

        x_up1 = self.up1(x_down4_np, x_bottom)
        x_up2 = self.up2(x_down3_np, x_up1)
        x_up3 = self.up3(x_down2_np, x_up2)
        x_up4 = self.up4(x_down1_np, x_up3)

        x_final = self.final_conv(x_up4)
        # x_final = x_final[:, 0, :, :] > x_final[:, 1, :, :]
        # x_final = x_final.unsqueeze(1).to(torch.float)
        transform = transforms.Resize((x.size(2), x.size(3)), interpolation=transforms.InterpolationMode.NEAREST, antialias=False)

        return F.sigmoid(transform(x_final))