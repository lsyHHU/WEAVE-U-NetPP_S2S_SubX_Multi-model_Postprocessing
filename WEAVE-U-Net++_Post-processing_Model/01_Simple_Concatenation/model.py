#%%
import torch
import torch.nn as nn
import torch.nn.functional as F



class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm = True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        if pre_Batch_Norm:
          self.Conv_forward = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
 
        else:
          self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ELU(inplace=True))
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x


class UnetPlusPlus(nn.Module):
    """
    模型输入: 
    273通道 (S2S+SubX multimodel total precip ensemble + atmospheric varibales emsmean)
    空间范围48*80

    模型输出:
    当前空间范围的CSGD分布三个参数的空间分布 (3通道)
    """
    
    def __init__(self, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        self.stage0 = ContinusParalleConv(273, 512, pre_Batch_Norm=False)
        self.stage1 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)
        self.stage2 = ContinusParalleConv(1024, 2048, pre_Batch_Norm=False)
        self.stage3 = ContinusParalleConv(2048, 4096, pre_Batch_Norm=False)

        self.CONV0_1 = ContinusParalleConv(512*2, 512, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(512*3, 512, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(512*4, 512, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(1024*2, 1024, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(1024*3, 1024, pre_Batch_Norm=True)

        self.CONV2_1 = ContinusParalleConv(2048*2, 2048, pre_Batch_Norm=True)

        self.pool = nn.MaxPool2d(2, 2)

        self.upsample_0_1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(4096, 2048, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ELU(inplace=True),
          nn.Conv2d(512, 3, 3, padding=1))
        
        self.final_super_0_2 = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ELU(inplace=True),
          nn.Conv2d(512, 3, 3, padding=1))
        
        self.final_super_0_3 = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ELU(inplace=True),
          nn.Conv2d(512, 3, 3, padding=1))
        

    def forward(self, x):
        x_0_0 = self.stage0(x)
        x_1_0 = self.stage1(self.pool(x_0_0))
        x_2_0 = self.stage2(self.pool(x_1_0))
        x_3_0 = self.stage3(self.pool(x_2_0))

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], dim=1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], dim=1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], dim=1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_1, x_0_0], dim=1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_1, x_1_0], dim=1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_2, x_0_1, x_0_0], dim=1)
        x_0_3 = self.CONV0_3(x_0_3)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            
            raw_shift1 = out_put1[:, 0, :, :]
            raw_mu1 = out_put1[:, 1, :, :]
            raw_sigma1 = out_put1[:, 2, :, :]
            shift1 = -torch.sqrt(torch.square(raw_shift1) + 1e-6)  # shift <= 0
            mu1 = F.softplus(raw_mu1)     # mu > 0
            sigma1 = F.softplus(raw_sigma1)   # sigma > 0

            raw_shift2 = out_put2[:, 0, :, :]
            raw_mu2 = out_put2[:, 1, :, :]
            raw_sigma2 = out_put2[:, 2, :, :]
            shift2 = -torch.sqrt(torch.square(raw_shift2) + 1e-6)  # shift <= 0
            mu2 = F.softplus(raw_mu2)     # mu > 0
            sigma2 = F.softplus(raw_sigma2)   # sigma > 0

            raw_shift3 = out_put3[:, 0, :, :]
            raw_mu3 = out_put3[:, 1, :, :]
            raw_sigma3 = out_put3[:, 2, :, :]
            shift3 = -torch.sqrt(torch.square(raw_shift3) + 1e-6)  # shift <= 0
            mu3 = F.softplus(raw_mu3)     # mu > 0
            sigma3 = F.softplus(raw_sigma3)   # sigma > 0

            return [
                torch.cat([shift1.unsqueeze(1), mu1.unsqueeze(1), sigma1.unsqueeze(1)], dim=1),
                torch.cat([shift2.unsqueeze(1), mu2.unsqueeze(1), sigma2.unsqueeze(1)], dim=1),
                torch.cat([shift3.unsqueeze(1), mu3.unsqueeze(1), sigma3.unsqueeze(1)], dim=1)
            ]

        else:
            out_put = self.final_super_0_3(x_0_3)
            raw_shift = out_put[:, 0, :, :]
            raw_mu = out_put[:, 1, :, :]
            raw_sigma = out_put[:, 2, :, :]
            shift = -torch.sqrt(torch.square(raw_shift) + 1e-6)  # shift <= 0
            mu = F.softplus(raw_mu)     # mu > 0
            sigma = F.softplus(raw_sigma)   # sigma > 0
            return torch.cat([shift.unsqueeze(1), mu.unsqueeze(1), sigma.unsqueeze(1)], dim=1)



    


# x = torch.randn((2, 273, 48, 40))
# model = UnetPlusPlus(True)
# shift, mu, sigma = model(x)











# %%
