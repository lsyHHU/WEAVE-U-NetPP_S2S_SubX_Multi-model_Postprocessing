#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


def sin_cos_encoding(lat, lon):
    """
    对经纬度序列进行正余弦编码
    输入: lat/lon: tensor (batch, height, width)
    输出形状不变 
    """
    return torch.sin(lat * torch.pi / 180.0), \
            torch.cos(lon * torch.pi / 180.0)


def dem_normalization(dem):
    """
    对DEM高程数据标准化0-1
    输入: dem: tensor (batch, height, width) 
    输出形状不变 
    """
    dem_min, dem_max = dem.min(), dem.max()
    return (dem - dem_min) / (dem_max - dem_min)


class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        if pre_Norm:
          self.Conv_forward = nn.Sequential(
            nn.GroupNorm(self.in_channels//4, self.in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.GroupNorm(self.out_channels//4, self.out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
 
        else:
          self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.GroupNorm(self.out_channels//4, self.out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.GroupNorm(self.out_channels//4, self.out_channels),
            nn.ELU(inplace=True))
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x


class UnetPlusPlus(nn.Module):
    """
    模型输入: 
    273通道 (S2S+SubX multimodel total precip ensemble + atmospheric varibales emsmean)
    3通道 geo features
    空间范围48*80

    不同模型的输入为加权和（固定权重）
    一共18个模型，每个模型的权重为1/18

    模型输出:
    当前空间范围的CSGD分布三个参数的空间分布 (3通道)
    """
    
    def __init__(self, in_ch_list, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        # Encoder for each model
        self.encoder_eachModel_1 = nn.ModuleList(
            [ContinusParalleConv(in_ch_list[i], 128, pre_Norm=False) for i in range(len(in_ch_list))]
        )  # n * (B, 64, 48, 80)

        self.encoder_eachModel_2 = nn.ModuleList(
            [ContinusParalleConv(in_ch_list[i], 256, pre_Norm=False) for i in range(len(in_ch_list))]
        )  # n * (B, 128, 24, 40)

        self.encoder_eachModel_3 = nn.ModuleList(
            [ContinusParalleConv(in_ch_list[i], 512, pre_Norm=False) for i in range(len(in_ch_list))]
        )  # n * (B, 256, 12, 20)

        self.encoder_eachModel_4 = nn.ModuleList(
            [ContinusParalleConv(in_ch_list[i], 1024, pre_Norm=False) for i in range(len(in_ch_list))]
        )  # n * (B, 512, 6, 10)

        self.avgpool2 = nn.AvgPool2d(2, 2)
        self.avgpool3 = nn.AvgPool2d(4, 4)
        self.avgpool4 = nn.AvgPool2d(8, 8)

        # Unet++ backbone 公共下采样
        # self.stage0 = ContinusParalleConv(128, 512, pre_Norm=False)
        self.stage1 = ContinusParalleConv(128, 256, pre_Norm=False)
        self.stage2 = ContinusParalleConv(256, 512, pre_Norm=False)
        self.stage3 = ContinusParalleConv(512, 1024, pre_Norm=False)

        self.CONV0_1 = ContinusParalleConv(128*2, 128, pre_Norm=True)
        self.CONV0_2 = ContinusParalleConv(128*3, 128, pre_Norm=True)
        self.CONV0_3 = ContinusParalleConv(128*4, 128, pre_Norm=True)

        self.CONV1_1 = ContinusParalleConv(256*2, 256, pre_Norm=True)
        self.CONV1_2 = ContinusParalleConv(256*3, 256, pre_Norm=True)

        self.CONV2_1 = ContinusParalleConv(512*2, 512, pre_Norm=True)

        self.pool = nn.MaxPool2d(2, 2)

        self.upsample_0_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
          nn.GroupNorm(128//4, 128),
          nn.ELU(inplace=True),
          nn.Conv2d(128, 3, 3, padding=1))
        
        self.final_super_0_2 = nn.Sequential(
          nn.GroupNorm(128//4, 128),
          nn.ELU(inplace=True),
          nn.Conv2d(128, 3, 3, padding=1))
        
        self.final_super_0_3 = nn.Sequential(
          nn.GroupNorm(128//4, 128),
          nn.ELU(inplace=True),
          nn.Conv2d(128, 3, 3, padding=1))
        

    def forward(self, fcst, geo):
        # Geo data preprocessing
        dem_embed = dem_normalization(geo[:, 0, :, :])   # (b, 1, 48, 80) -> (b, 48, 80)
        lat_embed, lon_embed = sin_cos_encoding(geo[:, 1, :, :], geo[:, 2, :, :])
        # Concatenate all geo data
        geo_embed = torch.cat([
            lat_embed.unsqueeze(1),
            lon_embed.unsqueeze(1),
            dem_embed.unsqueeze(1),
        ], dim=1)       # (b, 3, 48, 80)

        # fcst分开降水和环流变量
        pre_start_idx = [0, 13, 21, 26, 36, 53, 59, 67, 110, 124, 144, 165, 179, 200, 219, 231, 245, 257]
        pre_end_idx = [10, 16, 25, 30, 47, 54, 63, 100, 114, 134, 155, 169, 190, 210, 223, 239, 248, 264] 
        atmos_end_idx = [13, 21, 26, 36, 53, 59, 67, 110, 124, 144, 165, 179, 200, 219, 231, 245, 257, 272]

        pre = []
        atmos = []
        for i in range(18):
            pre.append(fcst[:, pre_start_idx[i]:pre_end_idx[i], :, :])
            atmos.append(torch.cat([fcst[:, pre_end_idx[i]:atmos_end_idx[i], :, :], geo_embed], dim=1))

        # Fixed weight for each model
        w = torch.tensor(1/18, dtype=torch.float32, device=fcst.device)
        weight = w.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        weight = weight.repeat(fcst.size(0), 18, fcst.size(2), fcst.size(3))  # (B, 18, 48, 80) 

        # 模型各自下采样
        x1 = [self.encoder_eachModel_1[i](pre[i]) for i in range(18)]  # n * (B, 64, 48, 80)
        x1 = torch.stack(x1, dim=1)
        x1_w = torch.einsum('bnchw,bnhw->bchw', x1, weight)  # (B, 64, 48, 80)

        x2 = [self.encoder_eachModel_2[i](self.avgpool2(pre[i])) for i in range(18)]  # n * (B, 128, 24, 40)
        x2 = torch.stack(x2, dim=1)
        x2_w = torch.einsum('bnchw,bnhw->bchw', x2, self.avgpool2(weight))  # (B, 128, 24, 40)

        x3 = [self.encoder_eachModel_3[i](self.avgpool3(pre[i])) for i in range(18)]  # n * (B, 256, 12, 20)
        x3 = torch.stack(x3, dim=1)
        x3_w = torch.einsum('bnchw,bnhw->bchw', x3, self.avgpool3(weight))  # (B, 256, 12, 20)

        x4 = [self.encoder_eachModel_4[i](self.avgpool4(pre[i])) for i in range(18)]  # n * (B, 512, 6, 10)
        x4 = torch.stack(x4, dim=1)
        x4_w = torch.einsum('bnchw,bnhw->bchw', x4, self.avgpool4(weight))  # (B, 512, 6, 10)

        # 公共下采样
        x_0_0 = x1_w      # (B, 64, 48, 80)
        x_1_0 = self.stage1(self.pool(x_0_0)) + x2_w  # (B, 128, 24, 40)
        x_2_0 = self.stage2(self.pool(x_1_0)) + x3_w  # (B, 256, 12, 20)
        x_3_0 = self.stage3(self.pool(x_2_0)) + x4_w  # (B, 512, 6, 10)

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], dim=1) # (B, 64*2, 48, 80)
        x_0_1 = self.CONV0_1(x_0_1)   # (B, 64, 48, 80)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], dim=1) # (B, 128*2, 24, 40)
        x_1_1 = self.CONV1_1(x_1_1)   # (B, 128, 24, 40)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], dim=1) # (B, 256*2, 12, 20)
        x_2_1 = self.CONV2_1(x_2_1)   # (B, 256, 12, 20)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_1, x_0_0], dim=1)  # (B, 64*3, 48, 80)
        x_0_2 = self.CONV0_2(x_0_2)   # (B, 64, 48, 80)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_1, x_1_0], dim=1)  # (B, 128*3, 24, 40)
        x_1_2 = self.CONV1_2(x_1_2)   # (B, 128, 24, 40)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_2, x_0_1, x_0_0], dim=1) # (B, 64*4, 48, 80)
        x_0_3 = self.CONV0_3(x_0_3)   # (B, 64, 48, 80)

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



    


# fcst = torch.randn((2, 273, 48, 80))
# geo = torch.randn(2, 3, 48, 80)


# in_ch_list = [10, 3, 4, 4, 11, 1, 4, 33, 4, 10, 11, 4, 11, 10, 4, 8, 3, 7]
# model = UnetPlusPlus(in_ch_list, False)
# print(model(fcst, geo).shape)










# %%
