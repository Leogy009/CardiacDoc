# 文件作用：定义用于生成rPPG信号的N3DED系列3D卷积网络
# 包含多个网络类，forward返回rPPG序列
# written by Yongzhi

import torch.nn as nn

#%% 3DED128 (baseline)

class N3DED128(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED128, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_244_244 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=128, Height=128]
        # 编码部分
        x = self.Conv1(x)		            # [b, F=3, T=128, W=128, H=128]->[b, F=16, T=128, W=128, H=128]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=16, T=128, W=128, H=128]->[b, F=16, T=64, W=32, H=32]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=32, H=32]->[b, F=32, T=64, W=32, H=32]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=32, T=64, W=32, H=32]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # 解码部分
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG

#%% 3DED64

class N3DED64(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED64, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_244_244 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        self.MaxpoolSpaTem_222_222 = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=64, Height=64]
        # ENCODER
        x = self.Conv1(x)		            # [b, F=3, T=128, W=64, H=64]->[b, F=16, T=128, W=64, H=64]
        x = self.MaxpoolSpaTem_222_222(x)   # [b, F=16, T=128, W=64, H=64]->[b, F=16, T=64, W=32, H=32]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=32, H=32]->[b, F=32, T=64, W=32, H=32]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=32, T=64, W=32, H=32]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG

#%% 3DED32

class N3DED32(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED32, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_244_244 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        self.MaxpoolTem_211_211 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=32, Height=32]
        # ENCODER
        x = self.Conv1(x)		            # [b, F=3, T=128, W=32, H=32]->[b, F=16, T=128, W=32, H=32]
        x = self.MaxpoolTem_211_211(x)   # [b, F=16, T=128, W=32, H=32]->[b, F=16, T=64, W=32, H=32]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=32, H=32]->[b, F=32, T=64, W=32, H=32]
        x = self.MaxpoolSpaTem_244_244(x)   # [b, F=32, T=64, W=32, H=32]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG

#%% 3DED16

class N3DED16(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED16, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.MaxpoolSpaTem_222_222 = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolTem_211_211 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=16, Height=16]
        # ENCODER
        x = self.Conv1(x)		            # [b, F=3, T=128, W=16, H=16]->[b, F=16, T=128, W=16, H=16]
        x = self.MaxpoolTem_211_211(x)   # [b, F=16, T=128, W=16, H=16]->[b, F=16, T=64, W=16, H=16]

        x = self.Conv2(x)		            # [b, F=16, T=64, W=16, H=16]->[b, F=32, T=64, W=16, H=16]
        x = self.MaxpoolSpaTem_222_222(x)   # [b, F=32, T=64, W=16, H=16]->[b, F=32, T=32, W=8, H=8]
        
        x = self.Conv3(x)		            # [b, F=32, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        x = self.Conv4(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=32, W=8, H=8]
        
        # DECODER
        x = self.TrConv1(x)		            # [b, F=64, T=32, W=8, H=8]->[b, F=64, T=64, W=8, H=8]
        x = self.TrConv2(x)		            # [b, F=64, T=64, W=8, H=8]->[b, F=64, T=128, W=8, H=8]        
        x = self.poolspa(x)                 # [b, F=64, T=128, W=8, H=8]->[b, F=64, T=128, W=1, H=1]
        x = self.ConvBlock5(x)             # [b, F=64, T=128, W=1, H=1]->[b, F=1, T=128, W=1, H=1]
        
        rPPG = x.view(-1,x.shape[2])        # [b,128]
        return rPPG


#%% 3DED8 (CardioCareRPPG) - 注意：3DED8、3DED4和3DED2结构相同。

class N3DED8(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED8, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # 只在空间维度上进行池化 
        self.poolspa = nn.AdaptiveAvgPool3d((128,1,1))

        
    def forward(self, x):	    	# x [6, T=128, 8, 8]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [6, T=128, 8, 8]->[16, T=128,  8, 8]
        x = self.MaxpoolTem(x)       # x [16, T=128, 8, 8]->[16, T=64, 8, 8]

        x = self.ConvBlock2(x)		     # x [16, T=64, 8, 8]->[32, T=64, 8, 8]
        x = self.MaxpoolTem(x)       # x [32, T=64, 8, 8]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG



#%% 3DED128_Enhanced 平滑通道增长 + 渐进式池化

class N3DED128_Enhanced(nn.Module):
    def __init__(self, frames=128):  
        super(N3DED128_Enhanced, self).__init__()

        
        self.Conv1_1 = nn.Sequential(
            nn.Conv3d(3, 8, [1,3,3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.Conv1_2 = nn.Sequential(
            nn.Conv3d(8, 16, [1,3,3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.Conv1_3 = nn.Sequential(
            nn.Conv3d(16, 32, [1,3,3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # 第二层：32 → 64 
        self.Conv2 = nn.Sequential(
            nn.Conv3d(32, 64, [1,5,5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # 第三层：64 → 128 
        self.Conv3 = nn.Sequential(
            nn.Conv3d(64, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # 第四层：128 → 256 
        self.Conv4 = nn.Sequential(
            nn.Conv3d(128, 256, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        
        # 第五层：256 → 512  
        self.Conv5 = nn.Sequential(
            nn.Conv3d(256, 512, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        
        # 时间维度池化 
        self.MaxpoolTem_211 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        
        # 空间维度池化 
        self.MaxpoolSpa_222 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        
        # 时空联合池化 
        self.MaxpoolSpaTem_222 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        
        # === 解码器 - 对称的通道数减少 ===
        self.TrConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.BatchNorm3d(256),
            nn.ELU(),
        )
        self.TrConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.BatchNorm3d(128),
            nn.ELU(),
        )
        self.TrConv3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.TrConv4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.BatchNorm3d(32),
            nn.ELU(),
        )
 
        self.ConvBlock5 = nn.Conv3d(32, 1, [1,1,1], stride=1, padding=0)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # [batch, Features=3, Temp=128, Width=128, Height=128]
        # === 编码部分 - 渐进式特征提取 ===
        
        # 第一层：渐进式通道增长 3 → 8 → 16 → 32
        x = self.Conv1_1(x)     # [b, 3, 128, 128, 128] → [b, 8, 128, 128, 128]
        x = self.Conv1_2(x)     # [b, 8, 128, 128, 128] → [b, 16, 128, 128, 128]
        x = self.Conv1_3(x)     # [b, 16, 128, 128, 128] → [b, 32, 128, 128, 128]
        x = self.MaxpoolTem_211(x)  # [b, 32, 128, 128, 128] → [b, 32, 64, 128, 128]

        # 第二层：32 → 64
        x = self.Conv2(x)       # [b, 32, 64, 128, 128] → [b, 64, 64, 128, 128]
        x = self.MaxpoolSpa_222(x)  # [b, 64, 64, 128, 128] → [b, 64, 64, 64, 64]

        # 第三层：64 → 128
        x = self.Conv3(x)       # [b, 64, 64, 64, 64] → [b, 128, 64, 64, 64]
        x = self.MaxpoolTem_211(x)  # [b, 128, 64, 64, 64] → [b, 128, 32, 64, 64]

        # 第四层：128 → 256
        x = self.Conv4(x)       # [b, 128, 32, 64, 64] → [b, 256, 32, 64, 64]
        x = self.MaxpoolSpaTem_222(x)  # [b, 256, 32, 64, 64] → [b, 256, 16, 32, 32]
        
        # 第五层：256 → 512  
        x = self.Conv5(x)       # [b, 256, 16, 32, 32] → [b, 512, 16, 32, 32]
        x = self.MaxpoolSpaTem_222(x)  # [b, 512, 16, 32, 32] → [b, 512, 8, 16, 16]
        
        # === 解码部分 - 对称的特征重建 ===
        x = self.TrConv1(x)     # [b, 512, 8, 16, 16] → [b, 256, 16, 16, 16]
        x = self.TrConv2(x)     # [b, 256, 16, 16, 16] → [b, 128, 32, 16, 16]
        x = self.TrConv3(x)     # [b, 128, 32, 16, 16] → [b, 64, 64, 16, 16]
        x = self.TrConv4(x)     # [b, 64, 64, 16, 16] → [b, 32, 128, 16, 16]
        
        x = self.poolspa(x)     # [b, 32, 128, 16, 16] → [b, 32, 128, 1, 1]
        x = self.ConvBlock5(x)  # [b, 32, 128, 1, 1] → [b, 1, 128, 1, 1]
        
        rPPG = x.view(-1, x.shape[2])  # [b, 128]
        return rPPG