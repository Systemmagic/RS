import os
import glob
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2  # 用于 resize 图片
# --- 1 ---
class DeepKoopmanModel(nn.Module):
    def __init__(self, input_h=50, input_w=50):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.feature_h = input_h // 4
        self.feature_w = input_w // 4
        self.flatten_dim = 8 * self.feature_h * self.feature_w
        
        self.koopman_matrix = nn.Linear(self.flatten_dim, self.flatten_dim, bias=False)
        
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x):
        """只负责编码，提取特征 z"""
        z_spatial = self.encoder_cnn(x)
        z_vector = z_spatial.view(x.size(0), -1)
        return z_vector

    def forward(self, x):
        # 1. 编码
        z_t = self.encode(x)
        
        # 2. 演化 (预测 z_{t+1})
        z_next_pred = self.koopman_matrix(z_t)
        
        # 3. 解码 (重建 x_t)
        z_spatial_recon = z_t.view(x.size(0), 8, self.feature_h, self.feature_w)
        x_recon = self.decoder_cnn(z_spatial_recon)
        
        # 4. 解码 (预测 x_{t+1})
        z_spatial_pred = z_next_pred.view(x.size(0), 8, self.feature_h, self.feature_w)
        x_pred = self.decoder_cnn(z_spatial_pred)
        
        # 返回所有关键变量以便计算 Loss
        return x_recon, x_pred, z_t, z_next_pred
    def predict_future(self, x_start, steps=7):
        """
        predict_future 的 Docstring
        利用学到的线性动力学 K，进行长期预测
        x_start: 起始时刻的地图 (Batch, 1, H, W)
        steps: 预测未来几步
        返回: 一个列表，包含未来 steps 步的预测地图
        
        """
        # 1. 先把当前地图编码到 latent 空间
        z_current = self.encode(x_start)
        
        predictions = []
        
        # 2. 在 latent 空间里迭代演化
        for _ in range(steps):
            # 核心公式: z_{t+1} = K * z_t
            z_next = self.koopman_matrix(z_current)
            
            # 把这一步的 z 解码成地图存起来
            # 注意：这里需要先把 z 变回 (B, C, H, W) 的形状才能进 decoder
            z_spatial = z_next.view(x_start.size(0), 8, self.feature_h, self.feature_w)
            x_future = self.decoder_cnn(z_spatial)
            predictions.append(x_future)
            
            # 更新状态，准备下一步
            z_current = z_next
            
        return predictions

class PM25TIFFDataset(Dataset):
    def __init__(self, data_dir, img_size=64):
        """
        data_dir: 存放 .tif 文件的文件夹路径
        img_size: 强制缩放到的尺寸 (比如 64x64)
        """
        self.img_size = img_size
        
        # 1. 找到所有 tif 文件并排序
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        if len(self.file_paths) == 0:
            raise ValueError(f"在 {data_dir} 没找到 .tif 文件！请检查路径。")
            
        print(f"找到 {len(self.file_paths)} 张 TIFF 文件。正在加载...")
        
        # 2. 一次性把所有图读进内存 (如果数据量巨大，建议改为按需读取)
        self.data = []
        for path in self.file_paths:
            img = tifffile.imread(path)
            
            # 处理 NaN: 把 NaN 变成 0
            img = np.nan_to_num(img, nan=0.0)
            
            # 缩放尺寸 (Resize) -> 64x64
            img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            
            # 归一化 (Normalization): 把数值缩放到 [0, 1] 之间
            # 这对神经网络收敛非常重要！
            img_max = img_resized.max()
            img_min = img_resized.min()
            if img_max - img_min > 0:
                img_norm = (img_resized - img_min) / (img_max - img_min)
            else:
                img_norm = img_resized # 全是0的情况
                
            self.data.append(img_norm)
            
        self.data = np.array(self.data) # Shape: (Total_Days, 64, 64)
        
    def __len__(self):
        # 样本数 = 总天数 - 1 (因为最后一天没有"下一天"做标签)
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 输入: 今天 (Day t)
        x = self.data[idx]
        # 目标: 明天 (Day t+1)
        y = self.data[idx + 1]
        
        # 转为 Tensor: (Channel, H, W)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        
        return x_tensor, y_tensor

# --- 使用示例 ---
# 请把这里换成你真实的文件夹路径！
# 注意：Windows路径要用双反斜杠 \\ 或者在字符串前加 r
real_data_dir = r"D:\workspace\RS\CZT_PM25_Daily" 

# 初始化数据集 (强制缩放到 64x64，适配模型)
# 如果你没有真实数据，这段代码会报错。请确保文件夹里有 tif。
try:
    dataset = PM25TIFFDataset(real_data_dir, img_size=64) # 使用 64 以防万一
    real_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print("真实数据集加载成功！")
    
    # 重新初始化模型 (输入尺寸改为 64)
    model = DeepKoopmanModel(input_h=64, input_w=64)
    
    # 剩下的训练代码和之前一模一样...
    
except Exception as e:
    print(f"出错啦: {e}")
    print("请检查 data_dir 路径是否正确，或者是否安装了 tifffile 库。")
