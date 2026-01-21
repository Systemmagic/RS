# 导入所有必要的库
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from osgeo import gdal 

# ==============================================================================
# 0. 全局配置与工具函数
# ==============================================================================
class Config:
    """
    中央配置类 - 已针对效率和物理控制进行优化
    """
    # 数据参数
    IMG_SIZE = 256           
    SEQUENCE_LENGTH = 4      
    VALID_DATA = 0.001 
    
    # [修改 1] 开关：是否使用气象控制变量
    USE_METEO_CONTROL = True  # <--- 设置为 False 即可关闭风场控制

    # 气象控制参数
    METEO_DIM = 2            # 气象数据维度 (例如: U-wind, V-wind)
    # 架构参数 (轻量化)
    LATENT_DIM = 2048         # 从2048降至512
    
    # 训练参数
    BATCH_SIZE = 4           
    EPOCHS = 150             
    LEARNING_RATE = 1e-4     
    # --- 损失函数权重 ---
    W_BASE_L1 = 1.0          
    W_GRAD = 2.0             
    W_TOPK = 5.0             
    TOPK_RATIO = 0.15 
    
    # --- 动力学参数 ---
    W_SPECTRAL = 0.005
    EIGEN_MAX = 2.0          
    EIGEN_MIN = 0.8          

# 确保实验可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. 增强版数据集 (支持气象数据输入)
# ==============================================================================
class EnhancedPM25Dataset(Dataset):
    def __init__(self, data_dir, config, augment=False):    
        self.config = config
        self.augment = augment
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        if not self.file_paths: raise FileNotFoundError(f"No .tif found in {data_dir}")
        
        print(f"---Dataset: {len(self.file_paths)} files. Res={config.IMG_SIZE}, Control={config.USE_METEO_CONTROL}---")
        
        # 读取地理参考信息
        ref_ds = gdal.Open(self.file_paths[0])
        self.orig_w, self.orig_h = ref_ds.RasterXSize, ref_ds.RasterYSize
        self.geo_transform = ref_ds.GetGeoTransform()
        self.projection = ref_ds.GetProjection()
        
        # 加载图像数据
        self.data = self._load_data()
        self.global_min, self.global_max = np.min(self.data), np.max(self.data)
        print(f"Data Loaded. Range: [{self.global_min:.2f}, {self.global_max:.2f}]")
        
        # [修改 2] 根据开关生成气象数据
        if self.config.USE_METEO_CONTROL:
            # 启用: 使用正弦波模拟周期性风场 (或读取真实CSV)
            t = np.linspace(0, 4*np.pi, len(self.data))
            self.meteo_data = np.stack([np.sin(t), np.cos(t)], axis=1).astype(np.float32)
        else:
            # 禁用: 生成全0数据作为占位符，保持数据格式一致
            self.meteo_data = np.zeros((len(self.data), self.config.METEO_DIM), dtype=np.float32)

    def _load_data(self):
        data = []
        max_side = max(self.orig_w, self.orig_h)
        for p in self.file_paths:
            ds = gdal.Open(p)
            arr = ds.GetRasterBand(1).ReadAsArray()
            # Padding
            canvas = np.zeros((max_side, max_side), dtype=np.float32)
            y_off, x_off = (max_side - self.orig_h) // 2, (max_side - self.orig_w) // 2
            canvas[y_off:y_off+self.orig_h, x_off:x_off+self.orig_w] = arr
            # Resize
            resized = cv2.resize(canvas, (self.config.IMG_SIZE, self.config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            data.append(resized)
        data = np.array(data, dtype=np.float32)
        return np.nan_to_num(data, nan=0.0)

    def __len__(self):
        return len(self.data) - self.config.SEQUENCE_LENGTH

    def __getitem__(self, idx):
        # 1. 获取图像序列
        seq = self.data[idx : idx + self.config.SEQUENCE_LENGTH]
        if self.global_max > self.global_min:
            seq = (seq - self.global_min) / (self.global_max - self.global_min)
        
        # 2. 获取气象序列 (如果是False，这里取到的就是全0)
        meteo_seq = self.meteo_data[idx : idx + self.config.SEQUENCE_LENGTH]
        
        # 数据增强
        if self.augment:
            if random.random() > 0.5: seq = np.flip(seq, axis=2).copy()
            if random.random() > 0.5: seq = np.flip(seq, axis=1).copy()
            if random.random() > 0.5: seq = np.rot90(seq, k=1, axes=(1,2)).copy()
            
        # 转Tensor
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(1) # (T, 1, H, W)
        meteo_t = torch.tensor(meteo_seq, dtype=torch.float32)      # (T, 2)
        mask_t = (seq_t > Config.VALID_DATA).float()
        
        return seq_t, meteo_t, mask_t

    def get_all_data(self):
        if self.global_max > self.global_min:
             return (self.data - self.global_min) / (self.global_max - self.global_min)
        return self.data
        
# ==============================================================================
# 2. 模型架构 (ResNet + Koopman with Control)
# ==============================================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = torch.sigmoid(self.conv(x_cat))
        return x * attention_map

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = ResidualBlock(32, 32, 1)
        self.layer2 = ResidualBlock(32, 64, 2)
        self.layer3 = ResidualBlock(64, 128, 2)
        self.layer4 = ResidualBlock(128, 256, 2)
        self.att = SpatialAttention()
        final_h = config.IMG_SIZE // 32 
        self.flat_dim = 256 * final_h * final_h 
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, config.LATENT_DIM),
            nn.LayerNorm(config.LATENT_DIM),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.att(x)
        z = self.fc(x)
        return z

class ResNetDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.init_h = config.IMG_SIZE // 32
        self.flat_dim = 256 * self.init_h * self.init_h
        
        self.fc = nn.Linear(config.LATENT_DIM, self.flat_dim)
        self.up_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.init_h, self.init_h)
        x = self.up_layers(x)
        return self.final_conv(x)

class ControlledKoopmanModel(nn.Module):
    """
    [修改] 引入外部控制变量的 Koopman 模型
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = ResNetEncoder(config)
        self.decoder = ResNetDecoder(config)
        
        # 内部演化矩阵 K
        self.K = nn.Linear(config.LATENT_DIM, config.LATENT_DIM, bias=False)
        
        # [修改 3] 控制矩阵 B: 仅当 USE_METEO_CONTROL 为 True 时初始化
        if config.USE_METEO_CONTROL:
            self.B = nn.Linear(config.METEO_DIM, config.LATENT_DIM, bias=False)
        else:
            self.B = None # 标记为 None

    # [新增] 辅助函数，处理有无控制的情况
    def compute_control(self, u_current):
        if self.B is not None:
            return self.B(u_current)
        return 0 # 如果没启用控制，控制项为0

    def forward(self, x_current, u_current):
        z_t = self.encoder(x_current)
        # 动力学演化: K*z + (B*u or 0)
        z_next_pred = self.K(z_t) + self.compute_control(u_current)
        delta = self.decoder(z_next_pred)
        x_next_pred = torch.relu(x_current + delta)
        return x_next_pred, z_t, z_next_pred

    def predict_future(self, x_start, u_sequence, steps):
        preds = []
        x_curr = x_start
        z_curr = self.encoder(x_curr)
        
        for t in range(steps):
            u_t = u_sequence[t].unsqueeze(0).to(x_start.device) # (1, 2)
            
            # 演化
            z_curr = self.K(z_curr) + self.compute_control(u_t)
            
            # 解码
            delta = self.decoder(z_curr)
            x_next = torch.relu(x_curr + delta)
            
            preds.append(x_next)
            x_curr = x_next
            
        return preds

# ==============================================================================
# 3. 损失函数 (保持 SharpnessAwareLoss)
# ==============================================================================
class SharpnessAwareLoss(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.l1 = nn.L1Loss(reduction='none')

    def gradient_loss(self, pred, target, mask):
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        loss_dy = torch.abs(dy_pred - dy_target) * mask_y
        loss_dx = torch.abs(dx_pred - dx_target) * mask_x
        
        return (loss_dy.sum() + loss_dx.sum()) / (mask_y.sum() + mask_x.sum() + 1e-8)

    def forward(self, pred, target, mask):
        abs_diff = self.l1(pred, target) * mask
        loss_base = abs_diff.sum() / (mask.sum() + 1e-8)
        
        valid_pixels = abs_diff[mask > 0.5]
        if valid_pixels.numel() > 0:
            k = int(valid_pixels.numel() * self.config.TOPK_RATIO)
            k = max(k, 1)
            topk_vals, _ = torch.topk(valid_pixels, k)
            loss_topk = topk_vals.mean()
        else:
            loss_topk = 0.0
            
        loss_grad = self.gradient_loss(pred, target, mask)
        
        return (self.config.W_BASE_L1 * loss_base + 
                self.config.W_TOPK * loss_topk + 
                self.config.W_GRAD * loss_grad)

# ==============================================================================
# 4. 评估与导出模块 (适配新接口)
# ==============================================================================
def export_geotiff(filename, data_array, dataset_obj):
    max_side = max(dataset_obj.orig_w, dataset_obj.orig_h)
    canvas = cv2.resize(data_array, (max_side, max_side), interpolation=cv2.INTER_CUBIC)
    
    y_offset = (max_side - dataset_obj.orig_h) // 2
    x_offset = (max_side - dataset_obj.orig_w) // 2
    actual_img = canvas[y_offset:y_offset+dataset_obj.orig_h, x_offset:x_offset+dataset_obj.orig_w]
    actual_img = actual_img * (dataset_obj.global_max - dataset_obj.global_min) + dataset_obj.global_min
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(filename, dataset_obj.orig_w, dataset_obj.orig_h, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(dataset_obj.geo_transform)
    out_ds.SetProjection(dataset_obj.projection)
    out_ds.GetRasterBand(1).WriteArray(actual_img)
    out_ds.FlushCache()
    out_ds = None

def predict_and_export_n_days(model, dataset, device, output_dir, start_day_idx=0, n_days=7):
    print(f"--- 正在预测未来 {n_days} 天(从索引 {start_day_idx}开始) ---")
    model.eval()
    
    # 1. 准备启动图像
    x_current_raw = dataset.get_all_data()[start_day_idx]
    x_current_norm = np.nan_to_num(x_current_raw, nan=0.0)
    x_current_tensor = torch.tensor(x_current_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 2. 准备未来的气象预报数据
    future_meteo = dataset.meteo_data[start_day_idx+1 : start_day_idx+1+n_days]
    if len(future_meteo) < n_days:
        pad = np.tile(dataset.meteo_data[-1], (n_days - len(future_meteo), 1))
        future_meteo = np.vstack([future_meteo, pad]) if len(future_meteo) > 0 else pad
        
    u_sequence = torch.tensor(future_meteo, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_list = model.predict_future(x_current_tensor, u_sequence, steps=n_days)
        
        for i, pred_tensor in enumerate(preds_list):
            day = i + 1
            pred_img = pred_tensor.squeeze().cpu().numpy()
            filename = os.path.join(output_dir, f"Pred_Day_{day:02d}.tif")
            export_geotiff(filename, pred_img, dataset)
            print(f"  ✓ 已保存: {filename}")

# ==============================================================================
# 5. 主程序：训练与评估
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 初始化 ---
    config = Config()
    # 假设图片在 "CZT_PM25_Daily" 文件夹
    if not os.path.exists("CZT_PM25_Daily"): os.makedirs("CZT_PM25_Daily") 
    try:
        train_dataset = EnhancedPM25Dataset(r"CZT_PM25_Daily", config, augment=True)
    except:
        print("未找到数据，请确保'CZT_PM25_Daily'文件夹内有.tif文件。")
        exit()
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    
    # [修改] 使用新的 Controlled 模型
    model = ControlledKoopmanModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    criterion = SharpnessAwareLoss(config).to(device)
    
    print(f"\n>>> 启动轻量化训练 (Control={config.USE_METEO_CONTROL}, No Skips, Dim={config.LATENT_DIM}) <<<")
    
    # --- 2. 训练循环 ---
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_pred_loss = 0
        total_spectral_loss = 0
        
        for seq_img, seq_meteo, mask in train_loader:
            seq_img = seq_img.to(device)
            seq_meteo = seq_meteo.to(device) # (B, Seq, 2)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            x_curr = seq_img[:, 0] 
            # 初始编码
            z_curr = model.encoder(x_curr)
            
            loss_batch = 0 
            # 递归预测
            for t in range(1, config.SEQUENCE_LENGTH):
                # 获取驱动 t 时刻变化的气象控制量
                u_curr = seq_meteo[:, t-1] 
                
                # 1. 动力学演化: K*z + (B*u or 0)
                # [修改 4] 使用 compute_control 处理 B 可能为 None 的情况
                control_term = model.compute_control(u_curr)
                z_next_pred = model.K(z_curr) + control_term
                
                # 2. 解码 (无 Skip)
                delta = model.decoder(z_next_pred)
                
                # 3. 物理残差更新
                x_next_pred = torch.relu(x_curr + delta)
                
                # 4. 计算损失
                x_target = seq_img[:, t]
                m_target = mask[:, t]
                
                loss_step = criterion(x_next_pred, x_target, m_target)
                loss_batch += loss_step
                
                # 5. 更新状态
                x_curr = x_next_pred 
                z_curr = z_next_pred              
                 
            # --- 谱正则化 (针对 K 矩阵) ---
            K_weight = model.K.weight
            eigs = torch.linalg.eigvals(K_weight)
            loss_spectral = torch.mean(torch.relu(torch.abs(eigs) - config.EIGEN_MAX)) + \
                            torch.mean(torch.relu(config.EIGEN_MIN - torch.abs(eigs)))
            
            # 汇总
            loss_spectral_weighted = config.W_SPECTRAL * loss_spectral
            loss = loss_batch + loss_spectral_weighted
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_pred_loss += loss_batch.item()
            total_spectral_loss += loss_spectral_weighted.item()
            
        # 打印日志
        if (epoch+1) % 1 == 0:
            avg_loss = total_loss / len(train_loader)
            avg_pred = total_pred_loss / len(train_loader)
            avg_spec = total_spectral_loss / len(train_loader)
            max_eig = torch.max(torch.abs(eigs)).item()
            print(f"Epoch {epoch+1:03d} | Total={avg_loss:.4f} | Pred={avg_pred:.4f} | Spec={avg_spec:.4f} | Max Eig={max_eig:.4f}")

    # --- 3. 产出 ---
    output_dir = "results_controlled"
    os.makedirs(output_dir, exist_ok=True)
    predict_and_export_n_days(model, train_dataset, device, output_dir, start_day_idx=0, n_days=7)
    print(f"\n任务完成! 结果保存在: {output_dir}")
