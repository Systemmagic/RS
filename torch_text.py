import os
import glob
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==============================================================================
# ================= 1. 环境与模型定义 =================
# ==============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepKoopmanModel(nn.Module):
    def __init__(self, input_h=64, input_w=64):
        super().__init__()
        self.encoder_cnn = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.feature_h, self.feature_w = input_h // 4, input_w // 4
        self.flatten_dim = 8 * self.feature_h * self.feature_w
        self.koopman_matrix = nn.Linear(self.flatten_dim, self.flatten_dim, bias=False)
        self.decoder_cnn = nn.Sequential(nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1))
    def encode(self, x): return self.encoder_cnn(x).view(x.size(0), -1)
    def forward(self, x):
        z_t = self.encode(x); z_next_pred = self.koopman_matrix(z_t)
        x_recon = self.decoder_cnn(z_t.view(x.size(0), 8, self.feature_h, self.feature_w))
        x_pred = self.decoder_cnn(z_next_pred.view(x.size(0), 8, self.feature_h, self.feature_w))
        return x_recon, x_pred, z_t, z_next_pred
    def predict_future(self, x_start, steps=7):
        z_current = self.encode(x_start); predictions = []
        for _ in range(steps):
            z_next = self.koopman_matrix(z_current)
            x_future = self.decoder_cnn(z_next.view(x_start.size(0), 8, self.feature_h, self.feature_w))
            predictions.append(x_future); z_current = z_next
        return predictions

# ==============================================================================
# ================= 2. 数据集定义 =================
# ==============================================================================
class PM25TIFFDataset(Dataset):
    def __init__(self, data_dir, img_size=64):
        self.img_size = img_size; self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        if not self.file_paths: raise FileNotFoundError(f"路径 '{data_dir}' 中未找到 .tif 文件。")
        print(f"找到 {len(self.file_paths)} 张 TIFF 文件。正在加载..."); self.data = np.array([cv2.resize(tifffile.imread(p), (self.img_size, self.img_size), cv2.INTER_AREA) for p in self.file_paths], dtype=np.float32)
        self.global_min, self.global_max = np.nanmin(self.data), np.nanmax(self.data); print(f"加载完成。全局Min: {self.global_min:.2f}, Max: {self.global_max:.2f}")
    def __len__(self): return len(self.data) - 1
    def __getitem__(self, idx):
        x_raw, y_raw = self.data[idx], self.data[idx + 1]; mask = ~np.isnan(y_raw)
        if self.global_max > self.global_min:
            x_norm, y_norm = [(d - self.global_min) / (self.global_max - self.global_min) for d in (x_raw, y_raw)]
        else: x_norm, y_norm = x_raw, y_raw
        x, y = np.nan_to_num(x_norm, nan=0.0), np.nan_to_num(y_norm, nan=0.0)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    def get_all_data(self):
        if self.global_max > self.global_min: return (self.data - self.global_min) / (self.global_max - self.global_min)
        return self.data
    def denormalize(self, img_norm): return img_norm * (self.global_max - self.global_min) + self.global_min

# ==============================================================================
# ================= 3. 验证、分析与导出函数 =================
# ==============================================================================
def moving_horizon_evaluation_and_export(model, dataset, device, output_dir, forecast_window=7):
    model.eval(); data_full = dataset.get_all_data(); total_time, H, W = data_full.shape
    tif_export_dir = os.path.join(output_dir, "predicted_tifs"); os.makedirs(tif_export_dir, exist_ok=True)
    print(f"\n开始移动视窗评估... (结果将保存在 '{output_dir}')"); list_rmse, list_mae = [], []; prediction_full = np.full_like(data_full, np.nan); exported = False
    with torch.no_grad():
        for t in range(total_time - forecast_window):
            x_start_raw = data_full[t]; x_start = np.nan_to_num(x_start_raw, nan=0.0)
            x_start_tensor = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            preds_list = model.predict_future(x_start_tensor, steps=forecast_window)
            truth_block = data_full[t+1 : t+1+forecast_window]; preds_block_norm = torch.cat(preds_list, dim=0).squeeze(1).cpu().numpy()
            E = preds_block_norm - truth_block; mae_val, rmse_val = np.nanmean(np.abs(E)), np.sqrt(np.nanmean(E**2))
            if not np.isnan(mae_val): list_mae.append(mae_val)
            if not np.isnan(rmse_val): list_rmse.append(rmse_val)
            if t < len(prediction_full) - 1: prediction_full[t+1] = preds_block_norm[0]
            if not exported:
                preds_block_unnorm = preds_block_norm * (dataset.global_max - dataset.global_min) + dataset.global_min
                output_tif_path = os.path.join(tif_export_dir, f"forecast_from_day_{t}.tif")
                tifffile.imwrite(output_tif_path, preds_block_unnorm.astype(np.float32)); print(f"已导出一个7天预测样本: {output_tif_path}"); exported = True
    global_mae, global_rmse = (np.mean(l) if l else np.nan for l in (list_mae, list_rmse)); print(f"评估完成！Global RMSE: {global_rmse:.6f}, Global MAE : {global_mae:.6f}")
    plt.figure(figsize=(12, 6)); flat_true = data_full[~np.isnan(data_full)]; flat_pred = prediction_full[~np.isnan(prediction_full)]
    plt.hist(flat_true, bins=50, density=True, alpha=0.6, color='blue', label='True Data PDF'); plt.hist(flat_pred, bins=50, density=True, alpha=0.6, color='orange', label='Forecast Data PDF')
    plt.title('Histogram (Probability Density Function)'); plt.legend(); plt.grid(True, alpha=0.3); pdf_path = os.path.join(output_dir, "PDF_Comparison.png")
    plt.savefig(pdf_path); plt.close(); print(f"已保存PDF对比图: {pdf_path}")

def analyze_stability(model, output_dir):
    K = model.koopman_matrix.weight.detach().cpu().numpy(); eigenvalues = np.linalg.eigvals(K); amplitudes = np.abs(eigenvalues)
    print("\n=== Koopman 矩阵谱分析 ==="); print(f"最大特征模长: {np.max(amplitudes):.4f}"); print(f"不稳定特征值数量: {np.sum(amplitudes > 1.0)} / {len(eigenvalues)}")
    plt.figure(figsize=(6, 6)); plt.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), 'k--', label='Unit Circle')
    plt.scatter(eigenvalues.real, eigenvalues.imag, c='red', alpha=0.6, label='Eigenvalues'); plt.axvline(0, color='gray', lw=0.5); plt.axhline(0, color='gray', lw=0.5)
    plt.xlabel('Real'); plt.ylabel('Imaginary'); plt.title('Koopman Eigenvalue Spectrum'); plt.axis('equal'); plt.legend()
    eig_path = os.path.join(output_dir, "Eigenvalue_Spectrum.png"); plt.savefig(eig_path); plt.close(); print(f"已保存特征值谱图: {eig_path}")

# ==============================================================================
# ================= 4. 主程序入口 (最终大师版) =================
# ==============================================================================
if __name__ == "__main__":
    
    # --- 模块 1: 初始化 ---
    device = get_device()
    output_dir = "results"; os.makedirs(output_dir, exist_ok=True)
    real_data_dir = r"CZT_PM25_Daily" 
    try: dataset = PM25TIFFDataset(real_data_dir, img_size=64)
    except FileNotFoundError as e: print(e); exit()
    
    real_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    model = DeepKoopmanModel(input_h=64, input_w=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    class MaskedL1Loss(nn.Module):
        def forward(self, pred, target, mask):
            error = torch.abs(pred - target); masked_error = error * mask.float()
            return torch.sum(masked_error) / (torch.sum(mask) + 1e-8)
    
    criterion_masked = MaskedL1Loss()
    criterion_latent = nn.L1Loss()
    
    SPECTRAL_WEIGHT = 1e-4 

    print(f"Using device: {device}")
    print(f"开始最终训练... (统一加权损失, Spectral Weight = {SPECTRAL_WEIGHT})")

    # --- 模块 2: 训练模型 (统一战线策略) ---
    for epoch in range(30):
        model.train()
        total_base_loss = 0.0
        total_spectral_loss = 0.0
        
        for x_t, x_next_real, mask in real_loader:
            x_t, x_next_real, mask = x_t.to(device), x_next_real.to(device), mask.to(device)
            optimizer.zero_grad()
            
            x_recon, x_next_pred, z_t, z_next_pred = model(x_t)
            with torch.no_grad(): z_next_real = model.encode(x_next_real)

            loss_recon = criterion_masked(x_recon, x_t, mask)
            loss_pred = criterion_masked(x_next_pred, x_next_real, mask)
            loss_latent = criterion_latent(z_next_pred, z_next_real)
            base_loss = loss_recon + loss_pred + (0.1 * loss_latent)
            
            K = model.koopman_matrix.weight
            eigen_magnitudes = torch.abs(torch.linalg.eigvals(K))
            loss_spectral = torch.sum(torch.relu(eigen_magnitudes - 1.0))

            total_loss = base_loss + SPECTRAL_WEIGHT * loss_spectral
            
            total_loss.backward()
            optimizer.step()
            
            total_base_loss += base_loss.item()
            total_spectral_loss += loss_spectral.item()
        
        if (epoch + 1) % 10 == 0:
            avg_base_loss = total_base_loss / len(real_loader)
            avg_spectral_loss = total_spectral_loss / len(real_loader)
            print(f"Epoch {epoch+1:04d}: Base Loss = {avg_base_loss:.6f}, Avg Raw Spectral Loss = {avg_spectral_loss:.6f}")

    # --- 模块 3: 评估、导出与分析 ---
    print("\n--- 训练完成 ---")
    try:
        moving_horizon_evaluation_and_export(model, dataset, device, output_dir, forecast_window=7)
        analyze_stability(model, output_dir)
    except Exception as e:
        print(f"验证出错: {e}")
