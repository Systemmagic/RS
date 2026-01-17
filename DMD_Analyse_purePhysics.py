import rasterio
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from numpy import linalg as LA

# ================= 1. 配置路径与模型加载 =================
# 请确保这个文件夹里直接只有 .tif 文件，没有嵌套文件夹
INPUT_DIR = 'CZT_Daily_Feb_2023/' 
MODEL_PATH = 'PM25_RF_Model.joblib'           
OUTPUT_MODE_DIR = 'CZT_DMD_Result/'

if not os.path.exists(OUTPUT_MODE_DIR): os.makedirs(OUTPUT_MODE_DIR)

print(f"正在加载模型: {MODEL_PATH}")
try:
    rf_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"找不到模型文件！请确认文件名是否为 PM2.5_RF_Model.joblib")
    exit()

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')])
if len(files) == 0:
    print("❌ 输入文件夹为空！请检查路径。")
    exit()

# ================= 2. 空间掩膜与快照矩阵构建 =================
print("Step 1: 正在进行全区反演并构建快照矩阵...")

with rasterio.open(os.path.join(INPUT_DIR, files[0])) as src:
    profile = src.profile
    h, w = src.height, src.width
    sample_aod = src.read(1)
    # 获取有效像素索引
    mask = (sample_aod != src.nodata) & (~np.isnan(sample_aod))
    valid_idx = np.where(mask.flatten())[0]

snapshot_list = []

for f in files:
    with rasterio.open(os.path.join(INPUT_DIR, f)) as src:
        img_data = src.read() 
        X_all = img_data.reshape(img_data.shape[0], -1).T
        
        # 提取有效区域
        X_valid = X_all[valid_idx, :]
        
        # 【关键修改】防止 NaN 导致 RF 崩溃 (将 NaN 填为 0)
        X_valid = np.nan_to_num(X_valid, nan=0.0)

        pm25_valid = rf_model.predict(X_valid)
        snapshot_list.append(pm25_valid)

Data_Matrix = np.array(snapshot_list).T 
print(f"✅ 快照矩阵构建完成，维度: {Data_Matrix.shape}")

# ================= 3. DMD 核心算法 =================
print("Step 2: 正在执行 DMD 分解...")

X1 = Data_Matrix[:, :-1]
X2 = Data_Matrix[:, 1:]

# 截断秩 r
r = 6 
U, s, Vh = LA.svd(X1, full_matrices=False)
Ur = U[:, :r]
Sr = np.diag(s[:r])
Vr = Vh[:r, :].T.conj()

Atilde = Ur.T.conj() @ X2 @ Vr @ LA.inv(Sr)
eigvals, W = LA.eig(Atilde)
Phi = X2 @ Vr @ LA.inv(Sr) @ W

# 按幅度排序
b = LA.lstsq(Phi, X1[:, 0], rcond=None)[0]
idx = np.argsort(np.abs(b))[::-1]
eigvals = eigvals[idx]
Phi = Phi[:, idx]

print("✅ Koopman 算子计算完成。")
import scipy.ndimage as ndimage  # 新增：用于图像平滑

# ... (前面的加载和 DMD 计算部分保持不变) ...

# ================= 4. 结果可视化与修正 =================

# A. 特征值单位圆 (物理意义解释优化)
plt.figure(figsize=(6, 6))
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle (Stable)') 
plt.scatter(eigvals.real, eigvals.imag, c=np.abs(b[idx]), cmap='viridis', s=80, edgecolors='k')

# 画一个 0.8 的内圆，辅助判断
plt.plot(0.8*np.cos(theta), 0.8*np.sin(theta), 'r:', label='Fast Decay Zone')

plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.legend(loc='upper right')
plt.title('Koopman Eigenvalues\n(Points inside circle = Pollution Dissipating)')
plt.xlabel('Real'); plt.ylabel('Imag')
plt.axis('equal')
plt.savefig(os.path.join(OUTPUT_MODE_DIR, 'Eigenvalues_Circle_Fixed.png'))
plt.show()

# B. 空间模态导出 (增加平滑处理，消除方块)
print("Step 3: 正在导出优化后的模态...")

profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)

for i in range(3):
    # 1. 提取原始模态
    mode_vec = np.real(Phi[:, i])
    
    # 2. 还原回 2D 矩阵
    mode_map = np.full(h * w, np.nan)
    mode_map[valid_idx] = mode_vec
    mode_map = mode_map.reshape(h, w)
    
    # ================= 关键修正：高斯平滑消除方块 =================
    # sigma=2 代表约 2km 的平滑半径，足以抹平 ERA5 的锐利边缘，保留趋势
    # 这里的 mask 保证只平滑有效区域，不把边界外的 NaN 卷进来
    
    # 先把 NaN 填为 0 以便平滑
    temp_map = np.nan_to_num(mode_map, nan=0)
    # 进行平滑
    smoothed_map = ndimage.gaussian_filter(temp_map, sigma=1.5) 
    # 把原来的 NaN 掩膜盖回去
    smoothed_map[np.isnan(mode_map)] = np.nan
    
    # ===========================================================
    
    # 3. 导出 TIFF
    mode_path = os.path.join(OUTPUT_MODE_DIR, f'DMD_Mode_{i+1}_Smoothed.tif')
    with rasterio.open(mode_path, 'w', **profile) as dst:
        dst.write(smoothed_map.astype(np.float32), 1)
    
    # 4. 绘图预览
    plt.figure(figsize=(8, 6))
    limit = np.nanmax(np.abs(smoothed_map)) * 0.8 
    plt.imshow(smoothed_map, cmap='RdBu_r', vmin=-limit, vmax=limit) 
    plt.colorbar(label='Amplitude')
    
    # 自动生成物理标题
    amp = np.abs(eigvals[i])
    if amp > 0.95:
        status = "Slow Decay (Persistent)"
    elif amp > 0.8:
        status = "Normal Decay"
    else:
        status = "Fast Decay (Noise/Transient)"
        
    plt.title(f'Mode #{i+1} (Smoothed)\nλ={eigvals[i]:.3f} | {status}')
    plt.axis('off')
    plt.show()

print("✅ 优化完成！请检查 Smoothed TIFF 是否消除了方块效应。")
