import rasterio
import numpy as np
import joblib
import os

# ================= 1. 环境与路径配置 =================
input_dir = 'CZT_Daily_2023/'        # GEE 导出的 28 张特征底图文件夹
model_path = 'PM25_RF_Model.joblib'           # 之前训练好的随机森林模型
output_pm25_dir = 'CZT_PM25_Daily/'       # 反演后的 PM2.5 图像存放处
if not os.path.exists(output_pm25_dir): os.makedirs(output_pm25_dir)

# 加载模型
rf_model = joblib.load(model_path)
# 获取排序后的文件列表，确保时间顺序 (20230201, 20230202...)
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])


# ================= 2. 空间掩膜与元数据初始化 =================
# 我们以第一张图为基准，确定地理范围和有效像素（陆地/非云区）
print("初始化空间掩膜...")
with rasterio.open(os.path.join(input_dir, files[0])) as src:
    profile = src.profile
    h, w = src.height, src.width
    # 读取第一波段（AOD）来做掩膜
    base_img = src.read(1)
    # 只要不是 NoData 且不是 NaN 的像素都视为有效
    # 注意：GEE 导出的 NoData 值需根据你脚本中的 unmask 设置判断
    mask = (base_img != src.nodata) & (~np.isnan(base_img))
    valid_idx = np.where(mask.flatten())[0] # 获取有效像素的一维索引

print(f"全图尺寸: {h}x{w}, 有效建模像素: {len(valid_idx)}")

# ================= 3. 批量循环：反演 + 矩阵重构 =================
# 创建一个列表，用来存储每一天的“有效像素列向量”
snapshot_vectors = []

print("开始批量反演 89 天数据...")
for f in files:
    with rasterio.open(os.path.join(input_dir, f)) as src:
        # 读取 5 个波段: [AOD, TEMP, WIND, PRESS, RAIN]
        img_data = src.read() 
        c, h, w = img_data.shape
        
        # 陷阱处理：将 (C, H, W) 转为 (H*W, C) 方便模型读取
        X_all = img_data.reshape(c, -1).T
        
        # 预测该日的 PM2.5 (只针对有效区域)
        pm25_valid = rf_model.predict(X_valid)
        
        # 将结果存入快照列表
        snapshot_vectors.append(pm25_valid)
        
        # --- 导出反演图 (用于 QGIS 检查) ---
        pm25_full_map = np.full(h * w, np.nan) # 先填满 NaN
        pm25_full_map[valid_idx] = pm25_valid  # 把预测值填回对应位置
        pm25_full_map = pm25_full_map.reshape(h, w)
        
        out_name = f.replace('Inversion_Input', 'PM25_Map')
        profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(os.path.join(output_pm25_dir, out_name), 'w', **profile) as dst:
            dst.write(pm25_full_map.astype(np.float32), 1)
    
    print(f"已完成: {f}")

# ================= 4. 构建最终快照矩阵 (Space x Time) =================
# 列表转为矩阵: (Time, Space) -> 转置 -> (Space, Time)
Data_Matrix = np.array(snapshot_vectors).T 

print("\n" + "="*30)
print("快照矩阵构建完成！")
print(f"矩阵维度: {Data_Matrix.shape} (行:有效像素, 列:时间步)")
print(f"该矩阵可直接进入 DMD 算法或保存为 .npy / .mat 供后续使用")
print("="*30)

# 保存矩阵，方便在 MATLAB 或其他 Python 脚本中直接调用
np.save('CZT_Feb_Snapshot_Matrix.npy', Data_Matrix)