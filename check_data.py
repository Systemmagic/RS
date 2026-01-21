import os
import glob
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ！！！修改这里！！！
# 把下面的路径改成你存放 .tif 文件的真实文件夹路径
# 注意：Windows路径如果包含反斜杠 \，请在引号前加 r，例如 r"D:\MyData\PM25"
DATA_DIR = r"D:\workspace\RS\CZT_PM25_Daily" 
# ==========================================

def diagnose_data(data_dir):
    # 1. 检查文件是否存在
    tif_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    print(f"--- 诊断报告 ---")
    print(f"1. 文件夹路径: {data_dir}")
    
    if not tif_files:
        print("❌ 错误: 该文件夹下没有找到 .tif 文件！请检查路径是否正确。")
        return

    print(f"✅ 发现 {len(tif_files)} 个 .tif 文件。")
    print(f"   示例文件: {os.path.basename(tif_files[0])}")

    # 2. 尝试读取第一个文件，检查形状和数值范围
    try:
        ds = gdal.Open(tif_files[0])
        if ds is None:
            print("❌ 错误: 无法通过 GDAL 打开文件。文件可能损坏或格式不支持。")
            return
            
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        print(f"2. 图像尺寸: 宽={width}, 高={height}, 通道数={bands}")
        
        # 读取数据
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        
        if arr is None:
             print("❌ 错误: 读取像素数据失败。")
             return

        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        has_nan = np.isnan(arr).any()
        
        print(f"3. 数值统计:")
        print(f"   最小值: {min_val:.4f}")
        print(f"   最大值: {max_val:.4f}")
        print(f"   是否存在 NaN (空值): {'是 (这会导致模型报错!)' if has_nan else '否'}")

        # 4. 这里的建议
        print(f"--- 建议与调整 ---")
        if width != height:
            print(f"⚠️ 注意: 图像不是正方形 ({width}x{height})。原代码会自动补零填充成正方形，可能会产生黑边。")
        
        if max_val > 1000:
             print("ℹ️ 提示: 数值较大，原代码包含归一化处理，应该没问题。")
             
        if bands > 1:
             print("⚠️ 警告: 你的数据有多个通道，但原代码默认只读取第1个通道。如果需要多通道，需要修改 Dataset 类。")

    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    diagnose_data(DATA_DIR)
