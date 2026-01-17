import pandas as pd

# 1. 读取数据
# 请确保文件名与您电脑上的实际文件名完全一致
try:
    df_pm25 = pd.read_csv('国控日均_2023_2024.csv')
    df_stations = pd.read_csv('工作簿1.csv') # 如果报错，请检查这里的文件名或使用绝对路径
except FileNotFoundError as e:
    print(f"文件读取失败，请检查路径: {e}")
    exit()

# === 数据清洗优化 ===
# 您上传的站点表中经纬度带有 '°' 符号 (如 113.1012°)，这会变成字符串。
#我们需要去掉它并转为数字，方便后续使用。
def clean_coord(val):
    if isinstance(val, str):
        return float(val.replace('°', ''))
    return val

if '经度 (E)' in df_stations.columns:
    df_stations['经度 (E)'] = df_stations['经度 (E)'].apply(clean_coord)
    df_stations['纬度 (N)'] = df_stations['纬度 (N)'].apply(clean_coord)

# 2. 宽表转长表 (Melt)
# 自动提取 PM2.5 数据中存在的站点列，防止硬编码报错
station_cols = ['1335A', '1336A', '1337A', '1338A', '1339A', '1340A', '1341A', '1342A', '1343A', '1344A']
available_cols = [c for c in station_cols if c in df_pm25.columns]

df_long = df_pm25.melt(id_vars=['datetime'],
                       value_vars=available_cols,
                       var_name='Station_ID', 
                       value_name='PM25')

# 3. 关联经纬度
# 确保两边都有关联键。这里左表是 Station_ID，右表是 国控编号
train_base = pd.merge(df_long, df_stations, left_on='Station_ID', right_on='国控编号', how='left')

# 4. 查看结果
print("合并成功！数据预览：")
print(train_base[['datetime', 'Station_ID', 'PM25', '监测站点名称（标准名）', '经度 (E)', '纬度 (N)']].head())

# 5. [Python] 保存清洗好的地面真实值 (Ground Truth)
encoding='utf-8-sig' #可以防止中文乱码
train_base.to_csv('Station_PM25_GroundTruth.csv', index=False, encoding='utf-8-sig')

print("文件已保存为: Station_PM25_GroundTruth.csv")
print("下一步：请编写正确的 GEE 代码，去提取这些站点对应的 AOD 和气象数据。")

