import pandas as pd

import numpy as np

# 1. 读取 GEE 导出的特征文件

df_2023 = pd.read_csv('CZT_Station_Data_AccumulatedRain_2023.csv')

df_2024 = pd.read_csv('CZT_Station_Data_AccumulatedRain_2024.csv')

# 2. 上下合并

df_features = pd.concat([df_2023, df_2024], ignore_index=True)

# 3. 数据清洗

# 方案 A (严谨): 直接删除。如果云太多，这会导致样本量大幅减少。

df_features = df_features[df_features['AOD_055'] != -999]

# 3.2 格式化日期 (确保和 PM2.5 表一致)

df_features['date'] = pd.to_datetime(df_features['date_str'])

# 4. 导出清洗后的特征数据
output_filename = 'Final_Model_Features_2023_2024.csv'

# index=False 表示不保存行号，encoding='utf-8-sig' 防止中文乱码
df_features.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"✅ 成功导出！文件已保存为: {output_filename}")
print(f"包含列: {list(df_features.columns)}")
