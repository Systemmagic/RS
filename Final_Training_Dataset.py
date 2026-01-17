import pandas as pd

print("🚀 开始数据合并流程...")

# ================= 1. 读取数据 =================
# 读取 Y (地面真实值)
try:
    df_y = pd.read_csv('Station_PM25_GroundTruth.csv')
    print(f"✅ 地面数据读取成功: {len(df_y)} 行")
except FileNotFoundError:
    print("❌ 错误: 找不到 Station_PM25_GroundTruth.csv")
    exit()

# 读取 X (卫星特征值)
try:
    df_x = pd.read_csv('Final_Model_Features_2023_2024.csv')
    print(f"✅ 卫星特征读取成功: {len(df_x)} 行")
except FileNotFoundError:
    print("❌ 错误: 找不到 Final_Model_Features_2023_2024.csv")
    exit()

# ================= 2. 统一时间格式 =================
# 这一步至关重要，因为 CSV 里日期变成了字符串，格式可能不一样（比如 '2023/1/1' vs '2023-01-01'）
# pd.to_datetime 会自动处理这些差异
df_y['date'] = pd.to_datetime(df_y['datetime']) 
# 注意：之前保存特征表时如果已经有了 'date' 列就用 'date'，如果没有就用 'date_str'
# 这里做一个容错处理
date_col_x = 'date' if 'date' in df_x.columns else 'date_str'
df_x['date'] = pd.to_datetime(df_x[date_col_x])

# ================= 3. 执行合并 (Inner Merge) =================
# Inner Join: 只保留“既有地面监测又有卫星数据”的那些天
df_final = pd.merge(df_y, df_x, 
                    left_on=['Station_ID', 'date'], 
                    right_on=['sid', 'date'], 
                    how='inner')

# ================= 4. 检查与保存 =================
print(f"\n📊 合并结果统计:")
print(f"地面原始数据量: {len(df_y)}")
print(f"卫星原始数据量: {len(df_x)}")
print(f"最终匹配样本量: {len(df_final)} (这是用于训练的有效数据)")

if len(df_final) > 0:
    # 预览前5行
    print("\n数据预览:")
    print(df_final[['date', 'Station_ID', 'PM25', 'AOD_055', 'TEMP_C', 'RAIN_MM']].head())
    
    # 保存最终文件
    output_file = 'Final_Training_Dataset.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n🎉 恭喜！最终训练集已保存为: {output_file}")
    print("您现在可以直接运行随机森林模型代码了！")
else:
    print("\n⚠️ 警告: 合并后数据量为 0！")
    print("可能原因：")
    print("1. 站点ID不匹配 (例如: '1335A' vs 'Changsha_JKQ')")
    print("2. 日期范围不重叠")
