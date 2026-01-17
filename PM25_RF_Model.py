import pandas as pd
import numpy as np
import joblib  # 用于保存模型
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ================= 1. 数据准备 =================
print("🚀 Step 1: 加载训练数据...")
try:
    df = pd.read_csv('Final_Training_Dataset.csv')
    print(f"✅ 数据加载成功，总样本量: {len(df)}")
except FileNotFoundError:
    print("❌ 错误: 找不到 Final_Training_Dataset.csv，请先运行数据合并脚本。")
    exit()

# 定义特征 (X) 和 标签 (y)
# 注意：一定要和 GEE 导出的列名保持一致
feature_cols = ['AOD_055', 'TEMP_C', 'WIND_SPEED', 'PRESSURE_HPA', 'RAIN_MM']
target_col = 'PM25'

X = df[feature_cols]
y = df[target_col]

# 划分数据集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#random_state = 42)

print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

# ================= 2. 模型训练 =================
print("\n🚀 Step 2: 训练随机森林模型 (Random Forest)...")

# 初始化模型
# n_estimators: 树的数量，越多通常越稳，但越慢
# max_depth: 树的深度，限制深度防止过拟合
# n_jobs=-1: 调用所有 CPU 核心加速
rf_model = RandomForestRegressor(n_estimators=300, 
                                 max_depth=20, 
                                 min_samples_split=5, 
                                 min_samples_leaf=2,
                                 random_state=42, 
                                 n_jobs=-1)

# 拟合数据
rf_model.fit(X_train, y_train)
print("✅ 模型训练完成！")

# ================= 3. 模型评估 =================
print("\n🚀 Step 3: 模型性能评估")

# 预测
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# 计算指标
def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{name}] R²: {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f}")
    return r2

r2_train = evaluate(y_train, y_pred_train, "训练集")
r2_test = evaluate(y_test, y_pred_test, "测试集 (CV)")

# ================= 4. 结果可视化 =================
print("\n🚀 Step 4: 生成评估图表...")

plt.figure(figsize=(14, 6))

# 图 1: 特征重要性
plt.subplot(1, 2, 1)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=[feature_cols[i] for i in indices], palette="viridis")
plt.title("变量重要性 (Feature Importance)")
plt.xlabel("相对重要性")

# 图 2: 散点拟合图 (只画测试集)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='#3498db', alpha=0.6, label='Test Samples')
# 画 1:1 线
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
plt.xlabel(r'地面监测 PM2.5 ($ \mu g / m^3 $)')
plt.ylabel(r'模型反演 PM2.5 ($ \mu g / m^3 $)')
plt.title(f'模型精度验证 ($R^2={r2_test:.2f}$)')
plt.legend()
plt.tight_layout()
plt.show()

# ================= 5. 保存模型 =================
# 将训练好的模型保存到本地，以便下一步生成地图时直接调用
model_filename = 'PM25_RF_Model.joblib'
joblib.dump(rf_model, model_filename)
print(f"\n✅ 模型已保存为: {model_filename}")
print("下一步：下载区域遥感影像，使用此模型进行空间制图！")
