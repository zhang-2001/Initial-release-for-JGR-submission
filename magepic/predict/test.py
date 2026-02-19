import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 读取数据
with open('dis.txt', 'r') as f:
    lines = f.readlines()

# 第一行：真实震级
y_true = np.array([float(x) for x in lines[0].strip().split()])
# 第二行：预测震级
y_pred = np.array([float(x) for x in lines[1].strip().split()])
# 第三行：真实值与预测值差的绝对值
abs_errors = np.array([float(x) for x in lines[2].strip().split()])

# 计算评估指标
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
std_error = np.std(abs_errors)

# 打印结果
print("地震震级预测性能评估：")
print(f"R²:     {r2:.4f}")
print(f"MAE:    {mae:.4f}")
print(f"RMSE:   {rmse:.4f}")
print(f"误差标准差: {std_error:.4f}")

# 保存结果到文件
# with open('result.txt', 'w') as f:
#     f.write("地震震级预测性能评估：\n")
#     f.write(f"R²:     {r2:.4f}\n")
#     f.write(f"MAE:    {mae:.4f}\n")
#     f.write(f"RMSE:   {rmse:.4f}\n")
#     f.write(f"误差标准差: {std_error:.4f}\n")
#
# print("\n结果已保存到 result.txt")