from expansion_v2 import expand_v2
from result_sorting import sort_result
from result_displaying import display_result
from sis_ana import sis
from coefficients_fitting import fit
from result_plotting import plot_result
from timer import Timer
import pandas as pd
import torch
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 代码运行目录
path = os.getcwd()
# path = "yourfilepath"

# 读取数据
print("读取数据...")
data = pd.read_csv(os.path.join(path, "data.csv"))
focus = pd.read_csv(os.path.join(path, "focus.csv"))
print(f"  输入特征: {list(data.columns)}, 样本数: {len(data)}")

# 初始输入的扩充
print("特征空间扩展 (v2)...")
data_expanded, _ = expand_v2(data)
print(f"  扩展后特征数: {data_expanded.shape[1]}")

# 确定性独立筛选
print("SIS 确定性独立筛选...")
data_sis = sis(data_expanded, focus.to_numpy(), 200)  # 增大SIS保留数以覆盖5变量特征
print(f"  筛选后保留: {data_sis.shape[1]} 个特征")

# 系数拟合
print("系数拟合 (解析求解)...")
r2, coef, loss = fit(data_sis, focus.to_numpy(), device)

# 结果整理
results = sort_result(data_sis.to_numpy(), 
                      data_sis.columns.to_numpy(),
                      r2, coef, loss, 10)

print(f"  最佳 R² = {results[0].get_r2():.4f}")
print(f"  最佳模型 = {results[0].get_full_name()}")

# 输出日志
display_result(results, 10, path)
print("  日志已保存到 log.log")

# 单独保存实验结果为 result_1.log
result_log = f"R²={results[0].get_r2():.4f}, 模型={results[0].get_full_name()}\n"
with open(os.path.join(path, "result_1.log"), "w", encoding="utf-8") as f:
    f.write(result_log)
print("  实验结果已保存到 result_1.log")

# 绘制图像
plot_result(results[0], focus, path, 'fit_1.png')
print("  拟合图像已保存到 fit_1.png")
