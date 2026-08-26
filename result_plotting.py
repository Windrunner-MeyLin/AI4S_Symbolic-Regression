import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import os
from feature_information import FeatureInformation


def plot_result(feature:FeatureInformation, target:pd.DataFrame, path, filename='fit.png'):

    # 定义颜色
    commoncolor3 = np.array([[81, 81, 81],
                            [241, 64, 64],
                            [26, 111, 223],
                            [55, 173, 107],
                            [177, 119, 222],
                            [204, 153, 0],
                            [0, 203, 204],
                            [125, 78, 78],
                            [142, 142, 0],
                            [251, 101, 1],
                            [102, 153, 204],
                            [111, 184, 2]]) / 255.0

    threecolors = np.array([[57, 156, 102],
                            [0, 128, 102],
                            [77, 133, 189],
                            [247, 144, 61],
                            [89, 169, 90]]) / 255.0

    colorblue = np.array([36, 103, 180]) / 255.0
    redcolor = np.array([255, 59, 59]) / 255.0

    x = feature.get_full_data().to_numpy()
#     x_name = feature.get_full_name()

    y = target.to_numpy()
#     y_name = target.columns[0]

    # 创建图形2
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_facecolor('w')

    # 绘制数据点
    ax.plot(x, y,  'o', color=threecolors[0,:],
            markersize=6,  
            markerfacecolor=threecolors[0,:], 
            markeredgecolor=threecolors[0,:], label='Data')

    x_max = max(x)+1
    y_max = max(y)+1

    # # 绘制拟合线
    ax.plot([-1, max(x_max, y_max)[0]], [-1, max(x_max, y_max)[0]], '-', 
            color=redcolor, linewidth=4, label='Fit')

    # 设置刻度和标签
    ax.set_xticks(np.arange(0, x_max + max(x_max // 5, 1), max(x_max // 5, 1)))
    ax.set_yticks(np.arange(0, y_max + max(y_max // 8, 1), max(y_max // 8, 1)))
    ax.set_xlim([-0.3, x_max])
    ax.set_ylim([-0.3, y_max])


    # 设置坐标轴标签
    ax.set_xlabel(r'$q_{pre}$', fontsize=30, 
                  fontweight='normal', family='DejaVu Sans')
    ax.set_ylabel(r'$q_{real}$', fontsize=30, fontweight='normal', 
                  family='DejaVu Sans')

    ax.spines['bottom'].set_linewidth(2)  # 底部轴加粗
    ax.spines['left'].set_linewidth(2)    # 左侧轴加粗
    ax.spines['top'].set_linewidth(2)     # 上侧轴加粗
    ax.spines['right'].set_linewidth(2)   # 右侧轴加粗

    # **显示上轴和右轴**
    ax.tick_params(axis='x', which='both', direction='in', 
                length=6, width=2, labelsize=12, top=True)  # X轴的刻度向内，上轴刻度开启
    ax.tick_params(axis='y', which='both', direction='in', 
                length=6, width=2, labelsize=12, right=True) # Y轴的刻度向内，右轴刻度开启
    ax.tick_params(axis='both', which='minor', bottom=False, 
                top=False, left=False, right=False)

    # 设置字体
    ax.tick_params(axis='both', labelsize=25)
    ax.minorticks_on()

    # 显示图形
    full_path = os.path.join(path, filename)
    plt.savefig(full_path)
    plt.close()