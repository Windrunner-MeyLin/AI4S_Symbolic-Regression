import pandas as pd
import numpy as np
from numpy.typing import NDArray as ND

def sis(x:pd.DataFrame, y:ND, n:int):
    '''
    使用sis方法进行筛选
    '''
    x_str= x.columns.to_numpy()
    X = x.to_numpy()

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # 标准化
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    X_s = (X - x_mean) / x_std

    y_mean = y.mean()
    y_std = y.std()
    y_s = (y - y_mean) / y_std

    # 计算相关系数并排序
    corr = X_s.T @ y_s
    corr = [abs(x) for x in list(corr.flatten())]

    # 返回相关系数排序后对应的索引列表
    sorted_ind_corr = np.argsort(corr)[::-1]

    # 如果允许，选择前n个相关系数最大的特征
    if n < len(sorted_ind_corr):
        topn_ind = sorted_ind_corr[:n]
    else:
        topn_ind = sorted_ind_corr

    out_str = x_str[topn_ind]

    return x[out_str]