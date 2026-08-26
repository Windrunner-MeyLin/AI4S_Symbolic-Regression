import pandas as pd
import numpy as np
import itertools
from numpy.typing import NDArray as ND


def expand(ori_data:pd.DataFrame):
    '''
    对数据进行扩展
    '''
    ## 初始化
    out_data = ori_data.copy()
    out_num = ori_data.to_numpy()
    out_str = ori_data.columns.to_numpy()

    ## 变换：[x1, x2, x3, ...] -> [[x1], [x2], [x3],...]
    out_num = np.hsplit(out_num, out_num.shape[1])
    out_str = np.hsplit(out_str, out_str.shape[0])

    ## 幂
    for index, name in enumerate(out_str):
        
        num_p1, str_p1 = power(out_num[index], name, 1/2)
        num_p2, str_p2 = power(out_num[index], name, 2)
        num_p3, str_p3 = power(out_num[index], name, 1/3)
        num_p4, str_p4 = power(out_num[index], name, 3)
        out_num[index] = np.hstack((out_num[index], num_p1, 
                                    num_p2, num_p3, num_p4))
        out_str[index] = np.hstack((out_str[index], str_p1,
                                    str_p2, str_p3, str_p4))
        
        num_b, str_b = power(out_num[index], out_str[index], -1)
        out_num[index] = np.hstack((out_num[index], num_b))
        out_str[index] = np.hstack((out_str[index], str_b))

    ## 组合
    # 返回所有组合的可能
    combinations = list(itertools.combinations(range(len(out_num)), 2))
    # 对每一种组合进行计算
    for combination in combinations:
        # 取出一个组合中对应特征的索引
        idx1, idx2 = combination
        # 计算组合特征
        num_c, str_c = combine(out_num[idx1], out_num[idx2],
                               out_str[idx1], out_str[idx2])
        # 将组合特征添加到结果中
        out_num.append(num_c)
        out_str.append(str_c)

    ## 整合数据
    out_num = np.hstack(out_num)
    out_str = np.hstack(out_str)
    out_data = pd.DataFrame(out_num, columns=out_str)

    ## 删除异常数据
    out_data = out_data.loc[:, ~(out_data.isna().any() | \
                                  out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]

    return out_data

def power(in_num:ND, in_name:ND, p):
    '''
    对数据进行幂运算
    '''
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)

    out_num = np.power(in_num, p)
    out_name = [f"({n}**{p})" for n in in_name]

    return out_num, out_name

def combine(in_num1:ND, in_num2:ND, in_name1:ND, in_name2:ND):
    '''
    对数据进行二元组合运算'
    '''
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)

    # 组合特征的数量
    num_datapoints = in_num1.shape[0]
    num_featurevars = in_num1.shape[1] * in_num2.shape[1]

    # 初始化组合特征矩阵和名称列表
    out_num = np.zeros((num_datapoints, num_featurevars))
    out_name = ['']*num_featurevars

    # 计算组合特征
    idx = 0
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            out_num[:, idx] = in_num1[:, i] * in_num2[:, j]
            out_name[idx] = f"({in_name1[i]}*{in_name2[j]})"
            idx += 1

    return out_num, out_name