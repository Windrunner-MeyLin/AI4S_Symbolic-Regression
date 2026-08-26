from simplify_str import simplify
from feature_information import FeatureInformation
import pandas as pd
import numpy as np
from numpy.typing import NDArray as ND


def sort_result(num_in:ND, str_in:ND, r2:list, p:list, 
                loss:list, n:int, num=0):
    # 初始化
    str_sp = np.empty(len(str_in), dtype=object)

    # 简化字符串
    for index in range(len(str_in)):
        str_sp[index] = simplify(str_in[index])

    # 根据r2排序
    sortlist = sorted(enumerate(r2), key=lambda x: x[1], reverse=True)
    # 得到排序后的索引
    idx_sort = [x[0] for x in sortlist] 

    # 根据排序后的索引整理输出
    str_sp = str_sp[idx_sort]
    num_in = num_in[:, idx_sort]
    r2_out = [r2[i] for i in idx_sort] 
    p_out = [p[i] for i in idx_sort]
    loss_out = [loss[i] for i in idx_sort]

    list_out = []
    for index in range(min(n, len(str_sp))):
        # 将结果存入FeatureInformation类中
        list_out.append(FeatureInformation(index+1+num, 
            pd.DataFrame(num_in[:, index], columns=[str_sp[index]]),
            p_out[index], r2_out[index], loss_out[index]))
        # 更新分数
        list_out[index].update_score()

    return list_out