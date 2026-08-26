"""
扩展版本的特征扩展模块 (7.6.3)
支持更多运算类型: 多种幂次、四则运算(+,-,*,/)、三元组合、一元函数(log,exp)
并记录空间膨胀程度
"""
import pandas as pd
import numpy as np
import itertools
from numpy.typing import NDArray as ND


def expand_v2(ori_data: pd.DataFrame, ops_config='full'):
    '''
    对数据进行扩展 (增强版)
    幂次/一元仅作用于原始特征，避免指数级膨胀
    '''
    out_data = ori_data.copy()
    n_orig = ori_data.shape[1]
    orig_cols = ori_data.columns.to_numpy()

    # 将原始特征拆分为独立组
    out_num_groups = [ori_data.iloc[:, i].to_numpy().reshape(-1, 1) for i in range(n_orig)]
    out_str_groups = [np.array([orig_cols[i]]) for i in range(n_orig)]

    stats = {'initial': n_orig}

    # ========== 一、幂次变换 (仅作用于原始特征) ==========
    powers = [2, 3, -1, -2, -3, 0.5, 1/3, 2/3, 3/2]
    for index in range(n_orig):
        base_num = ori_data.iloc[:, index].to_numpy().reshape(-1, 1)
        base_str = np.array([orig_cols[index]])
        for p in powers:
            num_p, str_p = power(base_num, base_str, p)
            out_num_groups[index] = np.hstack((out_num_groups[index], num_p))
            out_str_groups[index] = np.hstack((out_str_groups[index], str_p))

    dim_after_power = sum(arr.shape[1] for arr in out_num_groups)
    stats['after_power'] = dim_after_power

    # ========== 二、一元函数变换 (仅作用于原始特征) ==========
    unary_ops = [
        ('log', lambda x: np.log(np.abs(x) + 1e-10)),
        ('exp', lambda x: np.exp(np.clip(x, -10, 10))),
        ('sqrt', lambda x: np.sqrt(np.abs(x))),
    ]
    for index in range(n_orig):
        base_num = ori_data.iloc[:, index].to_numpy().reshape(-1, 1)
        base_str = np.array([orig_cols[index]])
        for op_name, op_func in unary_ops:
            num_u, str_u = unary_op(base_num, base_str, op_func, op_name)
            out_num_groups[index] = np.hstack((out_num_groups[index], num_u))
            out_str_groups[index] = np.hstack((out_str_groups[index], str_u))

    dim_after_unary = sum(arr.shape[1] for arr in out_num_groups)
    stats['after_unary'] = dim_after_unary

    # ========== 三、二元组合 (四则运算) ==========
    binary_ops = [
        ('+', lambda a, b: a + b),
        ('-', lambda a, b: a - b),
        ('*', lambda a, b: a * b),
        ('/', lambda a, b: a / (b + 1e-10)),
    ]

    combinations = list(itertools.combinations(range(n_orig), 2))
    for idx1, idx2 in combinations:
        for op_name, op_func in binary_ops:
            num_c, str_c = combine_op(out_num_groups[idx1], out_num_groups[idx2],
                                      out_str_groups[idx1], out_str_groups[idx2],
                                      op_func, op_name)
            out_num_groups.append(num_c)
            out_str_groups.append(str_c)

    dim_after_binary = sum(arr.shape[1] for arr in out_num_groups)
    stats['after_binary'] = dim_after_binary

    # ========== 四、三元组合 (乘法为主) ==========
    tri_combinations = list(itertools.combinations(range(n_orig), 3))
    for comb in tri_combinations[:20]:
        idx1, idx2, idx3 = comb
        num_c, str_c = combine_3(out_num_groups[idx1], out_num_groups[idx2], out_num_groups[idx3],
                                 out_str_groups[idx1], out_str_groups[idx2], out_str_groups[idx3])
        out_num_groups.append(num_c)
        out_str_groups.append(str_c)

    dim_after_ternary = sum(arr.shape[1] for arr in out_num_groups)
    stats['after_ternary'] = dim_after_ternary

    # ========== 五、五元组合 (a*b*c/(d*e)) ==========
    # 每个变量取原始列 + 关键幂次列, 逐位变化避免组合爆炸
    key_power_indices = [0, 1, 3, 6, 7]  # 原始, ^2, ^-1, ^0.5, ^(1/3)
    # 先加所有原始版本 (全部用第0列)
    for triple in itertools.combinations(range(n_orig), 3):
        nums = list(triple)
        dens = [i for i in range(n_orig) if i not in nums]
        n1 = out_num_groups[nums[0]][:, 0:1];  s1 = out_str_groups[nums[0]][0:1]
        n2 = out_num_groups[nums[1]][:, 0:1];  s2 = out_str_groups[nums[1]][0:1]
        n3 = out_num_groups[nums[2]][:, 0:1];  s3 = out_str_groups[nums[2]][0:1]
        n4 = out_num_groups[dens[0]][:, 0:1];  s4 = out_str_groups[dens[0]][0:1]
        n5 = out_num_groups[dens[1]][:, 0:1];  s5 = out_str_groups[dens[1]][0:1]
        num_c, str_c = combine_5(n1, n2, n3, n4, n5, s1, s2, s3, s4, s5)
        out_num_groups.append(num_c)
        out_str_groups.append(str_c)
    # 再对每个分子位置单独做关键幂次变体 (避开全笛卡尔积)
    for triple in itertools.combinations(range(n_orig), 3):
        nums = list(triple)
        dens = [i for i in range(n_orig) if i not in nums]
        for var_pos in [0, 1, 2]:  # 对3个分子位置各做幂次变化
            for p_idx in key_power_indices[1:]:  # 跳过 p_idx=0 (原始版已在上面加了)
                groups = [None] * 5
                strs = [None] * 5
                for i, idx in enumerate(nums):
                    if i == var_pos:
                        groups[i] = out_num_groups[idx][:, p_idx:p_idx+1]
                        strs[i] = out_str_groups[idx][p_idx:p_idx+1]
                    else:
                        groups[i] = out_num_groups[idx][:, 0:1]
                        strs[i] = out_str_groups[idx][0:1]
                groups[3] = out_num_groups[dens[0]][:, 0:1]
                strs[3] = out_str_groups[dens[0]][0:1]
                groups[4] = out_num_groups[dens[1]][:, 0:1]
                strs[4] = out_str_groups[dens[1]][0:1]
                num_c, str_c = combine_5(*groups, *strs)
                out_num_groups.append(num_c)
                out_str_groups.append(str_c)

    dim_after_quintic = sum(arr.shape[1] for arr in out_num_groups)
    stats['after_quintic'] = dim_after_quintic

    # ========== 整合数据 ==========
    out_num = np.hstack(out_num_groups)
    out_str = np.hstack(out_str_groups)
    out_data = pd.DataFrame(out_num, columns=out_str)

    # 删除异常和重复
    n_before_clean = out_data.shape[1]
    out_data = out_data.loc[:, ~(out_data.isna().any() | out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]
    stats['after_clean'] = out_data.shape[1]
    stats['removed'] = n_before_clean - out_data.shape[1]

    return out_data, stats


def expand_with_tracking(ori_data: pd.DataFrame, strategy_name: str):
    '''
    按指定策略扩展并返回空间膨胀统计
    '''
    if strategy_name == 'power_only':
        out_data = ori_data.copy()
        out_num = ori_data.to_numpy()
        out_str = ori_data.columns.to_numpy()
        out_num = np.hsplit(out_num, out_num.shape[1])
        out_str = np.hsplit(out_str, out_str.shape[0])
        powers = [2, 3, -1, -2, -3, 0.5, 1/3, 2/3, 3/2]
        for index in range(len(out_str)):
            for p in powers:
                num_p, str_p = power(out_num[index], out_str[index], p)
                out_num[index] = np.hstack((out_num[index], num_p))
                out_str[index] = np.hstack((out_str[index], str_p))
        out_num = np.hstack(out_num)
        out_str = np.hstack(out_str)
        out_data = pd.DataFrame(out_num, columns=out_str)

    elif strategy_name == 'binary_only':
        n = ori_data.shape[1]
        out_num_list, out_str_list = [], []
        for i in range(n):
            col = ori_data.iloc[:, i].to_numpy().reshape(-1, 1)
            out_num_list.append(col)
            out_str_list.append(np.array([ori_data.columns[i]]))
        combinations = list(itertools.combinations(range(n), 2))
        for idx1, idx2 in combinations:
            for op_name, op_func in [('+', lambda a, b: a + b), ('-', lambda a, b: a - b),
                                      ('*', lambda a, b: a * b), ('/', lambda a, b: a / (b + 1e-10))]:
                n1, s1 = out_num_list[idx1], out_str_list[idx1]
                n2, s2 = out_num_list[idx2], out_str_list[idx2]
                num_c, str_c = combine_op(n1, n2, s1, s2, op_func, op_name)
                out_num_list.append(num_c)
                out_str_list.append(str_c)
        out_num = np.hstack(out_num_list)
        out_str = np.hstack(out_str_list)
        out_data = pd.DataFrame(out_num, columns=out_str)

    elif strategy_name == 'full':
        out_data, _ = expand_v2(ori_data)

    else:
        out_data, _ = expand_v2(ori_data)

    out_data = out_data.loc[:, ~(out_data.isna().any() | out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]
    return out_data


def power(in_num: ND, in_name: ND, p):
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)
    out_num = np.power(in_num, p)
    out_name = [f"({n}**{p})" for n in in_name]
    return out_num, out_name


def unary_op(in_num: ND, in_name: ND, op_func, op_name: str):
    if len(in_num.shape) == 1:
        in_num = in_num.reshape(-1, 1)
    out_num = op_func(in_num)
    out_name = [f"{op_name}({n})" for n in in_name]
    return out_num, out_name


def combine_op(in_num1: ND, in_num2: ND, in_name1: ND, in_name2: ND,
               op_func, op_name: str):
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)
    n_features = in_num1.shape[1] * in_num2.shape[1]
    out_num_list, out_name_list = [], []
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            out_num_list.append(op_func(in_num1[:, i], in_num2[:, j]).reshape(-1, 1))
            out_name_list.append(f"({in_name1[i]}{op_name}{in_name2[j]})")
    out_num = np.hstack(out_num_list)
    out_name = np.array(out_name_list)
    return out_num, out_name


def combine_3(in_num1: ND, in_num2: ND, in_num3: ND,
              in_name1: ND, in_name2: ND, in_name3: ND):
    if len(in_num1.shape) == 1:
        in_num1 = in_num1.reshape(-1, 1)
    if len(in_num2.shape) == 1:
        in_num2 = in_num2.reshape(-1, 1)
    if len(in_num3.shape) == 1:
        in_num3 = in_num3.reshape(-1, 1)
    n_features = in_num1.shape[1] * in_num2.shape[1] * in_num3.shape[1]
    out_num_list, out_name_list = [], []
    for i in range(in_num1.shape[1]):
        for j in range(in_num2.shape[1]):
            for k in range(in_num3.shape[1]):
                out_num_list.append((in_num1[:, i] * in_num2[:, j] * in_num3[:, k]).reshape(-1, 1))
                out_name_list.append(f"({in_name1[i]}*{in_name2[j]}*{in_name3[k]})")
    out_num = np.hstack(out_num_list)
    out_name = np.array(out_name_list)
    return out_num, out_name


def combine_5(in_num1: ND, in_num2: ND, in_num3: ND,
              in_num4: ND, in_num5: ND,
              in_name1: ND, in_name2: ND, in_name3: ND,
              in_name4: ND, in_name5: ND):
    '''五元组合: (a*b*c)/(d*e) — 每组仅1列'''
    for arr in [in_num1, in_num2, in_num3, in_num4, in_num5]:
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
    numerator = in_num1 * in_num2 * in_num3
    denominator = in_num4 * in_num5 + 1e-10
    out_num = (numerator / denominator).reshape(-1, 1)
    out_name = np.array([f"({in_name1[0]}*{in_name2[0]}*{in_name3[0]}/{in_name4[0]}/{in_name5[0]})"])
    return out_num, out_name
