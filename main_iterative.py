"""
循环探索框架 (7.6.4)
迭代: 特征生成 -> 筛选 -> 拟合 -> 评估 -> 指导下一次特征生成
"""
import pandas as pd
import numpy as np
import torch
import os
import itertools
from collections import deque

from expansion_v2 import expand_v2, power, combine_op, unary_op
from sis_ana import sis
from coefficients_fitting import fit
from result_sorting import sort_result
from result_displaying import display_result
from timer import Timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FormulaNode:
    '''树状结构储存每次迭代的公式信息'''
    def __init__(self, expression, r2, loss, coef, iteration, parent=None):
        self.expression = expression
        self.r2 = r2
        self.loss = loss
        self.coef = coef
        self.iteration = iteration
        self.parent = parent
        self.children = []
        self.score = round(1 / (1.01 - r2), 4) if r2 < 1 else 9999

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"[Iter{self.iteration}] R²={self.r2:.4f}: {self.expression}"


class FormulaTree:
    '''管理所有公式节点的树'''
    def __init__(self):
        self.nodes = []
        self.best_node = None

    def add_node(self, expression, r2, loss, coef, iteration, parent=None):
        node = FormulaNode(expression, r2, loss, coef, iteration, parent)
        self.nodes.append(node)
        if parent is not None:
            parent.add_child(node)
        if self.best_node is None or r2 > self.best_node.r2:
            self.best_node = node
        return node

    def get_best_expression(self):
        return self.best_node.expression if self.best_node else None

    def get_history(self):
        return [(n.iteration, n.r2, n.expression) for n in self.nodes]


def expand_next(data_orig, best_features, n_orig=5):
    '''
    根据上轮最优特征，指导下一次扩展
    策略: 仅对原始特征做 expand_v2, 再独立组合最佳特征, 避免 N>5 时组合爆炸
    '''
    # 1. 提取前 n_orig 列（原始特征）做 v2 扩展
    data_base = data_orig.iloc[:, :min(n_orig, data_orig.shape[1])]
    expanded_base, _ = expand_v2(data_base)
    result_list = [expanded_base]

    # 2. 对每个最佳特征做简单变换并与原始特征交叉组合
    for feat_df in best_features[:2]:
        feat_series = feat_df.to_numpy().flatten()
        feat_name = str(feat_df.columns[0])

        powers = [2, 3, 0.5, -1]
        for p in powers:
            result_list.append(pd.DataFrame(
                np.power(np.abs(feat_series) + 1e-10, p).reshape(-1, 1),
                columns=[f"({feat_name}**{p})"]
            ))

        for i in range(n_orig):
            orig_col = data_base.iloc[:, i].to_numpy().flatten()
            orig_name = data_base.columns[i]
            for op, op_func in [('*', lambda a, b: a * b), ('+', lambda a, b: a + b),
                                ('-', lambda a, b: a - b), ('/', lambda a, b: a / (b + 1e-10))]:
                val = op_func(feat_series, orig_col)
                result_list.append(pd.DataFrame(
                    val.reshape(-1, 1),
                    columns=[f"({feat_name}{op}{orig_name})"
                ]))

    expanded = pd.concat(result_list, axis=1)
    expanded = expanded.loc[:, ~(expanded.isna().any() | expanded.apply(np.isinf).any())]
    expanded = expanded.loc[:, ~expanded.columns.duplicated()]
    return expanded


def expand_progressive(data_orig, best_feature_dfs, iteration, top_k=3):
    '''
    渐进式复杂度增长扩展 (针对原7.6.4第1轮全量扩展导致迭代无效的问题)

    策略: 避免log/exp等"死胡同"运算, 优先放开展开目标公式结构的关键运算
      Iter 1: 整数幂次 [2,3,-1,-2] + 原始特征间二元四则运算 (×,÷,+,-)
               → 发现 gam/ras (R²≈0.44)
      Iter 2: + sqrt 一元函数 + Best × 原始特征 链式组合
               → 发现 sqrt(gam)/sqrt(ras) (R²≈0.46) 或 alpl*gam/ras (R²≈0.78)
      Iter 3: + 分数幂次 [0.5,1/3,2/3,3/2] + Best × 原始₁ × 原始₂ 三元组合
               → 发现 alpl^(1/3)*gam*omeg/(ras*kT) (R²≈0.94)
      Iter 4+: 更多链式扩展 (Best × 原始 继续组合)
    '''
    n_orig = data_orig.shape[1]
    raw_cols = data_orig.columns.to_numpy()
    groups_num = [data_orig.iloc[:, i].to_numpy().reshape(-1, 1) for i in range(n_orig)]
    groups_str = [np.array([raw_cols[i]]) for i in range(n_orig)]

    int_powers = [2, 3, -1, -2]
    # 仅用乘除避免"作弊"：+- 会创造线性组合特征让单变量OLS看似很好
    mul_div_ops = [
        ('*', lambda a, b: a * b),
        ('/', lambda a, b: a / (b + 1e-10)),
    ]

    # ========== 通用: 整数幂次 (所有迭代都做) ==========
    for i in range(n_orig):
        base_n = groups_num[i]
        base_s = groups_str[i]
        for p in int_powers:
            num_p, str_p = power(base_n, base_s, p)
            groups_num[i] = np.hstack((groups_num[i], num_p))
            groups_str[i] = np.hstack((groups_str[i], str_p))

    # ========== 原始特征间乘除组合 (仅Iter 1, 之后靠 pool×raw 自然生长深度) ==========
    if iteration == 1:
        for idx1 in range(n_orig):
            for idx2 in range(idx1 + 1, n_orig):
                g1_n = groups_num[idx1]
                g1_s = groups_str[idx1]
                g2_n = groups_num[idx2]
                g2_s = groups_str[idx2]
                for op_name, op_func in mul_div_ops:
                    num_c, str_c = combine_op(
                        g1_n[:, 0:1], g2_n[:, 0:1],
                        g1_s[0:1], g2_s[0:1],
                        op_func, op_name
                    )
                    groups_num.append(num_c)
                    groups_str.append(str_c)

    # ========== Iter 2+: sqrt 一元函数 (原版 sqrt(gam)/sqrt(ras) 需要它) ==========
    if iteration >= 2:
        for i in range(n_orig):
            base_n = groups_num[i][:, 0:1]
            base_s = groups_str[i][0:1]
            num_u, str_u = unary_op(base_n, base_s,
                                     lambda x: np.sqrt(np.abs(x)), 'sqrt')
            groups_num.append(num_u)
            groups_str.append(str_u)

    # ========== Iter 2+: Best ×/÷ 原始特征 链式组合 ==========
    # 使用全部池中特征 (避免新特征排在dict末尾被[:top_k]截断)
    if iteration >= 2 and best_feature_dfs:
        for feat_df in best_feature_dfs:
            feat_val = feat_df.to_numpy().reshape(-1, 1)
            feat_name = str(feat_df.columns[0])
            for i in range(n_orig):
                raw_val = data_orig.iloc[:, i].to_numpy().reshape(-1, 1)
                raw_name = raw_cols[i]
                for op_name, op_func in mul_div_ops:
                    val = op_func(feat_val, raw_val)
                    groups_num.append(val)
                    groups_str.append(np.array([f"({feat_name}{op_name}{raw_name})"]))
                    # 反向: raw ±×÷ best
                    if op_name in ['*', '/']:
                        val2 = op_func(raw_val, feat_val)
                        groups_num.append(val2)
                        groups_str.append(np.array([f"({raw_name}{op_name}{feat_name})"]))

    # ========== Iter 3+: 分数幂次 (alpl^(1/3) 这时才出现) ==========
    # 立即与原始特征 + 池中特征做乘除组合, 以直接构造多变量公式
    frac_powers = [0.5, 1/3, 2/3, 3/2]
    frac_cols = []  # 存 (name, values)
    if iteration >= 3:
        for i in range(n_orig):
            base_n = groups_num[i][:, 0:1]
            base_s = groups_str[i][0:1]
            for p in frac_powers:
                num_p, str_p = power(base_n, base_s, p)
                groups_num.append(num_p)
                groups_str.append(str_p)
                frac_cols.append((str_p[0], num_p))

        # 分数幂 ×/÷ 原始特征
        for frac_name, frac_val in frac_cols:
            for i in range(n_orig):
                raw_val = data_orig.iloc[:, i].to_numpy().reshape(-1, 1)
                raw_name = raw_cols[i]
                for op_name, op_func in mul_div_ops:
                    val = op_func(frac_val, raw_val)
                    groups_num.append(val)
                    groups_str.append(np.array([f"({frac_name}{op_name}{raw_name})"]))
                    val2 = op_func(raw_val, frac_val)
                    groups_num.append(val2)
                    groups_str.append(np.array([f"({raw_name}{op_name}{frac_name})"]))

        # 关键: 分数幂 ×/÷ 池中特征 (直接构建高变量数公式)
        for frac_name, frac_val in frac_cols:
            for feat_df in best_feature_dfs:
                pool_val = feat_df.to_numpy().reshape(-1, 1)
                pool_name = str(feat_df.columns[0])
                for op_name, op_func in mul_div_ops:
                    val = op_func(frac_val, pool_val)
                    groups_num.append(val)
                    groups_str.append(np.array([f"({frac_name}{op_name}{pool_name})"]))
                    val2 = op_func(pool_val, frac_val)
                    groups_num.append(val2)
                    groups_str.append(np.array([f"({pool_name}{op_name}{frac_name})"]))

    # ========== Iter 3+: Best × 原始₁ × 原始₂ 三元组合 (仅乘法) ==========
    if iteration >= 3 and best_feature_dfs:
        for feat_df in best_feature_dfs:
            f_val = feat_df.to_numpy().reshape(-1, 1)
            f_name = str(feat_df.columns[0])
            for j in range(n_orig):
                r1 = data_orig.iloc[:, j].to_numpy().reshape(-1, 1)
                r1_name = raw_cols[j]
                for k in range(j + 1, n_orig):
                    r2 = data_orig.iloc[:, k].to_numpy().reshape(-1, 1)
                    r2_name = raw_cols[k]
                    val = f_val * r1 * r2
                    label = f"({f_name}*{r1_name}*{r2_name})"
                    groups_num.append(val)
                    groups_str.append(np.array([label]))

    # ========== 合并与清理 ==========
    out_num = np.hstack(groups_num)
    out_str = np.hstack(groups_str)
    out_data = pd.DataFrame(out_num, columns=out_str)
    out_data = out_data.loc[:, ~(out_data.isna().any() | out_data.apply(np.isinf).any())]
    out_data = out_data.loc[:, ~out_data.columns.duplicated()]
    return out_data


def run_iterative_search_progressive(data, y_arr, max_iterations=8,
                                      sis_n=30, top_k=5,
                                      window_size=3, patience=3):
    '''
    渐进式迭代搜索 (针对原版第1轮就已穷举的问题)
    
    关键设计:
      1. 二元运算仅用乘÷, 避免加减产生"作弊"线性组合特征
      2. 用 raw_feature_pool 存储每轮原始特征数据 (不带拟合系数),
         避免系数被带入下一轮导致 sympy 展开出 (a*x+b)*y = a*x*y + b*y
      3. 每轮系统性地用当前池中特征 ×/÷ 原始特征, 逐步生长出多变量公式
    '''
    tree = FormulaTree()
    history_r2 = []
    y_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

    # raw_feature_pool: 存储原始特征数据 (不带拟合系数)
    # key=特征名, value=单列DataFrame
    raw_feature_pool = {}

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")

        # 当前池中特征 (新特征优先: 倒序确保最复杂的特征先被使用)
        current_pool = list(reversed(list(raw_feature_pool.values())))

        # 1. 渐进式扩展
        expanded = expand_progressive(data, current_pool, iteration, top_k=5)
        print(f"  扩展后特征数: {expanded.shape[1]}")

        # 2. SIS 筛选
        expanded_sis = sis(expanded, y_2d, sis_n)
        print(f"  SIS筛选后: {expanded_sis.shape[1]}")

        # 3. 拟合
        r2_list, coef_list, loss_list = fit(expanded_sis, y_2d, device)

        # 4. 排序
        results = sort_result(expanded_sis.to_numpy(), expanded_sis.columns.to_numpy(),
                              r2_list, coef_list, loss_list, top_k)

        best_r2 = results[0].get_r2()
        history_r2.append(best_r2)
        print(f"  最佳 R² = {best_r2:.4f}: {results[0].get_full_name()}")

        # 5. 存储到树 + 将本轮最优特征的原始数据加入池中
        parent_node = None
        for res in results[:top_k]:
            expr = res.get_full_name()
            r2_val = res.get_r2()
            loss_val = res.get_loss()

            if parent_node is None:
                node = tree.add_node(expr, r2_val, loss_val, None, iteration)
                parent_node = node
            else:
                node = tree.add_node(expr, r2_val, loss_val, None, iteration, parent_node)

            # 关键: 存储 RAW 特征数据 (不带拟合系数)
            raw_data = res.get_data()       # 只含原始特征列, 无系数
            raw_name = res.get_name()       # 特征原始名称 (如 "(gam/ras)*alpl")
            if raw_name not in raw_feature_pool:
                raw_feature_pool[raw_name] = raw_data

        # 6. 检查终止条件
        if stop_or_not(history_r2, window_size, patience, min_r2=0.98):
            print(f"\n  终止条件满足, 在第{iteration}轮停止")
            break

    print(f"\n  累积原始特征池大小: {len(raw_feature_pool)}")
    return tree, history_r2


def construct_feature_from_input(data_orig, best_series, best_name):
    '''
    将最优输出特征与输入特征进行组合
    策略: 将上轮最优特征 (作为新变量) 与原始输入进行交叉组合
    '''
    n = data_orig.shape[1]
    out_num_list = [data_orig.to_numpy()]
    out_str_list = list(data_orig.columns)

    # 添加最优特征
    best_col = best_series.reshape(-1, 1)
    best_label = best_name

    # 幂次变换
    powers = [2, 3, 0.5, -1]
    for p in powers:
        out_num_list.append(np.power(best_col, p))
        out_str_list.append(f"({best_label}**{p})")

    # 与每个原始特征组合
    for i in range(n):
        orig_col = data_orig.iloc[:, i].to_numpy().reshape(-1, 1)
        orig_name = data_orig.columns[i]

        for op, op_name in [('*', lambda a, b: a * b), ('+', lambda a, b: a + b),
                            ('-', lambda a, b: a - b), ('/', lambda a, b: a / (b + 1e-10))]:
            val = op_name(best_col, orig_col)
            out_num_list.append(val)
            out_str_list.append(f"({best_label}{op}{orig_name})")

    out_df = pd.DataFrame(np.hstack(out_num_list), columns=out_str_list)
    out_df = out_df.loc[:, ~(out_df.isna().any() | out_df.apply(np.isinf).any())]
    out_df = out_df.loc[:, ~out_df.columns.duplicated()]
    return out_df


def stop_or_not(history_r2, window_size=3, patience=3, min_r2=0.98):
    '''
    判断是否终止迭代
    history_r2: R²历史记录列表
    window_size: 滑动窗口大小
    patience: 连续无提升次数
    min_r2: 目标R²
    '''
    if len(history_r2) < window_size + patience:
        return False

    # 达到目标
    if max(history_r2) >= min_r2:
        return True

    # 滑动窗口检测: 最近 window_size 轮 vs 之前的 window_size 轮
    recent = history_r2[-window_size:]
    earlier = history_r2[-(window_size + patience):-patience]

    if len(earlier) >= window_size and len(recent) >= window_size:
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        if recent_avg <= earlier_avg:
            return True

    # 无提升检测
    if len(history_r2) >= patience + 1:
        if history_r2[-1] <= history_r2[-(patience + 1)]:
            return True

    return False


def run_iterative_search(data, y_arr, max_iterations=10,
                         sis_n=20, top_k=5, window_size=3, patience=3):
    '''
    执行循环探索
    data: DataFrame (输入特征)
    y_arr: 1D或2D array (目标值)
    '''
    tree = FormulaTree()
    history_r2 = []
    current_data = data.copy()
    y_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")

        # 1. 扩展
        if iteration == 1:
            expanded, _ = expand_v2(current_data)
        else:
            expanded = expand_next(current_data, best_features)

        print(f"  扩展后特征数: {expanded.shape[1]}")

        # 2. SIS 筛选
        expanded_sis = sis(expanded, y_2d, sis_n)
        print(f"  SIS筛选后: {expanded_sis.shape[1]}")

        # 3. 拟合
        r2_list, coef_list, loss_list = fit(expanded_sis, y_2d, device)

        # 4. 排序
        results = sort_result(expanded_sis.to_numpy(), expanded_sis.columns.to_numpy(),
                              r2_list, coef_list, loss_list, top_k)

        best_r2 = results[0].get_r2()
        history_r2.append(best_r2)
        print(f"  最佳 R² = {best_r2:.4f}: {results[0].get_full_name()}")

        # 5. 存储到树
        parent_node = None
        best_features = []
        for res in results[:top_k]:
            expr = res.get_full_name()
            r2_val = res.get_r2()
            loss_val = res.get_loss()

            if parent_node is None:
                node = tree.add_node(expr, r2_val, loss_val, None, iteration)
                parent_node = node
            else:
                node = tree.add_node(expr, r2_val, loss_val, None, iteration, parent_node)

            # 保存前3个最佳特征数据用于下次迭代
            if len(best_features) < 3:
                best_features.append(res.get_full_data())

        # 6. 检查终止条件
        if stop_or_not(history_r2, window_size, patience, min_r2=0.98):
            print(f"\n终止条件满足, 在第{iteration}轮停止")
            break

        # 7. 用最优特征构建新输入
        best_1d = best_features[0].to_numpy().flatten()
        best_expr = results[0].get_full_name()
        current_data = construct_feature_from_input(data, best_1d, best_expr)

    return tree, history_r2


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(path, "data.csv"))
    focus = pd.read_csv(os.path.join(path, "focus.csv"))

    print("=" * 60)
    print("渐进式迭代搜索 (7.6.4改进版)")
    print("=" * 60)
    tree, history = run_iterative_search_progressive(
        data, focus.to_numpy(),
        max_iterations=8, sis_n=30, top_k=5
    )

    print("\n" + "=" * 60)
    print("搜索历史")
    print("=" * 60)
    for it, r2, expr in tree.get_history():
        print(f"  Iter {it}: R²={r2:.4f}  {expr}")

    print(f"\n最佳表达式: {tree.get_best_expression()}")
    print(f"R² 迭代历史: {[round(h, 4) for h in history]}")
