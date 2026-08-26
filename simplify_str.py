import re
import sympy as sp

def simplify(expr):
    # 替换 '^' 为 '**' 以符合 SymPy 的处理规则
    expr = expr.replace('^', '**')

    # 插入 '*' 在数字和 '(' 之间 (处理如 490.8165(alpl -> 490.8165*(alpl)
    expr = re.sub(r'(\d+(?:\.\d+)?)\(', r'\1*(', expr)

    try:
        # 使用 SymPy 解析表达式
        expression = sp.nsimplify(sp.sympify(expr))
        # 简化表达式
        simplified_expr = '(' + str(expression) + ')'
    except (sp.SympifyError, TypeError, ValueError):
        # 解析失败时返回原始表达式
        simplified_expr = '(' + expr + ')'
    
    return simplified_expr