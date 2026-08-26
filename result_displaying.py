from feature_information import FeatureInformation
import os


def display_result(results: list[FeatureInformation], n: int, path):
    '''
    输出结果文件
    '''

    log = ''
    str_show = [['id', 'r2', 'loss', 'score', 'model']]

    # 输出前n个结果
    for ind in range(min(n, len(results))):
        result = results[ind]
        str_show.append([result.get_id(), result.get_r2(), 
                         result.get_loss(), result.get_score(), 
                         result.get_full_name()])
        
    # 一行一行写入log中
    for line in str_show:
        log += '\n{:<5}{:<15}{:<15}{:<15}{}\n'.format(*line)

    # 输出log
    log_path = os.path.join(path, 'log.log')
    with open(log_path, 'w', encoding='utf-8') as file:
        file.write(log)