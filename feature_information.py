import pandas as pd
# import numpy as np


class FeatureInformation:
    '''
    储存每个特征的信息
    '''
    def __init__(self, id:int, data: pd.DataFrame, 
                 coef=[], r2=-1, loss=0, score=0):
        '''
        初始化
        '''
        # id: 特征的id
        # data: 特征的数据
        # coef: 特征的系数(默认排序为[c1, b])
        # r2: 特征的R2
        # loss: 特征与目标量的均方误差（实际上差一个1/2）
        # score: 特征的分数
        self.__id = id
        self.__data = data
        self.__coef = [round(i, 4) for i in coef]
        self.__score = score
        self.__r2 = round(r2, 4)
        self.__loss = round(loss, 4)

    def get_name(self):
        '''
        获取特征的名称
        '''
        return self.__data.columns[0]

    def get_data(self):
        '''
        获取特征的原始数据 (未乘系数)
        '''
        return self.__data

    def get_id(self):
        '''
        获取特征的id
        '''
        return self.__id
    
    def get_score(self):
        '''
        获取特征的分数
        '''
        return self.__score

    def get_r2(self):
        '''
        获取特征的R2
        '''
        return self.__r2
    
    def update_score(self):
        '''
        更新特征的分数
        '''
        self.__score = round(1/(1.01 - self.__r2), 4)

    def set_coef(self, *coefs):
        '''
        默认排序为[c1, b]
        '''
        for coef in coefs:
            self.__coef.append(round(coef, 3))

    def set_r2(self, r2:float):
        '''
        设置特征的R2
        '''
        self.__r2 = round(r2, 4)

    def set_loss(self, loss:float):
        '''
        设置特征与目标量的均方误差
        '''
        self.__loss = round(loss, 4)

    def get_loss(self):
        '''
        获取特征与目标量的均方误差
        '''
        return self.__loss

    def get_full_name(self):
        '''
        获取特征的完整名称
        '''
        return str(self.__coef[0]) + self.__data.columns[0] + '+' + str(self.__coef[1])

    def get_full_data(self):
        '''
        获取特征的完整数据（计算系数）
        '''
        data_str = self.__data.columns[0]
        data_num = self.__data[data_str].values

        # 暂时只考虑一维情况
        out_num = self.__coef[0] * data_num + self.__coef[1]
        out_str = self.get_full_name()

        return pd.DataFrame(out_num, columns=[out_str])
    
