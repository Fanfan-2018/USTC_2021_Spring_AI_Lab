import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}
        #start here, code by Fan Xiangyu PB18000006
        self.upperbound={}#增加上界，便于连续属性划分离散区间
        self.lowerbound={}#增加下界，便于连续属性划分离散区间

    #自定义函数，来确定对应区间下标
    def section(self,features,i,j):
        #区间长度
        len = (self.upperbound[j] - self.lowerbound[j]) / 100
        #四舍五入取整
        index = int(round((features[i][j] - self.lowerbound[j]) / len))
        return index
    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''
    def fit(self,traindata,trainlabel,featuretype):
        '''
        需要你实现的部分
        '''
        #start here, code by Fan Xiangyu PB18000006
        #选择方法一，划分成离散区间，均选择划分100个区间，即101个可能取值
        m = traindata.shape[0] #行数对应数据个数
        n = traindata.shape[1] #列数对应属性个数
        #list = []
        #for i in range(len(trainlabel)):
        #    if trainlabel[i] not in list:
        #        list.append(trainlabel[i])
        #print(len(list))
        #for i in range(len(trainlabel)):
        #    print(trainlabel[i])
        #查找每个属性的上下界
        for i in range(n):
            tmp_list = []
            #遍历添加数据
            for j in range(m):
                tmp_list.append(traindata[j][i])
            self.lowerbound[i] = min(tmp_list)
            self.upperbound[i] = max(tmp_list)
            #print(self.lowerbound[i])
            #print(self.upperbound[i])
        self.Pxc = np.zeros((4, n, 101), dtype = float)
        # 遍历trainlabel
        for i in range(4):
            D_c = 0
            D_xc = np.zeros((n,101),dtype = int)
            # 遍历每条数据
            for j in range(m):
                if trainlabel[j][0] == i:
                    D_c += 1
                    # 遍历每个属性
                    for k in range(n):
                        # 区间索引
                        index = int(traindata[j][k])
                        # 如果是连续属性
                        if featuretype[k] == 1:
                            index = self.section(traindata, j, k);
                            # print(index)
                        D_xc[k][index] += 1
            # 先验概率
            self.Pc[i] = math.log((D_c + 1) / (m + 3))
            # 计算条件概率
            for p in range(n):
                N_i = 3
                if featuretype[p] == 1:
                    N_i = 100
                for q in range(1,N_i + 1):
                    self.Pxc[i][p][q] = math.log((D_xc[p][q]+1)/(D_c+N_i))
    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        '''
        需要你实现的部分
        '''
        #start here, code by Fan Xiangyu PB18000006
        m = features.shape[0]#行数对应数据个数
        n = features.shape[1]#列数对应属性个数
        pred = np.zeros((m,1),dtype = int)
        for i in range(m):
            mult_best = -114514
            # 遍历寻找max
            for j in range(4):
                mult_log = self.Pc[j]
                # 由于取对数，所以累乘变为累加
                for k in range(n):
                    index = int(features[i][k])
                    if featuretype[k] == 1:
                        index = self.section(features,i,k)
                        mult_log += self.Pxc[j][k][index]
                # 比较max
                if mult_log > mult_best:
                    mult_best = mult_log
                    pred[i][0] = j
        return pred


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果
    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()