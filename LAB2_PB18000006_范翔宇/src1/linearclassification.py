from process_data import load_and_process_data
from evaluation import get_macro_F1,get_micro_F1,get_acc
import numpy as np


# 实现线性回归的类
class LinearClassification:

    '''参数初始化
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,lr=0.000005,Lambda= 0.001,epochs = 1000):
        self.lr=lr
        self.Lambda=Lambda
        self.epochs =epochs

    '''根据训练数据train_features,train_labels计算梯度更新参数W'''
    def fit(self,train_features,train_labels):
        ''''
        需要你实现的部分
        '''
        #start here, code by Fan Xiangyu PB18000006
        #shape[0]行数 shape[1]列数
        m = train_features.shape[0]
        n = train_features.shape[1]
        #初始化X 规格为m × (n + 1)
        X = np.c_[np.ones(m), train_features]
        #初始化W 规格为(n + 1) × 1
        W = np.ones((n + 1, 1))
        W = 0.001 * W
        #开始迭代
        for i in range(self.epochs):
            y = train_labels
            #A = XW - y
            A = np.dot(X,W) - y;
            #B = ((XW - y) ^ T)X
            B = np.dot(A.T, X)
            C = 2 * B + 2 * self.Lambda * W.reshape(1,-1)
            #print(C.shape[0],C.shape[1])
            #print(C)
            W = W - self.lr * C.T
            #print(W)
        self.w = W

    '''根据训练好的参数对测试数据test_features进行预测，返回预测结果
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目'''
    def predict(self,test_features):
        ''''
        需要你实现的部分
        '''
        #start here, code by Fan Xiangyu PB18000006
        m = test_features.shape[0]
        n = test_features.shape[1]
        #初始化
        T = np.c_[np.ones(m), test_features]
        predct = []
        for i in range(m):
            #计算，并且四舍五入加tag
            predct_num = np.dot(T[i],self.w)
            #print(predct_num)
            R = np.round(predct_num)
            predct.append(R)
        predct = np.array(predct).astype(int).reshape(-1,1)
        return predct


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    lR=LinearClassification()
    lR.fit(train_data,train_label) # 训练模型
    pred=lR.predict(test_data) # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
