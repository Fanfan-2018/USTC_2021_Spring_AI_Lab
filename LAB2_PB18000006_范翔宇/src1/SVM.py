
import numpy as np
import cvxopt #用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc


#根据指定类别main_class生成1/-1标签
def svm_label(labels,main_class):
    new_label=[]
    for i in range(len(labels)):
        if labels[i]==main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)

# 实现线性回归
class SupportVectorMachine:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,kernel,C,Epsilon):
        self.kernel=kernel
        self.C = C
        self.Epsilon=Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''
    def KERNEL(self, x1, x2, kernel='Linear', d=2, sigma=1):
        #d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1,x2)
        elif kernel == 'Poly':
            K = np.dot(x1,x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''
    def fit(self,train_data,train_label,test_data):
        '''
        需要你实现的部分
        '''
        train = train_data.shape[0]
        test = test_data.shape[0]
        predct = []
        P = np.zeros((train, train))
        for i in range(train):
            for j in range(train):
                P[i][j] = self.KERNEL(train_data[i], train_data[j], self.kernel) * train_label[i] * train_label[j]
        q = -1 * np.ones((train, 1))
        G = np.r_[np.eye(train, dtype = int), -1 * np.eye(train, dtype=int)]
        h = np.r_[self.C * np.ones((train,1)), np.zeros((train, 1))]
        A = train_label.reshape(1, train)
        b = np.zeros((1, 1))
        PP = cvxopt.matrix(P.astype(np.double))
        qq = cvxopt.matrix(q.astype(np.double))
        GG = cvxopt.matrix(G.astype(np.double))
        hh = cvxopt.matrix(h.astype(np.double))
        AA = cvxopt.matrix(A.astype(np.double))
        bb = cvxopt.matrix(b.astype(np.double))
        solvers = cvxopt.solvers.qp(PP, qq, GG, hh, AA, bb)
        x = solvers['x']
        a = np.array(x)
        b = 0
        for i in range(train):
            if a[i] < self.Epsilon:
                a[i] = 0
            if a[i] < self.C and a[i] > 0:
                b = train_label[i]
                for j in range(train):
                    b -= a[j] * train_label[j] * (self.KERNEL(train_data[j], train_data[i].reshape(-1,1)))
                break
        for i in range(test):
            tmp = b
            for j in range(train):
                tmp = tmp + a[j] * train_label[j] * (self.KERNEL(train_data[j], test_data[i].reshape(-1, 1) ))
            predct.append(tmp)
        predct = np.array(predct).reshape(test, 1)
        return predct


def main():
    # 加载训练集和测试集
    Train_data,Train_label,Test_data,Test_label=load_and_process_data()
    Train_label=[label[0] for label in Train_label]
    Test_label=[label[0] for label in Test_label]
    train_data=np.array(Train_data)
    test_data=np.array(Test_data)
    test_label=np.array(Test_label).reshape(-1,1)
    #类别个数
    num_class=len(set(Train_label))


    #kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    #C为软间隔参数；
    #Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    kernel='Linear'
    C = 1
    Epsilon=10e-5
    #生成SVM分类器
    SVM=SupportVectorMachine(kernel,C,Epsilon)

    predictions = []
    #one-vs-all方法训练num_class个二分类器
    for k in range(1,num_class+1):
        #将第k类样本label置为1，其余类别置为-1
        train_label=svm_label(Train_label,k)
        # 训练模型，并得到测试集上的预测结果
        prediction=SVM.fit(train_data,train_label,test_data)
        predictions.append(prediction)
    predictions=np.array(predictions)
    #one-vs-all, 最终分类结果选择最大score对应的类别
    pred=np.argmax(predictions,axis=0)+1
    #print(predictions)
    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
