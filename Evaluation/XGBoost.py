# coding：utf-8
import xgboost as xgb
from numpy import *
import random
import time

#数据读取-单因变量与多因变量(WYHXB)
def loadDataSet(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    row = len(arrayLines)
    x = mat(zeros((row, 6)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split('\t')
        x[index, :] = curLine[0:6]
        y[index, :] = curLine[-1]
        index += 1
    return x, y

'''
为了获得较可靠的结果，需测试数据（测验）和训练数据（学习）
--可按照6:4的比例划分数据集,即随机分成训练集与测试集
'''
def splitDataSet(x, y):
    m =shape(x)[0]
    train_sum = int(round(m * 0.6))
    test_sum = m - train_sum
    #利用range()获得样本序列
    randomData = range(0,m)
    randomData = list(randomData)
    #根据样本序列进行分割- random.sample(A,rep)
    train_List = random.sample(randomData, train_sum)
    #获取训练集数据-train
    train_x = x[train_List,: ]
    train_y = y[train_List,: ]
    #获取测试集数据-test
    test_list = []
    for i in randomData:
        if i in train_List:
            continue
        test_list.append(i)
    test_x = x[test_list,:]
    test_y = y[test_list,:]
    return train_x, train_y, test_x, test_y

# XGBoost模型建立：调参
def XGBoostModel(x0, y0):
    # XGBoost训练 -- 调参序列：max_depth，learning_rate，n_estimators--reg_lambda=0.01, learning_rate=0.018, n_estimators=500
    model = xgb.XGBRegressor(max_depth=6, silent=False, objective='reg:linear', booster='gbtree', gamma=1, base_score=0.5,
                             reg_lambda=0.01, learning_rate=0.01, n_estimators=300) #350, 500
    model.fit(x0, y0)
    row = shape(x0)[0]
    mean_y = mean(y0, 0)
    y_mean = tile(mean_y, (row, 1))

    # 对实验数据进行预测
    xgy_predict = model.predict(x0)
    xgy_predict = mat(xgy_predict).T
    XGB_SSE = sum(power((y0 - xgy_predict), 2), 0)
    XGB_SST = sum(sum(power((y0 - y_mean), 2), 0))
    XGB_SSR = sum(sum(power((xgy_predict - y_mean), 2), 0))
    XGB_RMSE = sqrt(XGB_SSE / row)  # 均方根误差
    R_squared = XGB_SSR / XGB_SST
    return XGB_RMSE, R_squared

if __name__ == '__main__':
    startime = time.clock()
    x, y = loadDataSet('0.74WY_subFeature_6.txt')
    # 训练集与测试集 - 0.6与0.4
    train_x, train_y, test_x, test_y = splitDataSet(x, y)

    XGB_RMSE, R_squared = XGBoostModel(x, y)
    print('=====================')
    print(u'基于MIC的近似马尔科夫毯--PLS：')
    # print(u'原始数据集--PLS：')
    print('R_squared:', R_squared)
    print('RMSE:', XGB_RMSE)
    print('=====================')
    endtime = time.clock()
    runtime = endtime - startime
    print(shape(x), shape(y))
    print(u'run_time: %fs' % runtime)