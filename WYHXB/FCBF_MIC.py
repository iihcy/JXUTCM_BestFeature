# coding:utf-8
#  _author_= Hcy Aliked
# Coding Time: 2018.12.19

from numpy import *
import time
from minepy import MINE
from sklearn import preprocessing
import xlwt
import re
import pandas as pd


# 实验数据读取--单自变量与多因变量
def loadDataSet(filename):
    file = open(filename)
    fRLines = file.readlines()
    row = len(fRLines)
    x = mat(zeros((row, 798)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in fRLines:
        curLine = line.strip().split('\t')
        x[index, :] = curLine[0: 798]
        y[index, :] = curLine[-1]
        index += 1
    return x, y


# 写文件-保存得分结果
def writeFile(x_index, scores):
    x_indexs = mat(x_index)
    x_names = []
    # 特征列表名处理
    for j in range(x_indexs.shape[1]):
        temp_index = x_indexs[:, j] + 1
        x_name = "x" + str(temp_index)
        if (len(x_name) == 6):
            x_name = x_name[0] + x_name[3]
            x_names.append(x_name)
        if (len(x_name) == 7):
            x_name = x_name[0] + x_name[3:5]
            x_names.append(x_name)
        if (len(x_name) == 8):
            x_name = x_name[0] + x_name[3:6]
            x_names.append(x_name)
    # 数组合并函数vstack与hstack
    x_names = array(x_names).T
    scores = array(scores).T
    filewrite = vstack((x_names, scores))
    filewrite = mat(filewrite).T
    # 文件写入-保存
    listMatrix = list(filewrite)
    fileE = xlwt.Workbook()
    sheet = fileE.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(len(listMatrix)):
        # write(i, 0, re.sub('[\[\]\']+', '', str(list[i])))
        sheet.write(i, 0, re.sub('[\[\]\']+', '', str(listMatrix[i])))
    fileE.save(u'data/每个特征的MIC值.xls')


# 计算单个特征与因变量、两特征之间的MIC值
def mine_Feature(InA, InB):
    InA = preprocessing.scale(InA)
    InB = preprocessing.scale(InB)
    m = InA.shape[0]
    temp_InA = array(InA)
    temp_InB = array(InB)
    temp_InA = temp_InA.reshape(m)
    temp_InB = temp_InB.reshape(m)
    mine = MINE()
    mine.compute_score(temp_InA, temp_InB)
    score = mine.mic()
    return score


# 计算多个特征与因变量相关性
def corr_Feature(x, y):
    # 遍历每个特征值序列
    x_shape = x.shape[1]
    scores_xy = []
    x_index = []
    for i in range(x_shape):
        temp_x = x[:, i]
        score_xy = mine_Feature(temp_x, y)
        scores_xy.append(score_xy)
        x_index.append(i)
    # 将结果写进文件
    writeFile(x_index, scores_xy)
    # 得分排序-返回索引
    scores = list(scores_xy)
    sort_scores = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    sort_scores = mat(sort_scores).T
    # 片段截取，去掉分数低的特征(即去掉排在最后的30%特征)--初始0.7（参数一）
    feature_num = x.shape[1]
    temp_m = int(feature_num * 0.88)
    Subset_scores = mat(sort_scores[0: temp_m]) + 1  # 输出558个特征子集
    # 根据Subset_scores的索引匹配特征名 -- 获取特征子集
    Allsub_name = []
    Testd = Subset_scores.shape[0]  # 维度
    for k in range(Subset_scores.shape[0]):
        temp_subset = Subset_scores[k, :]
        sub_name = 'x' + str(temp_subset)
        if (len(sub_name) == 6):
            sub_names = sub_name[0] + sub_name[3]
            Allsub_name.append(sub_names)
        if (len(sub_name) == 7):
            sub_names = sub_name[0] + sub_name[3:5]
            Allsub_name.append(sub_names)
        if (len(sub_name) == 8):
            sub_names = sub_name[0] + sub_name[3:6]
            Allsub_name.append(sub_names)
    # 获取特征子集列表名
    MINE_subFeature = mat(Allsub_name).T
    mat_str = array('mat')
    all_mat = vstack((mat_str, MINE_subFeature))
    # 选出的特征名——文件存储
    file = xlwt.Workbook()
    sheet1 = file.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    sel_row = len(all_mat)
    for i in range(sel_row):
        sheet1.write(i, 0, re.sub('[\[\]\']+', '', str(all_mat[i])))
    file.save('data/Feature_Submat.xls')
    return scores_xy, MINE_subFeature


# MIC选取的特征子集与匹配
def data_Matching(original_Data):
    # 获取原始数据
    allData = original_Data
    ##获取所有的列名
    allcolName = allData.columns.values.tolist()
    m = allData.shape[1]
    # 特征选择后的列名
    selectData = pd.read_excel("data/Feature_Submat.xls")
    selectData = mat(selectData)
    n = selectData.shape[0]
    selectData = mat(selectData)
    AllselectDatas = []
    # 匹配数据
    for i in range(n):
        f_s = selectData[i, :]
        f_str = str(f_s)
        # 正则化去除无关符号
        s_name = re.sub('[[\\]]', '', f_str)
        sJ_Name = re.sub('\'', '', s_name)
        for j in range(m):
            allNameI = allcolName[j]
            if allNameI == sJ_Name:
                allData = mat(allData)
                fs_data = allData[:, j]
                AllselectDatas.append(fs_data)
    # 获取MIC后的特征子集--针对X矩阵
    MIC_selectX = array(mat(array(AllselectDatas)).T)
    selectDataX = array(selectData).T
    # 拼接数组--整理无关特征剔除后的数据
    MIC_subFeatures = vstack((selectDataX, MIC_selectX))
    MIC_subFeatures = mat(MIC_subFeatures)
    # 导出MIC后的数据子集
    file = pd.DataFrame(MIC_subFeatures)
    file.to_csv(u'data/MICSub_selectX.csv')
    return MIC_subFeatures, MIC_selectX


# 近似马尔科夫毯--Approximate Markov blanket
def AppMB(MIC_selectX, y):
    mcol = MIC_selectX.shape[1]
    F_order = MIC_selectX
    F_opt = []
    F_optInedx = []
    red_Index = []
    # 冗余特征索引
    for i in range(mcol):
        newF_order = []
        # 寻找主元素--fi，并添加至F_opt，记下索引
        temp_FI = F_order[:, 0]
        F_opt.append(temp_FI)
        F_optInedx.append(i+1)
        # 将fi从F_order中删除
        F_order = delete(F_order, 0, axis=1)
        # 查找以fi为马尔科夫毯的特征子集{fj}
        score_CI = mine_Feature(temp_FI, y)
        m = F_order.shape[1]
        count = 0
        for j in range(m):
            temp_FJ = F_order[:, j]
            score_CJ = mine_Feature(temp_FJ, y)
            score_IJ = mine_Feature(temp_FI, temp_FJ)
            # fi是fj的近似马尔科夫毯的条件
            if score_CI > score_CJ and score_CJ < score_IJ:
                # 针对每个fi的冗余特征子集--索引
                red_Index.append(j+1)
                count += 1
            else:
                # 非冗余的特征子集
                newF_order.append(F_order[:, j])
        if F_order.shape[1] == 1:
            F_opt.append(F_order)
            break
        Inedxs = red_Index
        # 此轮去冗余后的特征子集
        F_order = mat(array(newF_order).T)
    Best_Feature = mat(array(F_opt)).T
    file = pd.DataFrame(Best_Feature)
    file.to_csv(u'近似马尔科夫毯删除冗余特征之后的最优特征子集.csv')
    return Best_Feature


# 模型主函数
if __name__ == '__main__':
    startime = time.clock()
    x, y = loadDataSet('WYHXB.txt')
    original_Data = pd.read_csv("data/WYHXB.csv")
    # 计算多个特征与因变量相关性
    scores_xy, MINE_subFeature = corr_Feature(x, y)
    # 第一阶段：获取MIC剔除冗余特征后的数据集
    MIC_subFeatures, MIC_selectX = data_Matching(original_Data)
    # 第二阶段：冗余特征的删除
    MIC_selectX = mat(MIC_selectX)
    # 近似马尔科夫毯  -- 获取强相关无冗余的特征子集
    Best_Feature = AppMB(MIC_selectX, y)
    print(Best_Feature)
    print(shape(Best_Feature))
    endtime = time.clock()
    time_s = endtime - startime
    print('run_time: %fs' % time_s)