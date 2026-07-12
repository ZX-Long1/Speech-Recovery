# -*- coding: UTF-8 -*-
"""
实验：朴素贝叶斯算法
功能：通过一个简单的文本分类实例（侮辱性言论过滤）理解朴素贝叶斯分类器原理与实现
"""

import numpy as np
# ---------------------------------------------------
# 1. 构建数据集
# ---------------------------------------------------
def loadDataSet():
    """
    创建实验样本
    Returns:
        postingList - 实验样本切分的词条列表
        classVec - 类别标签向量，1代表侮辱类，0代表非侮辱类
    """
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱类，0表示非侮辱类
    return postingList, classVec


# ---------------------------------------------------
# 2. 创建词汇表
# ---------------------------------------------------
def createVocabList(dataSet):
    """
    将样本整理成不重复的词条列表（词汇表）
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# ---------------------------------------------------
# 3. 文本向量化
# ---------------------------------------------------
def setOfWords2Vec(vocabList, inputSet):
    """
    将输入词条转换为词汇表向量（词集模型）
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# ---------------------------------------------------
# 4. 朴素贝叶斯训练函数
# ---------------------------------------------------
def trainNB0(trainMatrix, trainCategory):
    """
    训练朴素贝叶斯分类器
    Parameters:
        trainMatrix - 训练文档矩阵
        trainCategory - 类别标签向量
    Returns:
        p0Vect - 非侮辱类条件概率数组
        p1Vect - 侮辱类条件概率数组
        pAbusive - 文档属于侮辱类的先验概率
    """
    numTrainDocs = len(trainMatrix)           # 训练文档数目
    numWords = len(trainMatrix[0])            # 每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 属于侮辱类的先验概率

    # 初始化计数器与分母
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0

    # 遍历每一篇训练文档
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 计算条件概率
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom

    return p0Vect, p1Vect, pAbusive


# ---------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------
if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    print("样本数据：")
    for i, each in enumerate(postingList):
        print(f"{i+1}: {each}, 类别={classVec[i]}")

    # 创建词汇表
    myVocabList = createVocabList(postingList)
    print("\n词汇表：\n", myVocabList)

    # 文本向量化
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print("\n词条向量矩阵(trainMat)：\n", np.array(trainMat))

    # 训练朴素贝叶斯分类器
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print("\n非侮辱类条件概率(p0V)：\n", p0V)
    print("\n侮辱类条件概率(p1V)：\n", p1V)
    print("\n侮辱类先验概率(pAb)：", pAb)
