# kNN.py
# 实验：kNN算法的实现
# 包含：classify0 (普通投票 / 距离加权投票), autoNorm, 简单示例与 k 值验证

import numpy as np
import operator

def createDataSet():
    """
    返回示例数据集和标签
    每行是一个样本：[理综成绩, 文综成绩]
    labels 对应 "理科生" 或 "文科生"
    """
    group = np.array([
        [250, 100],  # 理科生
        [270, 120],  # 理科生
        [111, 230],  # 文科生
        [130, 260],  # 文科生
        [200,  80],  # 理科生
        [ 70, 190],  # 文科生
    ], dtype=float)
    labels = ['理科生', '理科生', '文科生', '文科生', '理科生', '文科生']
    return group, labels


def autoNorm(dataSet):
    """
    归一化特征到 [0,1]
    返回: normDataset, ranges, minVals
    公式: (x - min) / (max - min)
    """
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    # 防止除以 0
    ranges[ranges == 0] = 1.0
    normDataSet = (dataSet - minVals) / ranges
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k=3, weighted=False):
    """
    kNN 分类函数
    参数:
        inX: 待分类向量 (1D numpy array 或 list)
        dataSet: 训练集 (numpy array, 每行一个样本)
        labels: 训练集标签 (list)
        k: 选择最近邻的数目
        weighted: 是否使用距离加权投票（True 使用权重 1/d^2）
    返回:
        预测的分类标签
    说明:
        使用欧式距离
    """
    # 转为 numpy array
    inX = np.array(inX, dtype=float)
    dataSet = np.array(dataSet, dtype=float)
    labels = list(labels)

    # 计算欧氏距离
    diffMat = dataSet - inX  # 广播
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = np.sqrt(sqDistances)

    # 排序并取前 k 个下标
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(min(k, len(sortedDistIndices))):
        voteLabel = labels[sortedDistIndices[i]]
        if weighted:
            dist = distances[sortedDistIndices[i]]
            # 如果距离为0（完全重合），赋一个很大的权重（处理除0）
            if dist == 0.0:
                weight = 1e9
            else:
                # 常见的加权公式：权重 = 1 / (dist^2)
                weight = 1.0 / (dist ** 2)
        else:
            weight = 1.0

        classCount[voteLabel] = classCount.get(voteLabel, 0.0) + weight

    # 按权重/票数排序并返回最大的标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def testSimple():
    """
    使用实验文理科示例测试 kNN
    """
    group, labels = createDataSet()
    # 测试点 (105, 210)
    testPoint = [105, 210]

    # 先对数据归一化再分类
    normMat, ranges, minVals = autoNorm(group)
    normTestPoint = (np.array(testPoint, dtype=float) - minVals) / ranges

    # k=3 普通投票
    result1 = classify0(normTestPoint, normMat, labels, k=3, weighted=False)
    print("不加权 k=3 预测类别：", result1)

    # k=3 加权投票
    result2 = classify0(normTestPoint, normMat, labels, k=3, weighted=True)
    print("加权 k=3 预测类别：", result2)

    # 直接在原数据上（不归一化）测试
    result3 = classify0(testPoint, group, labels, k=3, weighted=False)
    print("原始量纲（未归一）不加权 k=3 预测类别：", result3)


def crossValidateK(dataSet, labels, k_values=(1,3,5,7), weighted=False):
    """
    留一交叉验证（LOOCV）用于挑选 k 值。
    对于数据集较小的情况使用留一验证能够给出直观的 k 值比较。
    返回字典: {k: 错误率}
    """
    dataSet = np.array(dataSet, dtype=float)
    n = dataSet.shape[0]
    normMat, ranges, minVals = autoNorm(dataSet)

    errors = {}
    for k in k_values:
        error_count = 0
        for i in range(n):
            # 留出第 i 个作为测试
            train_idx = [j for j in range(n) if j != i]
            train_data = normMat[train_idx]
            train_labels = [labels[j] for j in train_idx]
            test_point = normMat[i]
            true_label = labels[i]
            pred = classify0(test_point, train_data, train_labels, k=k, weighted=weighted)
            if pred != true_label:
                error_count += 1
        error_rate = error_count / n
        errors[k] = error_rate
    return errors

if __name__ == "__main__":
    print("=== kNN 实验演示 ===")
    testSimple()

    # 使用交叉验证比较不同 k 的错误率
    group, labels = createDataSet()
    k_candidates = (1, 3, 5)
    errors_unweighted = crossValidateK(group, labels, k_values=k_candidates, weighted=False)
    errors_weighted = crossValidateK(group, labels, k_values=k_candidates, weighted=True)
    print("\n留一交叉验证（不加权）错误率：", errors_unweighted)
    print("留一交叉验证（加权）错误率：", errors_weighted)



