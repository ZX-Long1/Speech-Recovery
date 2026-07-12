import numpy as np
import struct
from os import path
from sklearn.neighbors import KNeighborsClassifier

# ===========================
#  工具函数：读取 MNIST 的 idx 文件
# ===========================
def load_mnist_images(filename):
    """
    读取 MNIST 图像文件（idx3-ubyte）
    返回 numpy 数组，shape = (样本数, 784)
    """
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        return images


def load_mnist_labels(filename):
    """
    读取 MNIST 标签文件（idx1-ubyte）
    返回 numpy 数组，shape = (样本数,)
    """
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# ===========================
#      手写实现 kNN 算法
# ===========================
def classify0(inX, dataSet, labels, k):
    """
    kNN 分类核心函数
    inX: 输入向量
    dataSet: 训练数据集
    labels: 标签
    k: 选取前 k 个最近邻
    """
    # 计算距离
    diffMat = dataSet - inX
    sqDiff = diffMat ** 2
    distances = np.sqrt(sqDiff.sum(axis=1))

    # 排序并取前 k 个
    sortedDistIndices = distances.argsort()

    classCount = {}
    for i in range(k):
        vote = labels[sortedDistIndices[i]]
        classCount[vote] = classCount.get(vote, 0) + 1

    # 返回出现最多的类别
    return max(classCount, key=classCount.get)


# ===========================
#   使用手写 kNN 测试 MNIST
# ===========================
def test_knn_mnist_python(train_images, train_labels, test_images, test_labels, k=3):
    errorCount = 0
    total = len(test_images)

    print("开始使用手写 kNN 测试 MNIST... (由于无优化，会较慢)")

    for i in range(total):
        result = classify0(test_images[i], train_images, train_labels, k)
        if result != test_labels[i]:
            errorCount += 1

        if i % 100 == 0:
            print(f"测试进度：{i}/{total}")

    print("手写 kNN 错误数：", errorCount)
    print("手写 kNN 错误率：", errorCount / total)


# ===========================
#   使用 sklearn kNN 测试 MNIST
# ===========================
def test_knn_mnist_sklearn(train_images, train_labels, test_images, test_labels, k=3):
    print("开始使用 sklearn KNN 训练...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_images, train_labels)

    print("开始预测...")
    pred = knn.predict(test_images)

    errorCount = np.sum(pred != test_labels)
    print("sklearn kNN 错误数：", errorCount)
    print("sklearn kNN 错误率：", errorCount / len(test_labels))


# ===========================
#              主函数
# ===========================
if __name__ == "__main__":
    # 数据路径
    base_path = r"C:\Users\L_minghao\Desktop\学习文件夹\人工智能\实验5\data"

    train_img_path = path.join(base_path, "train-images.idx3-ubyte")
    train_lbl_path = path.join(base_path, "train-labels.idx1-ubyte")
    test_img_path = path.join(base_path, "t10k-images.idx3-ubyte")
    test_lbl_path = path.join(base_path, "t10k-labels.idx1-ubyte")

    print("加载 MNIST 数据集...")
    train_images = load_mnist_images(train_img_path)
    train_labels = load_mnist_labels(train_lbl_path)
    test_images = load_mnist_images(test_img_path)
    test_labels = load_mnist_labels(test_lbl_path)

    print("数据加载完成！")
    print("训练集：", train_images.shape, " 测试集：", test_images.shape)

    # ==========================
    #    第一部分：手写 kNN
    # ==========================
    # 注意：无任何优化，对大规模数据运行较慢
    # test_knn_mnist_python(train_images, train_labels, test_images, test_labels, k=3)

    # ==========================
    #    第二部分：sklearn kNN
    # ==========================
    test_knn_mnist_sklearn(train_images, train_labels, test_images, test_labels, k=3)



