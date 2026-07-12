# -*- coding: utf-8 -*-
"""
实验名称: 卷积神经网络手写数字识别
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# ============================
# 1. 配置参数
# ============================
BATCH_SIZE = 300 # 批大小
NUM_CLASSES = 10  # 分类类别数 (0-9)
EPOCHS = 10  # 训练轮次
IMG_ROWS, IMG_COLS = 28, 28  # 图像尺寸


def load_data():
    """
    加载并预处理 MNIST 数据
    """
    print("正在加载数据...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 根据 Keras 后端配置调整数据维度顺序
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
        x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
        input_shape = (1, IMG_ROWS, IMG_COLS)
    else:
        x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
        x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
        input_shape = (IMG_ROWS, IMG_COLS, 1)

    # 转换数据类型并归一化到 [0, 1]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('训练集样本数:', x_train.shape[0])
    print('测试集样本数:', x_test.shape[0])

    # 将类标转换为二值类别矩阵 (One-hot Encoding)
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test), input_shape


def create_cnn_model(input_shape):
    """
    构建卷积神经网络结构
    """
    model = Sequential()

    # 第一层卷积: 32个 3x3 卷积核
    # padding='same' 保证边缘信息不丢失
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    # 第二层卷积: 64个 3x3 卷积核 (加深网络宽度)
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # 最大池化层: 降低维度
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout层: 随机丢弃25%神经元，防止过拟合
    model.add(Dropout(0.25))
    # 展平层: 将多维特征图展开为一维向量
    model.add(Flatten())

    # 全连接层
    model.add(Dense(128, activation='relu'))
    # Dropout层: 再次丢弃50%
    model.add(Dropout(0.5))
    # 输出层: Softmax 归一化输出概率
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # 定义优化器 (使用 Adam，学习率默认)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


def plot_history(history):
    """绘制训练过程的 Loss 和 Accuracy 曲线"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # 准确率绘图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Test Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失绘图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# ============================
# 主运行逻辑
# ============================
if __name__ == '__main__':
    # 1. 准备数据
    (x_train, y_train), (x_test, y_test), in_shape = load_data()
    # 2. 构建模型
    cnn_model = create_cnn_model(in_shape)
    # 3. 训练模型
    print("\n--- 开始训练 ---")
    history_log = cnn_model.fit(x_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                verbose=2,
                                validation_data=(x_test, y_test))
    # 4. 最终评估
    print("\n--- 最终测试集评估 ---")
    score = cnn_model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {score[0]:.6f}')
    print(f'Test accuracy: {score[1] * 100:.3f}%')
    # 5. 绘图
    plot_history(history_log)