# ============================================
# CNN 手写数字识别（Keras）
# ============================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 去除无关警告

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback


# ======================================================
# 1. 设置随机种子
# ======================================================
seed = 7
np.random.seed(seed)


# ======================================================
# 2. 加载 MNIST 数据集
# ======================================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 变形为 [样本，高，宽，通道]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# 灰度归一化
X_train /= 255
X_test /= 255

# One-hot 编码
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 类别数量（=10）
num_classes = y_test.shape[1]


# ======================================================
# 3. 构建 CNN 模型
# ======================================================
def build_cnn_model():

    model = Sequential()

    # ----------------- 卷积 + 池化 第 1 层 -----------------
    model.add(Conv2D(
        30, (5, 5),
        input_shape=(28, 28, 1),
        activation='relu',
        padding='valid'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # ----------------- 卷积 + 池化 第 2 层 -----------------
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # ----------------- 全连接层 -----------------
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.4))

    # ----------------- softmax 输出层 -----------------
    model.add(Dense(num_classes, activation='softmax'))

    # 编译模型
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


# ======================================================
# 4. 训练模型
# ======================================================
model = build_cnn_model()
# print(model.summary())

class SimpleLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: "
              f"loss={logs['loss']:.6f}, "
              f"acc={logs['accuracy']:.4f}, "
              f"val_loss={logs['val_loss']:.6f}, "
              f"val_acc={logs['val_accuracy']:.4f}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=200,
    verbose=0,  # 不显示默认 Keras 输出
    callbacks=[SimpleLogger()]
)


# ======================================================
# 5. 评估模型
# ======================================================
scores = model.evaluate(X_test, y_test, verbose=0)
print("\nCNN Test Accuracy: %.3f%%" % (scores[1] * 100))


# ======================================================
# 6. 绘制训练曲线（准确率 & 损失）
# ======================================================
plt.figure(figsize=(12, 5))

# Accuracy 曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss 曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
