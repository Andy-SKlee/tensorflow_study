# AutoEncoder를 이용한 MNIST Reconstruction

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터를 다운로드, float32로 변경, 28x28 형태의 이미지를 784차원으로 flattening, [0, 255] -> [0, 1]로 normalize
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, x_test) = x_train.astype('float32'), x_test.astype('float32')
(x_train, x_test) = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
(x_train, x_test) = x_train / 255., x_test / 255.

# 학습에 필요한 설정값 정의
learning_rate = 0.02
training_epochs = 50    # 반복횟수
batch_size = 256        # 배치개수
display_step = 1        # 손실함수 출력 주기
examples_to_show = 10   # 보여줄 MNIST Reconstruction 이미지 개수
input_size = 784        # 28 x 28
hidden1_size = 256
hidden2_size = 128

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.shuffle(60000).batch(batch_size)

# AutoEncoder 모델을 정의
class AutoEncoder(object):
    # AutoEncoder 모델을 위한 tf.Variable들을 정의
    def __init__(self):
        # 인코딩(Encoding) - 784 -> 256 -> 128
        self.W1 = tf.Variable(tf.random.normal(shape=[input_size, hidden1_size]))
        self.b1 = tf.Variable(tf.random.normal(shape=[hidden1_size]))
        self.W2 = tf.Variable(tf.random.normal(shape=[hidden1_size, hidden2_size]))
        self.b2 = tf.Variable(tf.random.normal(shape=[hidden2_size]))
        # 디코딩(Decoding) - 128 -> 256 -> 784
        self.W3 = tf.Variable(tf.random.normal(shape=[hidden2_size, hidden1_size]))
        self.b3 = tf.Variable(tf.random.normal(shape=[hidden1_size]))
        self.W4 = tf.Variable(tf.random.normal(shape=[hidden1_size, input_size]))
        self.b4 = tf.Variable(tf.random.normal(shape=[input_size]))

    def __call__(self, x):
        H1_output = tf.nn.sigmoid(tf.matmul(x, self.W1) + self.b1)
        H2_output = tf.nn.sigmoid(tf.matmul(H1_output, self.W2) + self.b2)
        H3_output = tf.nn.sigmoid(tf.matmul(H2_output, self.W3) + self.b3)
        reconstructed_x = tf.nn.sigmoid(tf.matmul(H3_output, self.W4) + self.b4)

        return reconstructed_x

# MSE 손실 함수를 정의
@tf.function
def mse_loss(y_pred, y_true):
    return tf.reduce_mean(tf.pow(y_true - y_pred, 2))  # pow : 거듭제곱 값을 계

# 최적화를 위한 RMSProp 옵티마이저를 정의
optimizer = tf.optimizers.RMSprop(learning_rate)

# 최적화를 위한 function 정의
@tf.function
def train_step(model, x):
    # 타겟데이터는 인풋데이터와 같음
    y_true = x
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = mse_loss(y_pred, y_true)
    gradients = tape.gradient(loss, vars(model).values())
    optimizer.apply_gradients(zip(gradients, vars(model).values()))

# Autoencoder 모델을 선언
AutoEncoder_model = AutoEncoder()

# 지정된 횟수만큼 최적화를 수행, Autoencoder는 Unsupervised Learning이므로 타겟 레이블 y가 필요하지 않음
for epoch in range(training_epochs):
    for batch_x in train_data:
        # 옵티마이저를 실행해서 파라미터들을 업데이트
        _, current_loss = train_step(AutoEncoder_model, batch_x), mse_loss(AutoEncoder_model(batch_x), batch_x)
    if epoch % display_step == 0:
        print("반복(Epoch) : %d, 손실 함수(Loss) : %f" % ((epoch + 1), current_loss))

# 테스트 데이터로 Reconstruction을 수행
reconstructed_result = AutoEncoder_model(x_test[:examples_to_show])
# 원본 MNIST 데이터와 Reconstruction 결과를 비교
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(x_test[i], (28, 28)))
    a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))
f.savefig('reconstructed_mnist_image.png', bbox_inches='tight')
f.show()
plt.draw()
plt.waitforbuttonpress()