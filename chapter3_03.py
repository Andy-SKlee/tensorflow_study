import tensorflow as tf

# 선형 회귀 모델 (Wx+b)을 정의합니다.
W = tf.Variable(tf.random.normal(shape=[1]), name="W")
b = tf.Variable(tf.random.normal(shape=[1]), name="b")

@tf.function
def linear_model(x):
    return W * x + b

# 손실 함수를 정의합니다. MSE = mean(y' - y)^2
@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

# 최적화를 위한 그라디언트 디센트 옵티마이저를 정의합니다.
optimizer =tf.optimizers.SGD(0.01)  #SGD = 확률적 경사 하강법(Stochastic Gradient Descent), 러닝레이트 = 0.01

# 텐서보드 summary 정보들을 저장할 폴더 경로를 설정합니다.
summary_writer = tf.summary.create_file_writer('./tensorboard_log')

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:   # 텐서플로는 자동 미분(주어진 입력 변수에 대한 연산의 그래디언트(gradient)를 계산하는 것) 을 위한 API를 제공합니다.
        y_pred = linear_model(x)
        loss = mse_loss(y_pred, y)
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=optimizer.iterations)
    gradients = tape.gradient(loss, [W, b])  # 그라디언트(gradients) 계산
    optimizer.apply_gradients(zip(gradients, [W, b]))    # 오차역전파(Backpropagation) - weight 업데이트

# 트레이닝을 위한 입력값과 출력값 준비
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 경사하강법 1000번 실행
for i in range(1000):
    train_step(x_train, y_train)

# 테스트를 위한 입력
x_test = [3.5, 5, 5.5, 6]

print(linear_model(x_test).numpy())

'''
결과값 = [6.999526  9.998259 10.997836 11.997414]
x, y 의 train 값이 2배 차이남 -> 데이터의 경향성(y=2x)
x_test 값인 [3.5, 5, 5.5, 6]의 두배인 [7, 10, 11, 12]에 근접함.
'''