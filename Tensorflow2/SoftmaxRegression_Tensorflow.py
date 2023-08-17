# Tensorflow
import tensorflow as tf

# Numpy
import numpy as np

# Linear(16, 7)인 모델 구성, activation은 softmax
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense( 7, activation='softmax'),
])

# Loss function은 categorical CE, optimizer는 SGD, LR은 0.001
sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              )

# data-04-zoo.csv 데이터셋을 이용할 것이다.
zoo = np.loadtxt('data-04-zoo.csv', delimiter=',')

# Train data 정의 뒤의 10개는 Test Data로 쓰자 
x = zoo[:-10, :-1]
y = zoo[:-10, -1:]

print(zoo.shape) # (101, 17), 마지막 column은 0~6 중 어느 분류인지 표시 되어있다
X = tf.constant(x)
Y = tf.constant(y)

model.fit(X, Y)