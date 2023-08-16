# Tensorflow
import tensorflow as tf

# Linear(1, 1)인 모델 구성, activation은 sigmoid
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Loss function은 MSE, optimizer는 SGD, LR은 0.003
sgd = tf.keras.optimizers.SGD(learning_rate=0.003)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              )

# 데이터 정의
X = tf.constant([15, 24, 47, 88, 90, 114])
Y = tf.constant([0, 0, 0, 1, 1, 1])

# (None, 1) 꼴의 데이터를 Input으로 받을 것이다.
model.build((None, 1))
# 모델 요약 보기
model.summary()

# validation data도 넣어줘서 학습 해보자
model.fit(X, Y, epochs=10)

print(model(X))
# tf.Tensor(
# [[0.2761128 ]
#  [0.34599352]
#  [0.45904747]
#  [0.88625336]
#  [0.8933784 ]
#  [0.93305486]], shape=(6, 1), dtype=float32)
# Y = tf.constant([0, 0, 0, 1, 1, 1])