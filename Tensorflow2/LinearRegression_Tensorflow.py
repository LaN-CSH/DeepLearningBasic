# Tensorflow
import tensorflow as tf

# Linear(1, 1)인 모델 구성, bias는 일부러 y=Wx를 근사하기 위해 Fasle
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1, use_bias=False),
])

# Loss function은 MSE, optimizer는 SGD를 쓸건데 LR은 0.00003으로 설정(이 이상은 Loss가 발산)
sgd = tf.keras.optimizers.SGD(learning_rate=0.00003)
model.compile(optimizer=sgd,
              loss='MSE',
              )

# 데이터 정의
X = tf.constant([100, 125, 150, 190, 206])
Y = tf.constant([105, 122, 155, 176, 207])

# (None, 1) 꼴의 데이터를 Input으로 받을 것이다.
model.build((None, 1))
# 모델 요약 보기
model.summary()

# validation data도 넣어줘서 학습 해보자
X_val = tf.constant([110, 120, 130, 140])
Y_val = tf.constant([110, 120, 130, 140])
model.fit(X, Y, epochs=10, validation_data=(X_val, Y_val))

print(model(X))
# tf.Tensor([[ 98.79294 ]
# [123.491165]
#  [148.1894  ]
#  [187.70657 ]
#  [203.51344 ]], shape=(5, 1), dtype=float32)

# Y = tf.constant([105, 122, 155, 176, 207])