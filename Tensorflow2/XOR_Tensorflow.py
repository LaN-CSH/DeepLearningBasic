# Tensorflow
import tensorflow as tf

# Numpy
import numpy as np

# Hidden Layer가 있는 신경망 모델 구성, activation은 sigmoid
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4, activation='sigmoid'),
  tf.keras.layers.Dense(4, activation='sigmoid'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

X = tf.constant([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = tf.constant([0, 1, 1, 0])

# (None, 1) 꼴의 데이터를 Input으로 받을 것이다.
model.build((None, 2))
# 모델 요약 보기
model.summary()

# compile (Loss, optimizer 설정)
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              )

model.fit(X, Y, epochs=500)

print(model(X))