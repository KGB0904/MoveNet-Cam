import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# JSON 파일에서 데이터 로드
with open('sequence_data.json', 'r') as file:
    data = json.load(file)
    sequences = np.array(data['sequences'], dtype=np.float32)
    labels = np.array(data['labels'], dtype=np.float32)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# 모델 구축
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # 이진 분류를 위한 활성화 함수
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")