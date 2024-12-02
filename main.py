import time

import cv2
import serial
import numpy as np
import tensorflow as tf
import keras

# Load model và labels
class_names = open("labels.txt", "r").readlines()
model = keras.models.load_model('keras_model.h5')

# Kết nối với ESP32 qua Serial
esp32 = serial.Serial(port="COM4", baudrate=9600, timeout=1)

# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    else:
        # Resize and preprocess the frame for the model
        img_resized = cv2.resize(frame, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        # # Model prediction
        prediction = model.predict(img_input, verbose=0)
        name = class_names[np.argmax(prediction)]
        print("Doi nguoi mo cua da dang ky khuon mat")
        if int(name) == 2:
            esp32.write(b'o')
            print("Mo cua")
            break
        cv2.imshow("Diem danh", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()
