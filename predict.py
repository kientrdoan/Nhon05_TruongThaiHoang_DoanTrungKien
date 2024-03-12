from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('emotion_detection_v5.h5')

# emotion_labels = ['Angry','Disgust','Fear','Happy','Surprise', 'Sad', 'Neutral']
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Sad', 'Surprise', 'Neutral']


def predict_real_time():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to predict emotion from a given face image
def predict_emotion(roi_gray):
    # Chắc chắn rằng ảnh có một kênh (grayscale)
    if len(roi_gray.shape) == 3:
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

    # Resize ảnh thành kích thước mong đợi của mô hình (48, 48)
    roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Chuyển đổi sang dạng dữ liệu mong đợi của mô hình
    roi_image = roi_gray_resized.astype('float') / 255.0
    roi_image = np.expand_dims(roi_image, axis=0)

    # print(roi)
    # Dự đoán
    prediction = classifier.predict(roi_image)[0]
    label = emotion_labels[prediction.argmax()]

    return label


def predict_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Process each face in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            label = predict_emotion(roi_gray)
            label_position = (x, y)
            cv2.putText(image, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # return label
        else:
            return "No Face"
    
    return image
