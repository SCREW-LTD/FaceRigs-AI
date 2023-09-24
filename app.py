import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_model = load_model('model.h5')
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def detect_emotion(frame):
    frame = cv2.resize(frame, (48, 48))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.expand_dims(frame, axis=0)
    frame = np.expand_dims(frame, axis=-1)

    emotion_prediction = emotion_model.predict(frame)
    emotion_index = np.argmax(emotion_prediction)
    emotion = emotion_labels[emotion_index]

    return emotion

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        emotion = detect_emotion(face_roi)

        color = (0, 255, 0)
        thickness = 2
        line_type = cv2.LINE_AA

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness, lineType=line_type)

        text_color = color
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, thickness, lineType=line_type)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
