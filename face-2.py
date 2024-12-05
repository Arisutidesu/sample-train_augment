from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('ai_models.h5')

img_path = 'C:/Belajar pemrograman/Belajar-python/sample train_augment/validation/S008/aug_1_S008-01-t10_01.ppm'
img = cv2.imread(img_path)

detector = MTCNN()

faces = detector.detect_faces(img)

for face in faces:
    x, y, w, h = face['box']
    
    face_img = img[y:y + h, x:x + w]  
    face_img = cv2.resize(face_img, (128, 128))  
    face_img = np.expand_dims(face_img, axis=0) / 255.0 

    prediction = model.predict(face_img)
    predicted_class_idx = np.argmax(prediction)
    accuracy_score = prediction[0][predicted_class_idx]

    print(f"Accuracy: {accuracy_score * 300:.2f}%")

    label_text = f"Accuracy: {accuracy_score * 300:.2f}%"
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, label_text, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()