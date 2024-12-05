from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img_path = 'C:/Belajar pemrograman/Belajar-python/sample train_augment/validation/S182/aug_0_S182-02-t10_01.ppm'
img = cv2.imread(img_path)

analysis = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)

if isinstance(analysis, list):
    faces = analysis
else:
    faces = [analysis]

correct_predictions = 0  
total_predictions = len(faces)  

for face in faces:
    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    predicted_age = face['age']
    predicted_gender = face['dominant_gender']
    predicted_emotion = face['dominant_emotion']
    
    label = f"{predicted_gender}, {predicted_age} years, {predicted_emotion}"
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    correct_predictions += 1  

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

if total_predictions > 0:
    accuracy = correct_predictions / total_predictions * 100
else:
    accuracy = 0 

print("\nHasil Analisis:")
for face in faces:
    print(f"Usia: {face['age']}, Jenis Kelamin: {face['dominant_gender']}, Emosi: {face['dominant_emotion']}")
