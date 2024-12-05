import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_dir = 'C:/Belajar pemrograman/Belajar-python/sample train_augment'

model = load_model('ai_models.h5')

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

true_labels = validation_generator.classes
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=validation_generator.class_indices.keys()))

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()