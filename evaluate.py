import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Saved model load kiye
model = tf.keras.models.load_model("model/waste_classifier.keras")

# 2. Validation data generator
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

validation_generator = validation_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# 3. Predictions
validation_generator.reset()

predictions = model.predict(validation_generator)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes

# 4. Class names
class_labels = list(validation_generator.class_indices.keys())

# 5. Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 6. Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels
    )
)