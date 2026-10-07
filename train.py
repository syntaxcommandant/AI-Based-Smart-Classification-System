# import tensorflow as tf 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt 
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=15,
#     zoom_range=0.15, 
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
# )
# train_generator = train_datagen.flow_from_directory(
#     "dataset",
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode="categorical",
#     subset="training"
# )
# validation_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# validation_generator = validation_datagen.flow_from_directory(
#     "dataset",
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode="categorical",
#     subset="validation"
#     shuffle=False
# )
# print(train_generator.class_indices)
# model = Sequential([
#     Conv2D(32, (3,3), 
#     activation="relu", 
#     input_shape=(224,224,3)),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),

#     Conv2D(64, (3,3),
#      activation="relu"),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),

#     Conv2D(128, (3,3), 
#     activation="relu"),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),

    
#     Conv2D(256, (3,3), 
#     activation="relu"),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),

#     GlobalAveragePooling2D(),

#     Dense(256, activation="relu"),

#     Dropout(0.3),
    
#     Dense(128, activation="relu"),

#     Dropout(0.3),

#     Dense(7, activation="softmax") #7 categories -> 7 outputs -> softmax activation function --> softmax hr category ki probability nikalta hai
# ])
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=5,
#     restore_best_weights=True
# )
# history = model.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=40,
#     callbacks=[early_stop]
# )
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(["Train", "Validation"])
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title("Model Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend(["Train", "Validation"])
# plt.show()
# model.save("model/waste_classifier.keras")

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# DATA GENERATORS

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,

    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

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

print("Class indices:", train_generator.class_indices)

# MOBILE NET V2

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Pehle pretrained layers ko freeze karenge
base_model.trainable = False

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False
    
# OUR CLASSIFIER

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(7, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# TRAIN

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=40,
    callbacks=[early_stop]
)

model.save("model/waste_classifier.keras")

val_loss, val_acc = model.evaluate(validation_generator, verbose=1)

print("SAVED MODEL VALIDATION ACCURACY:", val_acc)

# ACCURACY GRAPH

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.show()

# LOSS GRAPH

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train", "Validation"])

plt.show()


