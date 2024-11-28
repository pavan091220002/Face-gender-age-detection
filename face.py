from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.callbacks import ModelCheckpoint

import cv2
import numpy as np
import tensorflow as tf

train_dir = "train"

train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rescale=1./255, validation_split=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

validation_generator = validation_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

class_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

images, labels = next(train_generator)
i = random.randint(0, images.shape[0] - 1)
image = images[i]
label_index = np.argmax(labels[i])
label = class_label[label_index]
plt.imshow(image)
plt.title(label)
plt.axis('off')
plt.show()

input_layer = Input(shape=(48, 48, 1))
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.1)(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.1)(x)
x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.1)(x)
x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.1)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

emotion_output = Dense(7, activation='softmax', name='emotion_output')(x)
gender_output = Dense(2, activation='softmax', name='gender_output')(x)
age_output = Dense(1, activation='linear', name='age_output')(x)
model = Model(inputs=input_layer, outputs=[emotion_output, gender_output, age_output])
model.compile(optimizer='adam',
              loss={'emotion_output': 'categorical_crossentropy',
                    'gender_output': 'categorical_crossentropy',
                    'age_output': 'mean_squared_error'},
              metrics={'emotion_output': 'accuracy',
                       'gender_output': 'accuracy',
                       'age_output': 'mean_absolute_error'})

print(model.summary())

checkpoint = ModelCheckpoint('model.weights.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1, save_weights_only=True)

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=1,
                    callbacks=[checkpoint])

model.load_weights('model.weights.h5')

validation_labels = validation_generator.classes
validation_pred_probs = model.predict(validation_generator)
validation_pred_labels = np.argmax(validation_pred_probs[0], axis=1)

confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)

class_names = list(train_generator.class_indices.keys())

sns.set()
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_roi = gray_frame[y:y+h, x:x+w]
        resized_frame = cv2.resize(face_roi, (48, 48))
        normalized_frame = resized_frame / 255.0
        frame_batch = np.expand_dims(normalized_frame, axis=0)
        frame_batch = np.expand_dims(frame_batch, axis=-1)  
        predictions = model.predict(frame_batch)
        predicted_emotion_index = np.argmax(predictions[0])
        predicted_emotion = class_label[predicted_emotion_index]
        predicted_gender_index = np.argmax(predictions[1])
        predicted_gender = 'Male' if predicted_gender_index == 0 else 'Female'
        predicted_age = int(predictions[2][0][0])

        cv2.putText(frame, f'Emotion: {predicted_emotion}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {predicted_gender}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {predicted_age}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion, Gender, and Age Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
