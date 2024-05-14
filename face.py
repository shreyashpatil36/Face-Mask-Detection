import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator



categories = ['with_mask', 'without_mask']
data = []

for category in categories:
    path = os.path.join('train', category)
    label = categories.index(category)

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        data.append([img, label])

np.random.shuffle(data)

X = []
Y = []

for features, label in data:
    X.append(features)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_data = []
for i, (img, label) in enumerate(data):
    augmented_data.append([img, label])
    if label == 0:  
        img = img.reshape((1,) + img.shape)  
        for batch in datagen.flow(img, batch_size=1):
            augmented_data.append([batch[0], label])
            if len(augmented_data) % 100 == 0:
                print("Augmented", len(augmented_data), "images")
            if len(augmented_data) >= 2 * len(data):  
                break

np.random.shuffle(augmented_data)

X_augmented = np.array([x[0] for x in augmented_data])
Y_augmented = np.array([x[1] for x in augmented_data])
X_augmented = X_augmented / 255.0

X_train_augmented, X_test_augmented, y_train_augmented, y_test_augmented = train_test_split(X_augmented, Y_augmented, test_size=0.2)

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg.layers[:-8]:  
    layer.trainable = True

model = Sequential()
for layer in vgg.layers:
    model.add(layer)

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_augmented, y_train_augmented, epochs=10, batch_size=32, validation_data=(X_test_augmented, y_test_augmented))

_, train_accuracy = model.evaluate(X_train_augmented, y_train_augmented)
_, test_accuracy = model.evaluate(X_test_augmented, y_test_augmented)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

train_predictions = model.predict(X_train_augmented)
test_predictions = model.predict(X_test_augmented)

train_mse = np.mean(np.square(train_predictions - y_train_augmented))
test_mse = np.mean(np.square(test_predictions - y_test_augmented))

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

cap = cv2.VideoCapture(0)

def detect_face_mask(img):
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    y_pred = model.predict(img)
    return y_pred[0][0]

def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    end_x = pos[0] + text_size[0] + 2
    end_y = pos[1] + text_size[1] - 2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, (pos[0], pos[1] + text_size[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (224, 224))

    y_pred = detect_face_mask(frame)

    if y_pred > 0.5:
        draw_label(frame, "Face with Mask", (30, 30), (0, 255, 0))
    else:
        draw_label(frame, "No Face Mask", (30, 30), (0, 0, 255))

    cv2.imshow('Real-Time Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
