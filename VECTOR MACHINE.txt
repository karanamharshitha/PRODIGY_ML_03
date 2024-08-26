pip install numpy pandas matplotlib scikit-learn opencv-python

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

- dataset/
  - cats/
  - dogs/


cat_path = 'dataset/cats/'
dog_path = 'dataset/dogs/'


def load_images(path, label):
    images = []
    labels = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (64, 64))  # Resize images to 64x64
            images.append(image)
            labels.append(label)
    return images, labels

cat_images, cat_labels = load_images(cat_path, 0)  # Label cats as 0
dog_images, dog_labels = load_images(dog_path, 1)  # Label dogs as 1

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)
X = X / 255.0


X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel='linear', random_state=42)  # You can try other kernels like 'rbf', 'poly', etc.
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

def show_image(img, label, pred):
    plt.imshow(img)
    plt.title(f"True: {label}, Pred: {pred}")
    plt.axis('off')
    plt.show()

for i in range(5):
    img = X_test[i].reshape(64, 64, 3)
    label = 'Cat' if y_test[i] == 0 else 'Dog'
    pred = 'Cat' if y_pred[i] == 0 else 'Dog'
    show_image(img, label, pred)

import joblib
joblib.dump(svm, 'svm_cat_dog_classifier.pkl')