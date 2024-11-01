import os
import numpy as np
import time
import cv2
from tensorflow.keras.applications import VGG16
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. Load Dataset
dataset_path = 'D:/AbtapXLA/Bai3/Dataset'
flower_path = os.path.join(dataset_path, 'h')
animal_path = os.path.join(dataset_path, 'x')

# Danh sách để lưu trữ hình ảnh và nhãn
images = []
labels = []

# Tải hình ảnh hoa
for img_name in os.listdir(flower_path):
    img_path = os.path.join(flower_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Resize về kích thước 64x64
        images.append(img)
        labels.append(0)  # Nhãn 0 cho hoa

# Tải hình ảnh động vật
for img_name in os.listdir(animal_path):
    img_path = os.path.join(animal_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Resize về kích thước 64x64
        images.append(img)
        labels.append(1)  # Nhãn 1 cho động vật

# Chuyển đổi danh sách sang numpy array
X = np.array(images)
y = np.array(labels)

# 2. Preprocess Images
# Normalize pixel values
X = X / 255.0  # Normalize pixel values

# 3. Feature Extraction using VGG16 (without the top layer)
# Chuyển đổi kích thước hình ảnh về kích thước (224, 224) cho VGG16
X = np.array([cv2.resize(img, (224, 224)) for img in X])

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
X_vgg = vgg_model.predict(X)

# Flatten features
X_flat = X_vgg.reshape(X_vgg.shape[0], -1)

# Standardize data
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)

# 4. Split dataset into training and testing sets
X_train_flat, X_test_flat, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# 5. Initialize models
models = {
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=2),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, criterion='entropy')
}

# 6. Train and evaluate models
results = []

for model_name, model in models.items():
    # Train model and measure time
    start_time = time.time()
    model.fit(X_train_flat, y_train)
    train_time = time.time() - start_time

    # Predict and measure time
    start_time = time.time()
    y_pred = model.predict(X_test_flat)
    pred_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Save results
    results.append({
        "Model": model_name,
        "Train Time (s)": train_time,
        "Prediction Time (s)": pred_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    })

# 7. Print results
results_df = pd.DataFrame(results)
print(results_df)