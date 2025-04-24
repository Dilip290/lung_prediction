from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Paths
train_path = 'cancer_xray/train'
val_path = 'cancer_xray/val'
test_path = 'cancer_xray/test'

# Data preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path, target_size=(128, 128), batch_size=32, class_mode='binary')
val_data = val_gen.flow_from_directory(val_path, target_size=(128, 128), batch_size=32, class_mode='binary')
test_data = test_gen.flow_from_directory(test_path, target_size=(128, 128), batch_size=32, class_mode='binary')

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Build model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile and train
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weights
)

# Save model
model.save('cancer-detection-model.h5')
print("✅ Model saved as cancer-detection-model.h5")
