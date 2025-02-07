import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

train_data_path = "./datasets/train"
validation_data_path = "./datasets/val"

training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

training_data = training_datagen.flow_from_directory(train_data_path, # this is the target directory
                                      target_size=(196, 196), # all images will be resized to 150x150
                                      batch_size=32,
                                      class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
 
print("Indices ",training_data.class_indices)

valid_datagen = ImageDataGenerator(rescale=1./255)
 
# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                  target_size=(196,196),
                                  batch_size=32,
                                  class_mode='binary')

# Model Save Path
model_path = "tomato_model.h5"

# Save only the best model
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Define Model
model = Sequential([
    Conv2D(32, kernel_size=3, input_shape=[196, 196, 3], activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(512, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.5),
    GlobalAveragePooling2D(),
    # Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(5, activation='softmax')  # Assuming 6 output classes
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(training_data, epochs=5, verbose=1, 
          validation_data=valid_data, callbacks=callbacks_list)

# Save Final Model
model.save(model_path)

print("Training Complete. Model Saved as 'tomato_model.h5'")


test_data_path = './datasets/test'


# Testing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the test dataset
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(196, 196),
    batch_size=32,
    class_mode='binary'
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")