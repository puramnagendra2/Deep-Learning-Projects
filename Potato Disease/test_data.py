from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


model = load_model('potato_model.h5')
test_data_path = './datasets/test'


# Testing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the test dataset
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")