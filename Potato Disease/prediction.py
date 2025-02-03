from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

def predict_disease(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = ['Early blight', 'Healthy', 'Late blight']  # Class labels
    return class_labels[class_index], prediction


image_path = r"./static/uploads/img_2.png"
trained_model = load_model('potato_model.h5')
print(predict_disease(trained_model, image_path))