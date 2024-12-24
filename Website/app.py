import os
import tensorflow as tf
import numpy as np
import streamlit as st
from werkzeug.utils import secure_filename
from PIL import Image

# Load the TensorFlow Lite model
tflite_model_path = "models/vgg-16.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the upload folder within the static directory (optional)
app_config = {'uploads': os.path.join('website', 'static', 'uploads')}
os.makedirs(app_config['uploads'], exist_ok=True)

# Function to preprocess the image and predict using the TensorFlow Lite model
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Resize image to match model input size
    img = np.array(img).astype(np.float32)  # Convert to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to [0, 1]
    return img

def predict_image(image_path):
    input_data = preprocess_image(image_path)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor (the prediction result)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Class names (for binary classification)
    class_names = ['The image is classified as Benign', 'The image is classified as Malignant']
    
    predicted_class = class_names[int(output_data[0] > 0.5)]  # Binary classification
    prediction_score = float(output_data[0])  # Prediction score (probability)
    
    return predicted_class, prediction_score

# Streamlit app
def main():
    st.title("Melanoma Detection")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'gif'])

    if uploaded_file is not None:
        # Secure the filename and save the file
        filename = secure_filename(uploaded_file.name)
        img_path = os.path.join(app_config['uploads'], filename)

        # Save the uploaded image to disk
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Make the prediction
        result, prediction_score = predict_image(img_path)

        # Display the result
        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        # Show prediction and warning message
        if "Malignant" in result:
            warning_message = "Warning: The image suggests a Malignant condition. Please consult a healthcare provider."
        else:
            warning_message = "The image suggests a Benign condition. No immediate action needed."

        st.subheader(result)
        st.write(f"Prediction score: {prediction_score:.2f}")
        st.warning(warning_message)

    else:
        st.info("Upload an image to get a prediction.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
