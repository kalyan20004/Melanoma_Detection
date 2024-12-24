import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Define the allowed extensions for image files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the TensorFlow Lite model
tflite_model_path = "models/vgg-16.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the upload folder within the static directory
app.config['uploads'] = os.path.join('website', 'static', 'uploads')

# Ensure the upload folder exists
os.makedirs(app.config['uploads'], exist_ok=True)

# Function to preprocess the image and predict using the TensorFlow Lite model
def preprocess_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((150, 150))  # Resize image to match model input size
    img = np.array(img).astype(np.float32)  # Convert to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize to [0, 1]
    return img

def predict_image(image_path):
    # Preprocess the image
    input_data = preprocess_image(image_path)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor (the prediction result)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Class names (for binary classification)
    class_names = ['The image is classified as Benign', 'The image is classified as Malignant']
    
    # Convert the output to a prediction (assuming output is a scalar)
    predicted_class = class_names[int(output_data[0] > 0.5)]  # Binary classification
    prediction_score = float(output_data[0])  # Prediction score (probability)
    
    return predicted_class, prediction_score

# Prediction page route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    background_image = 'prediction.jpg'  # Path for the prediction page background
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Check if the file is valid
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)

            # Define the full file path for saving the uploaded image
            img_path = os.path.join(app.config['uploads'], filename)

            # Save the uploaded image
            file.save(img_path)

            # Make the prediction
            result, prediction_score = predict_image(img_path)

            # Generate the URL for the uploaded image
            img_url = url_for('static', filename=f'uploads/{filename}')

            # Prepare message based on prediction
            if "Malignant" in result:
                warning_message = "Warning: The image suggests a Malignant condition. Please consult a healthcare provider."
            else:
                warning_message = "The image suggests a Benign condition. No immediate action needed."

            # Render the result with warning message
            return render_template(
                'prediction.html',
                title="Prediction",
                result=result,
                warning_message=warning_message,
                img_url=img_url,
                background_image=background_image  # Pass the background image to the template
            )

    return render_template('prediction.html', title="Prediction", background_image=background_image)


@app.route('/description')
def description():
    background_image = 'descritpion.jpg'  # Path for the description page background
    return render_template('description.html', title="Description", background_image=background_image)


@app.route('/contact')
def contact():
    background_image = 'contact.jpg'  # Path for the contact page background
    return render_template('contact.html', title="Contact Us", background_image=background_image)

# Homepage route
@app.route('/')
def index():
    return render_template('index.html', title="Home")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
