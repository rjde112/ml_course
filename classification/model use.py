import joblib
import numpy as np
from PIL import Image  # To load and manipulate images
import matplotlib.pyplot as plt
import argparse  # To handle command-line arguments

# Function to load and process the image
def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, resizes it to 8x8 pixels,
    and converts it into an array compatible with the model.
    """
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode)
    
    # Resize the image to 8x8 pixels (use LANCZOS for downscaling)
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    
    # Display the resized image
    plt.imshow(img, cmap='gray')
    plt.title('Image resized to 8x8')
    plt.show()
    
    # Convert the image to a NumPy array and invert pixel values (digits dataset uses inverted pixels)
    img_array = np.array(img)
    img_array = 16 - (img_array / 16.0)  # Invert pixel values and normalize to 0-16
    
    # Flatten the image from 8x8 to a 64-element vector
    img_flatten = img_array.flatten()
    
    # Return the flattened image ready for the model
    return img_flatten

# Function to load the model and make predictions
def predict_digit(model_path, image_path):
    """
    Loads the model from the .pkl file, processes the image, and returns the prediction.
    """
    # Load the model from the .pkl file
    model = joblib.load(model_path)
    
    # Process the image
    processed_image = preprocess_image(image_path)
    
    # Reshape the image to be compatible with the model (1, -1)
    processed_image = processed_image.reshape(1, -1)
    
    # Use the model to make the prediction
    prediction = model.predict(processed_image)
    
    # Return the prediction
    return prediction[0]

# Main function using argparse to pass the model and image
def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Predict the handwritten digit using a KNN model.')
    parser.add_argument('model', type=str, help='Path to the .pkl model file')
    parser.add_argument('image', type=str, help='Path to the image file to predict')
    
    # Parse the arguments
    args = parser.parse_args()

    # Perform the prediction
    predicted_digit = predict_digit(args.model, args.image)
    
    # Print the result
    print(f"The digit predicted by the model is: {predicted_digit}")

# Execute the main function if this file is run directly
if __name__ == '__main__':
    main()
