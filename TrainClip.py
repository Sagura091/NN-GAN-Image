from PIL import Image
import pytesseract
import os
import pickle
import numpy as np

# Path to the directory containing your images
image_directory = "D:\\clipart_images"
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Blake Bergstrom\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(target_size, Image.LANCZOS)
    image = image.convert("RGB")
    image = np.array(image) / 255.0  # Normalize pixel values
    return image

# Function to extract text from an image using OCR
def extract_text_from_image(image):
    # Convert the image back to the range [0, 255] and datatype uint8
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    text = pytesseract.image_to_string(image)
    return text


# Lists to store data
preprocessed_images = []
extracted_texts = []


# Iterate through images in the directory
for image_name in os.listdir(image_directory):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
        image_path = os.path.join(image_directory, image_name)
        
        # Load and preprocess the image
        preprocessed_image = load_and_preprocess_image(image_path)

        # Extract text using OCR
        extracted_text = extract_text_from_image(preprocessed_image)
        
        # Append data to lists
        preprocessed_images.append(preprocessed_image)
        extracted_texts.append(extracted_text)


# Save preprocessed_images and extracted_texts using pickle
data = {'preprocessed_images': preprocessed_images, 'extracted_texts': extracted_texts}
with open('D:\\data3.pickle', 'wb') as f:
    pickle.dump(data, f)
