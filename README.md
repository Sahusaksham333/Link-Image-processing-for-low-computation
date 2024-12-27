pip install requests beautifulsoup4 pillow opencv-python-headless# Link-Image-processing-for-low-computation
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image

# Directory to save processed images
OUTPUT_DIR = "processed_images"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to download and read an image
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Function to process an image
def process_image(image):
    try:
        # Grayscale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection using Canny
        edges = cv2.Canny(gray, 100, 200)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Combine processed images
        processed_images = {
            'grayscale': gray,
            'edges': edges,
            'enhanced': enhanced
        }
        return processed_images
    except Exception as e:
        print(f"Error processing image: {e}")
        return {}

# Save processed images
def save_images(images, base_name):
    for name, img in images.items():
        path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}.jpg")
        cv2.imwrite(path, img)
        print(f"Saved {path}")

# Main function
def main():
    image_url = "https://www.tigren.com/blog/wp-content/uploads/2022/02/how-much-cost-to-start-online-business-1536x878.jpg"

    print(f"Downloading image: {image_url}")
    image = download_image(image_url)
    if image is None:
        return

    print(f"Processing image...")
    processed_images = process_image(image)

    print(f"Saving processed images...")
    save_images(processed_images, "image_1")  # Assuming a single image

    print(f"All images processed and saved in {OUTPUT_DIR}.")

# Run the main function
main()
