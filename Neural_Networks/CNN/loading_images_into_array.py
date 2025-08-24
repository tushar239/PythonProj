import os
from PIL import Image
import numpy as np


def load_images_to_numpy_pil(folder_path):
    """
    Loads all images from a specified folder into a single NumPy array using Pillow.

    Args:
        folder_path (str): The path to the folder containing the images.

    Returns:
        numpy.ndarray: A NumPy array containing all loaded images.
                       The shape will be (num_images, height, width, channels) for color images,
                       or (num_images, height, width) for grayscale images.
    """
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath)
                image_list.append(np.array(img))
            except IOError:
                print(f"Could not open or read image: {filepath}")

    if image_list:
        # Pad or resize images to a common shape if they have different dimensions
        # For simplicity, this example assumes all images have the same dimensions.
        # If not, you would need to implement padding or resizing logic here.
        arr = np.array(image_list, dtype=object)
        print(arr.shape) # (4000,)
        print(arr)
        return arr
    else:
        return np.array([])  # Return an empty array if no images are found

# Example usage:
folder = "dataset/training_set/dogs"
all_images_array = load_images_to_numpy_pil(folder)
print(f"images: {all_images_array}")
print(f"Shape of the combined image array: {all_images_array.shape}")