# General imports 
import numpy as np
from PIL import Image
import os
import time
from concurrent.futures import ThreadPoolExecutor

def process_image(image_tuple):
  frame_id, img_array = image_tuple

  # Filter out pixels that are pure black (0, 0, 0)
  #print(f"Processing frame {frame_id}...")
  #print("Original image shape:", img_array.shape)

  # Only grab pixels that are not pure black and collapse them into a single array
  img_array = img_array[~np.all(img_array == [0, 0, 0], axis=-1)]

  print("Filtered image shape:", img_array.shape)

  # Compute variance of remaining pixels
  pixel_variance = np.var(img_array)

  return (frame_id, pixel_variance)

# Data comes from colums of pd dataframe
def mock_data():

  # We will mock ML modules, and just import the image frames
  frames_directory = "data/processed/"

  # Load each frame_0*.jpg image and store in list of np tuples (frame_id, avg_pixel_color)
  image_data = []

  print("Calculating variances for each image...")

  # Dont worry about regression for now
  
  # # Open all images in directory
  # for filename in os.listdir(frames_directory):
  #   if filename.endswith(".jpg"):
  #     # Extract frame_id from filename
  #     frame_id = int(filename.split("_")[1].split(".")[0])
  #     
  #     # Open image and convert to grayscale
  #     img = Image.open(os.path.join(frames_directory, filename))
  #     
  #     # Calculate average pixel color as rgb tuple (r, g, b)
  #     img_array = np.array(img)
  #     avg_pixel_color = tuple(np.mean(img_array, axis=(0, 1)).astype(int))

  #     # Append (frame_id, avg_pixel_color) to image_data list
  #     image_data.append((frame_id, avg_pixel_color))

  # Above works great, now do the same, but instead of avg_pixel_color, we will calculate the variance of the pixel colors in the image, and store that as a single value instead of a tuple. 

  start = time.perf_counter()

  # for filename in os.listdir(frames_directory):
  #   if filename.endswith(".jpg"):
  #     # Extract frame_id from filename
  #     frame_id = int(filename.split("_")[1].split(".")[0])
  #     
  #     # Open image and convert to grayscale
  #     img = Image.open(os.path.join(frames_directory, filename))
  #     
  #     # Calculate variance of pixel colors as single value
  #     img_array = np.array(img)

  #     #Filter out pixels that are pure black (0, 0, 0)
  #     #print(f"Processing {filename}...")
  #     #print("Original image shape:", img_array.shape)

  #     # Only grab pixels that are not pure black and collapse them into a single array
  #     img_array = img_array[~np.all(img_array == [0, 0, 0], axis=-1)]

  #     #print("Filtered image shape:", img_array.shape)

  #     # Compute variance of remaining pixels
  #     pixel_variance = np.var(img_array)

  #     # Append (frame_id, pixel_variance) to image_data list
  #     image_data.append((frame_id, pixel_variance))

  # Above works incredibly well, but we need parallization for speed
  
  # Do I/O here, and then parallelize the variance calculation
  np_images = []
  for filename in os.listdir(frames_directory):
    if filename.endswith(".jpg"):
      # Extract frame_id from filename
      frame_id = int(filename.split("_")[1].split(".")[0])
      
      # Open image and convert to grayscale
      img = Image.open(os.path.join(frames_directory, filename))
      
      # Convert image to numpy array and store in list with frame_id
      np_images.append((frame_id, np.array(img)))

  # Sort here
  np_images.sort(key=lambda x: x[0])

  # Now parallelize the variance calculation
  with ThreadPoolExecutor(max_workers=16) as executor:
    for result in executor.map(process_image, np_images):
      image_data.append(result)

  # Sort here just in case
  image_data.sort(key=lambda x: x[0])
 
  end = time.perf_counter()
  print(f"Processed {len(image_data)} images in {end - start:.2f} seconds.")

  return image_data

def fit_regression(X, y, degree): 
  # Normalize variances by dividing by max variance to get values between 0 and 1
  max_variance = max(y)
  y = [variance / max_variance for variance in y]

  # Fit polynomial of degree
  coefficients = np.polyfit(X, y, degree)
  # Calculate R^2 and RMSE
  y_pred = np.polyval(coefficients, X)
  r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
  rmse = np.sqrt(np.mean((y - y_pred) ** 2))

  print(f"Coefficients: {coefficients}")
  print(f"R^2: {r_squared:.4f}")
  print(f"RMSE: {rmse:.4f}")

  return coefficients, r_squared, rmse
