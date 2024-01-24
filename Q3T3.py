import cv2
import numpy as np

# Load the images
image1 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8833.jpg')
image2 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8834.jpg')
image3 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8835.jpg')

# Assuming H12 and H23 are the homography matrices
# Replace these with your actual homography matrices
H12 = np.identity(3)  # Example identity matrix
H23 = np.identity(3)  # Example identity matrix

# Warp Image 1 to Image 2's perspective
warped_image1_to_2 = cv2.warpPerspective(image1, H12, (image1.shape[1] + image2.shape[1], image1.shape[0]))

# Warp Image 2 to Image 3's perspective
warped_image2_to_3 = cv2.warpPerspective(image2, H23, (image2.shape[1] + image3.shape[1], image2.shape[0]))

# Warp the result of Image 1 to Image 3's perspective
warped_image1_to_3 = cv2.warpPerspective(warped_image1_to_2, H23, (warped_image1_to_2.shape[1] + image3.shape[1], warped_image1_to_2.shape[0]))

# Create a panorama canvas
height = max(warped_image1_to_3.shape[0], warped_image2_to_3.shape[0], image3.shape[0])
width = warped_image1_to_3.shape[1] + warped_image2_to_3.shape[1] + image3.shape[1]
panorama = np.zeros((height, width, 3), dtype=np.uint8)

# Place images on the panorama canvas
# Note: This is simplified and assumes left-to-right arrangement without vertical alignment issues
panorama[:warped_image1_to_3.shape[0], :warped_image1_to_3.shape[1]] = warped_image1_to_3
panorama[:warped_image2_to_3.shape[0], warped_image1_to_3.shape[1]:warped_image1_to_3.shape[1] + warped_image2_to_3.shape[1]] = warped_image2_to_3
panorama[:image3.shape[0], warped_image1_to_3.shape[1] + warped_image2_to_3.shape[1]:] = image3

# Crop the panorama to remove black areas
# Find all non-black pixels
coords = np.column_stack(np.where(panorama.any(axis=2)))
# Calculate the bounding box of non-black pixels
x_min, y_min = np.min(coords, axis=0)
x_max, y_max = np.max(coords, axis=0)

# Crop to the bounding box of non-black pixels
cropped_panorama = panorama[x_min:x_max+1, y_min:y_max+1]
# Save or display the panorama
cv2.imwrite('panorama.jpg', panorama)