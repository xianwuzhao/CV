import cv2
import numpy as np
from matplotlib import pyplot as plt

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Load an image
image_path = 'D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8833.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale (SIFT works on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw the keypoints on the image
img_with_keypoints = cv2.drawKeypoints(gray, keypoints, image)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints")
plt.show()