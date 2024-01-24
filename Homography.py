import cv2
import numpy as np

def compute_homography(src_points, dst_points):
    # Compute the homography matrix from point correspondences
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    return H

def apply_homography(image, H):
    # Transform the image using the homography matrix
    h, w = image.shape[:2]
    transformed_image = cv2.warpPerspective(image, H, (w, h))
    return transformed_image

def save_homography_matrix(H, filename):
    # Save the homography matrix to a file
    np.savetxt(filename, H, fmt='%0.6f')

# Replace these points with your corresponding points
src_points = np.float32([[742, 378], [791, 405], [774, 886], [830, 899]])
dst_points = np.float32([[621, 361], [686, 363], [578, 871], [659, 869]])

# Load your images 
image1 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\image_I1.jpg')
image2 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\image_S1.jpg')

# Compute the homography matrix
H = compute_homography(src_points, dst_points)

# Apply the homography to transform image1 into the perspective of image2
transformed_image = apply_homography(image1, H)

# Save the results
cv2.imwrite('transformed_image.jpg', transformed_image)
save_homography_matrix(H, 'homography_matrix.txt')

print("Homography transformation completed and files saved.")
