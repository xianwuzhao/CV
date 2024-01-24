import cv2
import numpy as np

# Load two images 
image1 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8833.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8834.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT
sift = cv2.SIFT_create()

# Detect keypoints and descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Homography estimation
if len(good) > 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Homography Matrix:\n", H)
else:
    print("Not enough matches are found - {}/{}".format(len(good), 4))
