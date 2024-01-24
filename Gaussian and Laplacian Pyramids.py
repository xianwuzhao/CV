import cv2
import numpy as np
import os

def create_and_save_pyramids(image_path, output_folder):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Crop to 512x512
    cropped_img = img[0:512, 0:512]

    # Gaussian Pyramid
    gaussian_pyramid = [cropped_img]
    for i in range(3):
        cropped_img = cv2.pyrDown(cropped_img)
        gaussian_pyramid.append(cropped_img)
        # Save each level of Gaussian Pyramid
        cv2.imwrite(f'{output_folder}/Gaussian_level_{i+1}.jpg', cropped_img)

    # Laplacian Pyramid
    laplacian_pyramid = []
    for i in range(3):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
        # Save each level of Laplacian Pyramid
        cv2.imwrite(f'{output_folder}/Laplacian_level_{i+1}.jpg', laplacian)

    # For the smallest level in the Laplacian pyramid, use the last level of the Gaussian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])
    cv2.imwrite(f'{output_folder}/Laplacian_level_4.jpg', gaussian_pyramid[-1])

create_and_save_pyramids('D:\\UCSB\\Fall 23\\CS 181\\hw06\\image_I.jpg', 'D:\\UCSB\\Fall 23\\CS 181\\hw06\\output_directory.jpg')
