import cv2
import numpy as np
# Load your image 
image_I1 = cv2.imread('D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8833.jpg')

# Define the homography matrix H
H = np.array([[1.5, 0.5, 0], [0, 2.5, 0], [0, 0, 1]])

# Apply the homography transformation
height, width = image_I1.shape[:2]
image_I1_prime = cv2.warpPerspective(image_I1, H, (width, height))
cv2.imwrite('warped_image_I1_prime.jpg', image_I1_prime)
# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors in both images
keypoints_I1, descriptors_I1 = sift.detectAndCompute(image_I1, None)
keypoints_I1_prime, descriptors_I1_prime = sift.detectAndCompute(image_I1_prime, None)

# Draw keypoints on each image
image_I1_with_keypoints = cv2.drawKeypoints(image_I1, keypoints_I1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image_I1_prime_with_keypoints = cv2.drawKeypoints(image_I1_prime, keypoints_I1_prime, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Initialize BFMatcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.match(descriptors_I1, descriptors_I1_prime)

# Draw the first 50 matches
matched_image = cv2.drawMatches(image_I1, keypoints_I1, image_I1_prime, keypoints_I1_prime, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the images with keypoints and matches
cv2.imwrite('Image_I1_with_Keypoints.jpg', image_I1_with_keypoints)
cv2.imwrite('Image_I1_prime_with_Keypoints.jpg', image_I1_prime_with_keypoints)
cv2.imwrite('Matches_between_I1_and_I1_prime.jpg', matched_image)

# Extract the keypoints locations from I1
points_I1 = np.float32([kp.pt for kp in keypoints_I1]).reshape(-1, 1, 2)

# Transform the points using the homography matrix
points_I1_transformed = cv2.perspectiveTransform(points_I1, H)

# Draw the groundtruth points on I1'
for pt in points_I1_transformed:
    cv2.circle(image_I1_prime, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)

# Save the image with groundtruth points
cv2.imwrite('I1_prime_with_groundtruth.jpg', image_I1_prime)

# Evaluate match accuracy
correct_matches = 0
threshold = 3  # Pixel threshold
for match in matches:
    # Groundtruth point in I1'
    groundtruth_pt = points_I1_transformed[match.queryIdx][0]
    
    # Matched point in I1'
    matched_pt = keypoints_I1_prime[match.trainIdx].pt

    # Calculate Euclidean distance
    distance = np.linalg.norm(np.array(groundtruth_pt) - np.array(matched_pt))

    # Count as correct if distance is within the threshold
    if distance <= threshold:
        correct_matches += 1

accuracy = (correct_matches / len(matches)) * 100
print(f"Match Accuracy: {accuracy:.2f}%")