import cv2

# Paths to your images
image_paths = ['D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8833.jpg', 'D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8834.jpg', 'D:\\UCSB\\Fall 23\\CS 181\\hw07\\IMG_8835.jpg']

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Dictionary to hold keypoints and descriptors for each image
keypoints_and_descriptors = {}

for path in image_paths:
    # Read each image
    image = cv2.imread(path)
    if image is None:
        print(f"Image {path} not found")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Store keypoints and descriptors
    keypoints_and_descriptors[path] = (keypoints, descriptors)

    # (Optional) Draw and display keypoints
    image_with_keypoints = cv2.drawKeypoints(gray, keypoints, image)
    # Save the image with keypoints
    output_filename = f"{path.split('.')[0]}_with_keypoints.jpg"
    cv2.imwrite(output_filename, image_with_keypoints)
    print(f"Saved: {output_filename}")