import cv2
import numpy as np

def load_images(left_img_path, right_img_path):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_COLOR)
    return left_img, right_img

def compute_fundamental_matrix(left_img, right_img):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(left_img, None)
    kp2, des2 = sift.detectAndCompute(right_img, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    return F

def drawEpipolarLine(x, y, right_image, F):
    # Draw the epipolar line on the right image given a point (x, y) on the left image
    pt = np.array([x, y, 1]).reshape(-1, 1)
    line = np.dot(F, pt)
    a, b, c = line[0], line[1], line[2]

    height, width = right_image.shape[:2]
    x0, y0 = 0, int(-c/b)
    x1, y1 = width, int(-(c + a * width) / b)

    right_image_with_line = right_image.copy()
    cv2.line(right_image_with_line, (x0, y0), (x1, y1), (0, 255, 0), 2)

    return right_image_with_line

def create_composite_image(left_img, right_img, point):
    # Draw the point on the left image
    cv2.circle(left_img, point, 5, (0, 0, 255), -1)

    # Concatenate images horizontally
    return np.hstack((left_img, right_img))

def save_fundamental_matrix(F, filename):
    np.savetxt(filename, F, fmt="%0.6f")

def save_image(image, filename):
    cv2.imwrite(filename, image)   

# Paths to your images
left_img_path = "D:\\UCSB\\Fall 23\\CS 191I\\left.jpg"
right_img_path = "D:\\UCSB\\Fall 23\\CS 191I\\right.jpg"

left_img, right_img = load_images(left_img_path, right_img_path)
F = compute_fundamental_matrix(left_img, right_img)
save_fundamental_matrix(F, 'fundamental_matrix.txt')
# The point in the left image
x, y = (100, 150) # Replace with your coordinates

right_img_with_line = drawEpipolarLine(x, y, right_img, F)
output_img = create_composite_image(left_img, right_img_with_line, (x, y))
save_image(right_img_with_line, 'right_image_with_epipolar_line.jpg')

# Display the output
cv2.imshow("Stereo Image with Epipolar Line", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
