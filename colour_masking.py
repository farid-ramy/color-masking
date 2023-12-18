import cv2
import numpy as np

img = cv2.imread("./colored.png", cv2.IMREAD_UNCHANGED)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

kernel = np.ones((7, 7), np.uint8)
morphology_result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# morphology_result = cv2.morphologyEx(morphology_result, cv2.MORPH_OPEN, kernel) # smoothing


black_image = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if morphology_result[i, j] != 0:
            black_image[i, j] = img[i, j]


combined_image = np.hstack((img, black_image))
cv2.namedWindow("Before and After", cv2.WINDOW_NORMAL)
cv2.imshow("Before and After", combined_image)
cv2.resizeWindow("Before and After", 800, 400)
cv2.waitKey(0)
cv2.destroyAllWindows()
