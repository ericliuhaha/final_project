import cv2


image = cv2.imread('t22.jpg')
image = cv2.resize(image, (350, 500), interpolation=cv2.INTER_AREA)
cv2.imwrite('tt22.jpg', image)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
