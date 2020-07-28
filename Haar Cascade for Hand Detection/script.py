import cv2

for i in ["1.jpg","2.jpg","3.jpg"] :
    img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(64,64))
    cv2.imwrite(i,img)
