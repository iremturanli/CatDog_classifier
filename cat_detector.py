import argparse
import cv2
import os
import numpy as np



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalcatface.xml",
	help="path to cat detector haar cascade")
args = vars(ap.parse_args())



image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(gray, scaleFactor=1.3,
	minNeighbors=10, minSize=(75, 75))

print(image)
print(rects)


os.chdir("yeni")
if rects!=():
# 	# while True:

	cv2.imwrite('irem4.jpg',image)
		









# for (i, (x, y, w, h)) in enumerate(rects):
# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 	cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# show the detected cat faces
# cv2.imshow("Cat Faces", image)
# cv2.waitKey(0)
