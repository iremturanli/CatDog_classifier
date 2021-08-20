import argparse
import cv2
import os
import numpy as np
from PIL import Image as im


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",nargs="+",required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalcatface.xml",
	help="path to cat detector haar cascade")
args = vars(ap.parse_args())



print(args["image"][0])
# print(type(args["image"][1]))
print(len(args["image"]))


for i in range (len(args["image"])):
	image = cv2.imread(args["image"][i])
	# print(type(image))
	# print(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	detector = cv2.CascadeClassifier(args["cascade"])
	rects = detector.detectMultiScale(gray, scaleFactor=1.3,
		minNeighbors=10, minSize=(75, 75))


	os.chdir("yeni")

	if rects!=():
		
		# image_reshaped=image.reshape(image.shape[0],-1)
		# print(len(image_reshaped))
		# cv2.imshow("cat",image_reshaped)
		# cv2.waitKey(0)

		# np.savetxt("yeni",image_reshaped )
		# loaded_arr = np.loadtxt("yeni")
		# data=im.fromarray(loaded_arr)

		# if data.mode!='RGB':
		# 	data=data.convert('RGB')
		
		# 	data.save('yeni7.jpg')

		cv2.imwrite("{}.jpg".format(i),image)
	


		# array=np.reshape(image_reshaped,(224,224))

		# data=im.fromarray(array)
		# data.save('yeni.jpg')



# print(type(image))
# print(rects)








# for (i, (x, y, w, h)) in enumerate(rects):
# 	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 	cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# show the detected cat faces
# cv2.imshow("Cat Faces", image)
# cv2.waitKey(0)
