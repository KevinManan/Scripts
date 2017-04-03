import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os, fnmatch, os.path, cv2, time
from matplotlib import pyplot as plt
import numpy as np
DimensionX = 224
DimensionY = 224
scaleX = 300
scaleY = 300
preCropFile = 'Images'
postCropFile = 'setImages'
temp = "a"
WHITE = [255,255,255]

def Cropper():
    for file in os.listdir(preCropFile):
        if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.jpeg'): #for now let's keep this JPG files.
       		path = preCropFile + os.sep + file
        	image = cv2.imread(path)
        	time.sleep(2)
        	# cv2.imshow("original", image)
        	# cv2.waitKey(0)
        	dimension = image.shape #returns a tuple.
        	(x, y, z) = dimension
        	# print x 
        	# print y 
        	res = image #this is just to keep the logic of this correct
        	while (x > scaleX) and (y > scaleY):
        		res = cv2.resize(res,None,fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)
        		dimension = res.shape
        		(x, y, z) = dimension
        	(x, y, z) = dimension
        	#this rescales our pictures. so that mian information is not lost. until the dimensions are 500 x 500. 
        	#next i would need to pad the picture if it's below DimensionX x DimensionY
        	if x < DimensionX or y < DimensionY:
        		if x > y:
        			bordersize = (DimensionY - y)
        		else:
        			bordersize = (DimensionX - x)
        		border = cv2.copyMakeBorder(res, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value= WHITE )
        		dimension = border.shape
        		(x, y, z) = dimension
        		Im = border
        		#Im = plt #this pads the dimension for x & y to be above the smallel dimension. 
        	else:
        		Im = res #this is used to keep the code uniform.
        	# Now that we have the images paded & rescaled to scaleX x scaleY > dimension > DimensionX x DimensionY. we can crop the image. 
        	dimension = Im.shape #our new dimensions. 
        	(x, y, z) = dimension 
        	xCenter = x/2
        	yCenter = y/2
        	widthStart = xCenter - 112
        	widthEnd = 112 + xCenter
        	heightEnd = 112 + yCenter
        	heightStart = yCenter - 112
        	# uncomment below if you want to supervise the cropping. 
        	try:
	        	crop = Im[widthStart:widthEnd, heightStart:heightEnd] #this is the new crop image. 
	        	# cv2.imshow("Cropped", crop)
	        	# cv2.waitKey(0)
	        except:
	        	print "You have hit a cropping error"
	        	# cv2.imshow("failed to crop", Im)
	        	# cv2.waitKey(0)
	        #if you wantto supervise the cropping, you can uncomment the imshow functions.
	        FinalPath = postCropFile + os.sep + file
	        cv2.imwrite(FinalPath, crop)

Cropper()

# ReadMe
# Dependencies:
# Run the following in terminal first: 
# export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
# needs openCV to be installed in order to work.
# In order for this script to work the File System must be labeled in the following manner:
# Parent File
#  |_ Images //pre crop Images
#  |_ setImages //post crop images.
