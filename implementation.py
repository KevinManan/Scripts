#Implement 3 image classifiers into one. 
import sys
sys.path.append('usr/local/lib/python/site-packages')
sys.path.append('/home/kevin/caffe/caffe/python')
import dlib
from PIL import Image
from skimage import io
from skimage import img_as_ubyte
import os, fnmatch, os.path, time
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt
import errno
import shutil
import urllib2
from bs4 import BeautifulSoup

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def MakeDirectory(Input_Directory):
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{}".format(Input_Directory))
    try:
        os.makedirs(directory)
    except OSError, exc:
        if exc.errno != errno.EEXIST:
            raise
    return directory

def GetImages(myurl, directory): 
    list = []
    i = 1
    k = 0
    url = myurl
    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    for j, img in enumerate(soup.findAll('img')):
        # print(img)
        temp = img.get('src')
        if temp[:1] == "/":
            image = myurl + temp
        else:
            image = temp
        # print (image)
        if (img.get('alt') != 'SJSU'):
            nametemp = img.get('alt')
            if len(nametemp) == 0:
                filename = os.path.join(directory, "foo" + str(i))
                i = i + 1
            else:
                filename = os.path.join(directory, nametemp)
            print(filename)
            imagefile = open(filename, 'wb')
            imagefile.write(urllib2.urlopen(image).read())
            imagefile.close()
        InputPath = r'/home/kevin/Documents/RacialIdentifier/Volatile Folder/UploadsPNG'
        if not os.path.exists(InputPath):
            os.makedirs(InputPath)
        imgareRaw = Image.open(filename)
        if(imgareRaw.format == 'JPEG'):
            imgareRaw.save("UploadsPNG/img{}.png".format(j),"PNG")
      
    # shutil.rmtree(directory)
    return list

def detect_faces(image):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()
    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left()    -((x.right()  - x.left())/4), 
                    x.top()     -((x.bottom() - x.top())/4),
                    x.right()   +((x.right()  - x.left())/4), 
                    x.bottom()  +((x.bottom() - x.top())/4)) 
                    for x in detected_faces]
    # print("Left{}".format(x.left()))
    # print("Right{}".format(x.right()))
    # print("Top{}".format(x.top()))
    # print("Bottom{}".format(x.bottom()))
    # //print("Face Frames{}".format(face_frames))
    return face_frames

temp = "a"
WHITE = [255,255,255]

def Cropper(DimensionX, DimensionY, scaleX, scaleY, preCropFile, postCropFile):
    for file in os.listdir(preCropFile):
        print("Resizing: {}".format(file))
        if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.png'): #for now let's keep this JPG files.
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
            widthEnd = (DimensionX / 2) + xCenter
            heightEnd = (DimensionY / 2) + yCenter
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




####Below is what I am adding to the implementation. 

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    try:
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        #Image Resizing
        img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    except:
        print "issue with the image"
    return img

#Read mean image for asians
mean_blobAZN = caffe_pb2.BlobProto()
with open('/home/kevin/Documents/RacialIdentifier/Asian Classifier --2k_iter | 74%/mean.binaryproto') as f:
    mean_blobAZN.ParseFromString(f.read())
mean_arrayAZN = np.asarray(mean_blobAZN.data, dtype=np.float32).reshape(
    (mean_blobAZN.channels, mean_blobAZN.height, mean_blobAZN.width))

#Read mean image for Blacks
mean_blobBLK = caffe_pb2.BlobProto()
with open('/home/kevin/Documents/RacialIdentifier/Black Classifier --2k iter | 88%/mean.binaryproto') as f:
    mean_blobBLK.ParseFromString(f.read())
mean_arrayBLK = np.asarray(mean_blobBLK.data, dtype=np.float32).reshape(
    (mean_blobBLK.channels, mean_blobBLK.height, mean_blobBLK.width))

#Read mean image for Caucasians
mean_blobWHT = caffe_pb2.BlobProto()
with open('/home/kevin/Documents/RacialIdentifier/White Classifier --2k iter | 76%/mean.binaryproto') as f:
    mean_blobWHT.ParseFromString(f.read())
mean_arrayWHT = np.asarray(mean_blobWHT.data, dtype=np.float32).reshape(
    (mean_blobWHT.channels, mean_blobWHT.height, mean_blobWHT.width))


#read model architectures
netAZN= caffe.Net('/home/kevin/Documents/RacialIdentifier/DeepLearning/caffe_models/caffe_model_2/caffenet_deploy_2.prototxt',
                '/home/kevin/Documents/RacialIdentifier/Asian Classifier --2k_iter | 74%/Snapshots/caffe_model_2_iter_2000.caffemodel',
                caffe.TEST)

netBLK = caffe.Net('/home/kevin/Documents/RacialIdentifier/DeepLearning/caffe_models/caffe_model_2/caffenet_deploy_2.prototxt',
                '/home/kevin/Documents/RacialIdentifier/Black Classifier --2k iter | 88%/Snapshots/caffe_model_2_iter_3000.caffemodel',
                caffe.TEST)

netWHT = caffe.Net('/home/kevin/Documents/RacialIdentifier/DeepLearning/caffe_models/caffe_model_2/caffenet_deploy_2.prototxt',
                '/home/kevin/Documents/RacialIdentifier/White Classifier --2k iter | 76%/Snapshots/caffe_model_2_iter_2000.caffemodel',
                caffe.TEST)

#Define image transformers asians
transformerA = caffe.io.Transformer({'data': netAZN.blobs['data'].data.shape})
transformerA.set_mean('data', mean_arrayAZN)
transformerA.set_transpose('data', (2,0,1))

#Define image transformers Blacks
transformerB = caffe.io.Transformer({'data': netBLK.blobs['data'].data.shape})
transformerB.set_mean('data', mean_arrayBLK)
transformerB.set_transpose('data', (2,0,1))

#Define image transformers Whites
transformerW = caffe.io.Transformer({'data': netWHT.blobs['data'].data.shape})
transformerW.set_mean('data', mean_arrayWHT)
transformerW.set_transpose('data', (2,0,1))

##for using submissions. **currently not using
test_idsA = []
predsA = []

test_idsB = []
predsB = []

test_idsW = []
predsW = []



##Preprocesssing the images collected.

# # Load image
# img_path = 'test2.png'
# image = io.imread(img_path)
# # Detect faces
# detected_faces = detect_faces(image)
# Create Directories fo
ScrapedPath = MakeDirectory("UploadsJPEG")
PNGPath = MakeDirectory("UploadsPNG")
CroppedPath = MakeDirectory("Cropped")
SizedPath = MakeDirectory("/home/kevin/Documents/RacialIdentifier/Volatile Folder/Sized")
# Scrapes
URL2DL = raw_input("-->")
GetImages(URL2DL, ScrapedPath)
for i, file in enumerate(os.listdir(PNGPath)):
    print("Cropping Faces: {}".format(file))
    image = io.imread('UploadsPNG/' +file)
    # Detect faces
    detected_faces = detect_faces(image)
    # Crop faces and plot
    for n, face_rect in enumerate(detected_faces):
        # print('{}'.format(face_rect))
        face = Image.fromarray(image).crop(face_rect)
        
        #Save cropped face to a png
        # face.save('New/sample{}'.format(n),face)
        plt.imsave('Cropped/sample{}{}.png'.format(n,i),face, format="png")
    
Cropper(224,            #DimensionX
        224,            #DimensionY
        300,            #scaleX
        300,            #scaleYs
        CroppedPath,    #preCropfolder
        SizedPath       #postCropfolder
)


#implmenting neural networks into the images. 

#to make the prediction
#below will only work for one image rather than an entire direcotry. 
#img_path = "/home/kevin/Documents/RacialIdentifier/DeepLearning/testImages/test14.jpg"

test_img_paths = [img_path for img_path in glob.glob("/home/kevin/Documents/RacialIdentifier/Volatile Folder/Sized/*")]
# test_img_paths = [img_path for img_path in glob.glob("/home/kevin/Documents/RacialIdentifier/Cropped/*")]

# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# imgT = detect_faces(img)

#for displaying purpose w/o edits.
#displayImg = cv2.imread(img_path)
#imgT = transform_img(imgT, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

#variable counters
AsianCount = 0
BlackCount = 0
WhiteCount = 0

AsianList = []
BlackList = []
WhiteList = []

#now to combing all the networks together. ''
for img_path in test_img_paths:
    print img_path
    if fnmatch.fnmatch(img_path, '*.png'):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #imgT = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        imgT = img.copy()
        netAZN.blobs['data'].data[...] = transformerA.preprocess('data', imgT)
        outA = netAZN.forward()
        pred_probasA = outA['prob']

        netBLK.blobs['data'].data[...] = transformerB.preprocess('data', imgT)
        outB = netBLK.forward()
        pred_probasB = outB['prob']

        netWHT.blobs['data'].data[...] = transformerW.preprocess('data', imgT)
        outW = netWHT.forward()
        pred_probasW = outW['prob']

        test_idsA = test_idsA + [img_path.split('/')[-1][:-4]]
        test_idsB = test_idsB + [img_path.split('/')[-1][:-4]]
        test_idsW = test_idsW + [img_path.split('/')[-1][:-4]]

        predsA = predsA + [pred_probasA.argmax()]
        predsB = predsB + [pred_probasB.argmax()]
        predsW = predsW + [pred_probasW.argmax()]

        print img_path
        #print "Are you Asian? 0 = Yes, 1 = No"
        if pred_probasA.argmax() :
            print "Processing"
        else:
            AsianCount += 1
            AsianList.append(img_path)
        	# print "You are Asian"
        #print "Are you black? 0 = Yes, 1 = No"
        if pred_probasB.argmax() :
            print "Processing"
        else:
            BlackCount += 1
            BlackList.append(img_path)
        #print "Are you White? 0 = Yes, 1 = No"
        if pred_probasW.argmax() :
            print "Processing"
        else:
            WhiteCount += 1
            WhiteList.append(img_path)
        print '-------'
#Below will show the image. if you don't want to see the image, comment the line below.
#cv2.imshow('test Image shown', displayImg)
#cv2.waitKey(0)



print "\n\nthese are the Asian pictures: "
for x in AsianList:
    print x
    cv2.imshow("Asian", cv2.imread(x))
    cv2.waitKey(0)
print "\n\nthese are the African pictures: "
for y in BlackList:
    print y
    cv2.imshow("Black", cv2.imread(y))
    cv2.waitKey(0)
print "\n\nthese are the Caucasian pictures: "
for z in WhiteList:
    print z
    cv2.imshow("White", cv2.imread(z))
    cv2.waitKey(0)


print "\n\n\nThere are a total of: "
print "Asians " + str(AsianCount)
print "Africans " + str(BlackCount)
print "Caucasian " + str(WhiteCount)


shutil.rmtree(ScrapedPath)
shutil.rmtree(PNGPath)
shutil.rmtree(CroppedPath)
