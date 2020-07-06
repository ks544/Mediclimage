import argparse as ap
import cv2
import imutils 
import numpy as np
import os	
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy._lib.six import xrange
from scipy.cluster.vq import *

# Load the classifier, class names, scaler, number of clusters and vocabulary 
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print("No such directory {}\nCheck if the file exists").format(test_path)
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
else:
    image_paths = [args["image"]]
    
# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []
count = 0
for image_path in image_paths:
    im = cv2.imread(image_path)
    count+=1
    print(count)
    print(image_path)
    #if im == None:
   #     print "No such file {}\nCheck if the file exists".format(image_path)
     #   exit()
    kpts, des =  sift.detectAndCompute(im, None)
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
