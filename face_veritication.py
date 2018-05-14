
import numpy as np
import cv2
from model import create_model
from align import AlignDlib
import argparse
import dlib


#load model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('nn4.small2.v1.h5')
embedded = np.load("embbed_face.npy")
label = np.load("label_singer.npy")
detec = dlib.get_frontal_face_detector()
alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')
# face verification
## calculate distance between 2 image
def distance(emb_im1,emb_im2):
    return np.sum(np.square(emb_im1-emb_im2))
def predict_face(image,threshold):
    #image = cv2.imread(image)
    img_cvt = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_normal = img_cvt/255
    emb_image = nn4_small2_pretrained.predict(np.expand_dims(img_normal,axis=0))
    dist = []
    for i in range(len(embedded)):
        dist.append(distance(embedded[i],emb_image))
        min_dist = np.min(np.array(dist))
        if min_dist > threshold:
            text = "Stranger"
        else :
            arg_min = np.argmin(np.array(dist))
            text = label[arg_min]
    return text     
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help = "Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
rects = detec(image,1)
if len(rects) == 1 :
    for rect in rects : 
        x,y,w,h = rect.left(),rect.top(),rect.right(),rect.bottom()
        
        roi = image[y:h,x:w]
        bb = alignment.getLargestFaceBoundingBox(image)
        image_alignment = alignment.align(96, image, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        text = predict_face(image_alignment,4)
        cv2.putText(image,text,(x,y-10),cv2.FONT_ITALIC,1,(0,255,0),2)
        cv2.rectangle(image,(x,y),(w,h),(0,255,0),2)
cv2.imshow("image",image)
cv2.waitKey()
cv2.destroyAllWindows()