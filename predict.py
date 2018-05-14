
import numpy as np
import cv2

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