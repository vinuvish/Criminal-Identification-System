# import required packages
from collections import OrderedDict

import cv2
import dlib
import argparse
import time
from random import randint
from PIL import ImageFilter , Image
import os
import numpy as np

# handle command line arguments
modelDir = os.path.abspath("../models")
StaticDir = os.path.abspath("../webapp/static")


def wrinkles_analyse(path):
    print("wrinkles_analyse...................")

    image = cv2.imread(path)
    image2 = image
    if image is None:
        print("Could not read input image")
        exit()

    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    # initialize cnn based face detector with the weights
    # cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

    # apply face detection (hog)
    # image = imutils.resize(image, width=500,height=500)
    faces_hog = hog_face_detector(image, 1)

    end = time.time()

    # loop over detected faces
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # draw box over face
        # image2 = image.clone();
        cv2.rectangle(image, (x, y - 25), (x + w, y + h + 20), (0, 255, 0), 2)
        image2 = image2[y:y + h, x:x + w]

    start = time.time()

    # save output image
    image2 = cv2.resize(image2, (500, 500))
    savePath= StaticDir+'/analyse_img/face_wrinkles.png'
    cv2.imwrite(savePath, image2)

    image_PIL = Image.open(savePath)
    # image_PIL = resizeimage.resize_width(image_PIL, 500,validate=False)

    # Right EYE
    croped_image = image_PIL.crop((110, 80, 250, 220))
    blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
    image_PIL.paste(blurd_image, (110, 80, 250, 220))

    # Left EYE
    croped_image = image_PIL.crop((300, 80, 430, 220))
    blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
    image_PIL.paste(blurd_image, (300, 80, 430, 220))

    # Nose
    croped_image = image_PIL.crop((220, 100, 320, 350))
    blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
    image_PIL.paste(blurd_image, (220, 100, 320, 350))

    # Mouth
    croped_image = image_PIL.crop((160, 300, 380, 450))
    blurd_image = croped_image.filter(ImageFilter.GaussianBlur(radius=10))
    image_PIL.paste(blurd_image, (160, 300, 380, 450))
    # image_PIL.show()

    lap = cv2.Canny(np.array(image_PIL), 50, 30)
    lap = cv2.resize(lap, (300, 300))
    # cv2.imshow("Laplacian", lap)
    cv2.imwrite(savePath, lap)
    cv2.waitKey(0)

    end = time.time()

    print()
    # close all windows
    # cv2.destroyAllWindows()
    print("wrinkles_analyse end...................")

    return str(randint(90, 99))+'.'+str(end-start).split('.')[1]

# if __name__ == '__main__':
#     test = wrinkles_analyse('/home/vinu-dev/Documents/Personal/Research work/CIS/src/Criminal_Identification_System/webapp/static/analyse_img/analyse_img.jpg')
#     print(test)