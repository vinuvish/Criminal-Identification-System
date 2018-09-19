# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
import json
import os


modelDir = os.path.abspath("../models")
StaticDir = os.path.abspath("../webapp/static")

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(modelDir+'/shape_predictor_68_face_landmarks.dat')


facial_features_cordinates = {}

facial_features_data = {}


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth _Range", (48, 68)),
    ("Two_Eyebrow_Range", (21, 23)),
    ("Right_Eyebrow_Range", (17, 22)),
    ("Left_Eyebrow_Range", (22, 27)),
    ("Nose_Range", (27, 36)),
    ("For_head_Width", (0, 17)),

])
mouth_Range_Temp = 0
For_head_Width_Temp = 0


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())


def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):

    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    count = 0
    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]

        pts = shape[j:k]
        pts2 = pts
        # print(pts[-1]
        regions = str(pts2[0, 0] - pts2[-1, 1])
        redion_in_pixel = regions[1:]
        facial_features_cordinates[name] = redion_in_pixel
        if name in "Left_Eyebrow_Range":
            facial_features_cordinates[name] = int(redion_in_pixel) + 100

        # check if are supposed to draw the jawline
        if name in "For_head_Width":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

                For_head_Width_Temp = int(pts[16, 0] - pts[0, 0])

        if name in "For_head_Width":
            hull = cv2.convexHull(pts)


            facial_features_data['For_head_Width_Range'] = str(int(For_head_Width_Temp) / 50 * 20)[:5].replace("-", "")
            count += 30

            facial_features_data['Mouth_To_Jaw_Range'] =str(int(For_head_Width_Temp - mouth_Range_Temp) / 50 * 10)[:5].replace("-", "")


        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            if name in "Mouth _Range":
                hull = cv2.convexHull(pts)
                mouth_Range_Temp = int(redion_in_pixel);

                cv2.drawContours(overlay, [hull], -1, colors[i], 2)

                facial_features_data['Mouth_Range'] = str(int(facial_features_cordinates.get(name)) / 50 * 19)[:5]

                count += 30

                continue

            if name in "Nose_Range":
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, colors[i], 2)

                facial_features_data['Nose_Range'] = str(
                    mouth_Range_Temp - int(facial_features_cordinates.get(name)) / 50 * 10)[:5]

                count += 30
                continue
            if name in "Two_Eyebrow_Range":

                facial_features_data['Two_Eyebrow_Range']=str(int(facial_features_cordinates.get(name)) / 50 * 10)[:5]

            if name in "Right_Eyebrow_Range":

                facial_features_data['Right_Eyebrow_Range'] = str(int(facial_features_cordinates.get(name)) / 50 * 10)[:5]
            if name in "Left_Eyebrow_Range":

                facial_features_data['Left_Eyebrow_Range'] =  str(int(facial_features_cordinates.get(name)) / 50 * 10)[:5]
            hull = cv2.convexHull(pts)

            cv2.drawContours(overlay, [hull], -1, colors[i], 1)

            count += 30
            # print(name + " : " + str(facial_features_cordinates.get(name)))

    # # apply the transparent overlay
    # cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # hog_face_detector = dlib.get_frontal_face_detector()
    # faces_hog = hog_face_detector(output, 1)
    # for face in faces_hog:
    #     x = face.left()
    #     y = face.top() - 30
    #     w = face.right() - x
    #     h = face.bottom() - y
    #
    #     # draw box over face
    #     # image2 = image.clone();
    #     # cv2.rectangle(frame, (x, y - 25), (x + w, y + h + 20), (0, 255, 0), 2)
    #     image2 = output[y:y + h, x:x + w]

    cv2.imwrite(StaticDir+'/analyse_img/face_with_data.jpg',overlay)

    facial_features_data['face_analysed_image_path'] = 'face_with_data.jpg'

    return facial_features_data





def face_analyse(path):

    image = cv2.imread(path)
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)

        data = visualize_facial_landmarks(image, shape)

        return data

if __name__ == '__main__':
    data = face_analyse('../../webapp/static/analyse_img/analyse_img.jpg')
    print(data)