# USAGE
# python detact_full_person.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from builtins import str

from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
import numpy as np
import imutils
import cv2
import dlib
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import time
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from statistics import mode
from backend.src.age_gender_Pridiction import age_gen_analyse
from backend.src.face_Wrinlkes import wrinkles_analyse
from backend.src.extract_face_featurs_data import face_analyse as face_analyse
import tensorflow as tf
import os

modelDir = os.path.abspath("../../models")
StaticDir = os.path.abspath("../../webapp/static")
print(StaticDir)

prototxt = modelDir + "/MobileNetSSD_deploy.prototxt.txt"
caffemodel= modelDir+"/MobileNetSSD_deploy.caffemodel"
confidence_range = 0.2



# parameters for loading data and images
haarcascade_frontalface = modelDir+"/emotion-recoganitation_trained_models/detection_models/haarcascade_frontalface_default.xml"
emotion_model_path = modelDir+"/emotion-recoganitation_trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5"
gender_model_path = modelDir+'/emotion-recoganitation_trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'
graph = tf.get_default_graph()



face_cascade = cv2.CascadeClassifier(haarcascade_frontalface)
eye_cascade = cv2.CascadeClassifier(modelDir+'/haarcascade_eye.xml')

mouth_cascade = cv2.CascadeClassifier(modelDir+'/Mouth.xml')
nose_cascade = cv2.CascadeClassifier(modelDir+'/nose.xml')



gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(haarcascade_frontalface)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)
# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]
# starting lists for calculating modes
emotion_window = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelDir+"/shape_predictor_68_face_landmarks.dat")

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading network...")
model = load_model(modelDir+'/hoodie_not_hoodie.model')


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
imageName = int(round(time.time() * 1000))









def checkFace(gray,frame) :
    isFaceAvailable = False
    dlibFace=False
    haarEyes =False
    haarFaces=False
    haarMouth = False

    rects = detector(gray, 0)
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        for (x, y) in shape:
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            if shape is not None :
                dlibFace = True
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                if faces is not None :
                    haarFaces = True
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    if eyes is not None :
                        haarEyes=True
                mouth = mouth_cascade.detectMultiScale(roi_gray)
                for(mx, my, mw, mh) in mouth :
                    if mouth is not None :
                        haarMouth = True





        if dlibFace and haarFaces and haarEyes and haarMouth :
            print('Dlib Face available : '+str(dlibFace) )
            print('Haar Face available : '+str(haarFaces))
            print('Haar Eye available : '+str(haarEyes))
            print('Haar Mouth available : '+str(haarMouth))

            # cv2.putText(frame, "Available", (200, 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            for (x, y) in shape:
                hog_face_detector = dlib.get_frontal_face_detector()
                faces_hog = hog_face_detector(frame, 1)
                for face in faces_hog:
                    x = face.left()
                    y = face.top()-50
                    w = face.right() - x
                    h = face.bottom() - y

                    # draw box over face
                    # image2 = image.clone();
                    # cv2.rectangle(frame, (x, y - 25), (x + w, y + h + 20), (0, 255, 0), 2)
                    image2 = frame[y:y + h, x:x + w]

                    imagePath = StaticDir+'/face/' + str(imageName) + '.jpg'
                    cv2.imwrite(imagePath, image2)
                    cv2.imwrite(StaticDir+'/analyse_img/analyse_img.jpg', frame)

                print("face Available")

                isFaceAvailable = True
                return isFaceAvailable



        else: return isFaceAvailable


def checkAbnormalCloth(image) :
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    (notHoodie, hoodie) = model.predict(image)[0]

    # build the label
    label = "Hoodies" if hoodie > notHoodie else "Not Hoodies Available"
    proba = hoodie if hoodie > notHoodie else notHoodie
    # label = "{}: {:.2f}%".format(label, proba * 100)
    return label , proba*100


def emotion_analyse(path):
    print('Emotion Analysing...........................')
    with graph.as_default():
        emotion_analyse_data = {'angry': 0, 'sad': 0, 'happy': 0, 'surprise': 0, 'fear': 0, 'angry': 0}

        # video_capture = cv2.VideoCapture(path)
        vs = VideoStream(path).start()

        # length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:

            # bgr_image = video_capture.read()[1]
            bgr_image = vs.read()
            if bgr_image is None:
                break

            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                    emotion_analyse_data['angry'] = emotion_analyse_data.pop('angry') + 1
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((0, 0, 255))
                    emotion_analyse_data['sad'] = emotion_analyse_data.pop('sad') + 1
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((255, 255, 0))
                    emotion_analyse_data['happy'] = emotion_analyse_data.pop('happy') + 1
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255))
                    emotion_analyse_data['surprise'] = emotion_analyse_data.pop('surprise') + 1
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))
                    emotion_analyse_data['fear'] = emotion_analyse_data.pop('fear') + 1

                color = color.astype(int)
                color = color.tolist()
                # print(emotion_text)

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

        print('Emotion Analysed...........................')

        return emotion_analyse_data




def analyse_video(path):
    pridictions_datas = {}
    pridictions_datas['hoodie'] = 'NotAvailable'
    terminate =False
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(path).start()

    # video_capture = cv2.VideoCapture(path)
    time.sleep(0.0)
    fps = FPS().start()
    count = 0
    # loop over the frames from the video stream

    face_Available = False

    while True and terminate is False:

        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        # frame = video_capture.read()[1]
        frame = vs.read()
        image = frame.copy()
        frame = imutils.resize(frame, width=1800)
        if not face_Available:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence >confidence_range:

                    # extract the index of the class label from the
                    # `detections`
                    idx = int(detections[0, 0, i, 1])

                    # if the predicted class label is in the set of classes
                    # we want to ignore then skip the detection
                    if CLASSES[idx] in IGNORE:
                        continue

                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)

                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    if confidence > 0.90:

                        if checkFace(gray,frame):
                            pridictions_datas["faceImagePath"] = str(imageName)+'.jpg'

                            face_Available = True
                        else:
                            # image = image[y - 100:endY + 100, y:startX]
                            with graph.as_default():
                                pridicted_labl, probability = checkAbnormalCloth(image)
                                face_mask=False
                                if pridicted_labl is 'Hoodies' and probability > 95 and checkFace(gray, frame) is None and face_mask:
                                    print(probability)
                                    pridictions_datas['abnormal_person'] = 'Available'
                                    terminate = True



        if face_Available:
            with graph.as_default():

                emotion_analyse_data = emotion_analyse(path)
                print(emotion_analyse_data)

                max_key = max(emotion_analyse_data, key=lambda k: emotion_analyse_data[k])
                if max_key is 'fear' or max_key is 'angry':
                    pridictions_datas["Identified_emotion"] =max_key
                    pridictions_datas["bg_color"] = '#FF0000'
                    pridictions_datas["Alart_msg"] = 'The Person Pridicted as Criminal'


                else:
                    pridictions_datas["Identified_emotion"] = max_key
                    pridictions_datas["bg_color"] = '#228B22'
                    pridictions_datas["Alart_msg"] = 'Normal Person'

                age_gen_pridiction_lable = age_gen_analyse(StaticDir+'/analyse_img/analyse_img.jpg')

                age = age_gen_pridiction_lable.split(',')[0]
                pridictions_datas["age"] = age
                pridictions_datas["gender"] = age_gen_pridiction_lable.split(',')[1]

                if int(age) >= 5 and int(age) <= 15:
                    pridictions_datas["age_range"] = 'Children'
                if int(age) >= 16 and int(age) <= 25:
                    pridictions_datas["age_range"] = 'Youth'
                if int(age) >= 26 and int(age) <= 40:
                    pridictions_datas["age_range"] = 'Adults'
                if int(age) >= 41:
                    pridictions_datas["age_range"] = 'Seniors'

                wrinkle_persetage=wrinkles_analyse(StaticDir+'/face/' + str(imageName) + '.jpg')
                pridictions_datas["wrinkle_persetage"] = str(wrinkle_persetage)

                face_analyse_data = face_analyse(StaticDir+'/face/' + str(imageName) + '.jpg')

            terminate = True

        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

    return pridictions_datas , emotion_analyse_data ,face_analyse_data



if __name__ == '__main__':
    pridictions_datas , emotion_analyse_data , face_analyse_data = analyse_video('../../webapp/static/video/faceuncoverdwithhoddie.mp4')
    print(pridictions_datas)
    print(emotion_analyse_data)
    print(face_analyse_data)

