
from flask import Flask, request, render_template, abort
import os
import logging
import backend.src.analyse_and_pridiction as analyse

frame_path = os.path.join('static', 'face')
analyse_img_path = os.path.join('static', 'analyse_img')



app = Flask(__name__)

app.config['FACE_FOLDER'] = frame_path
app.config['ANALYSE_IMG_FOLDER'] = analyse_img_path


@app.route('/')
def main():
    return render_template('index.html')


@app.route("/upload", methods=["POST"])
def upload():

    target='static/video/'
    try:
        for upload in request.files.getlist("file"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename

            # This is to verify files are supported
            ext = os.path.splitext(filename)[1]
            if (ext == ".mp4"):
                print("File supported moving on...")
                destination = "/".join([target, str(filename).replace(" ", "")])
                print("Accept incoming file:", filename)
                print("Save it to:", destination)

                upload.save(destination)
                pridictions_datas, emotion_analyse_data, face_analyse_data = analyse.analyse_video(destination)

                print(pridictions_datas)

                image = os.path.join(app.config['FACE_FOLDER'],pridictions_datas.pop('faceImagePath'))

                wrinkle_image = os.path.join(app.config['ANALYSE_IMG_FOLDER'], 'face_wrinkles.png')
                face_with_data = os.path.join(app.config['ANALYSE_IMG_FOLDER'], 'face_with_data.jpg')

                labels = ["Angry", "Fear", "Sad", "Happy", "Surprise"]
                values = [emotion_analyse_data.pop('angry'), emotion_analyse_data.pop('fear'), emotion_analyse_data.pop('sad'), emotion_analyse_data.pop('happy'), emotion_analyse_data.pop('surprise')]
                colors = ["#FF0000", "#800000", "#0000FF", "#228B22", "#ABCDEF"]


                return render_template('pridictPage.html', image_path=image, data=pridictions_datas,
                                       face_analyse_data=face_analyse_data, face_with_data=face_with_data
                                       ,wrinkle_image=wrinkle_image,set=zip(values, labels, colors))
            else:
                print("File Not supported...")
    except Exception as err:
        logging.error('An error has occurred whilst processing the file: "{0}"'.format(err))
        abort(400)


    return render_template('video.html')



if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
