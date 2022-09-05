from cv2 import imwrite
from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
app = Flask(__name__)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

emotion_dict = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful',
                3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}

# Load json and create model
json_file = open('emotion_model.json', 'r')
load_model = json_file.read()
json_file.close()
emotion_model = model_from_json(load_model)

emotion_model.load_weights('emotion_model.h5')
img_ext = ['jpg', 'png', 'gif', 'svg', 'webp']
video_ext = ['mp4', 'avi', 'mov', 'mkv']


def generate_frame(file):
        
    
    camera = cv2.VideoCapture(file)
    
    while(camera.isOpened()):

        success, frame = camera.read()  # read the camera frame
        print(success)
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5,flags=cv2.CASCADE_SCALE_IMAGE)
           
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w
                , y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                cropped_img = np.expand_dims(np.expand_dims(
                    cv2.resize(roi_gray, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3, cv2.LINE_AA)

           
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cv2.destroyAllWindows()             


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video', methods=['POST', 'GET'])
def result():
    img = request.files['file1']

    name = img.filename

    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    im = name.split('.')

    if im[1].lower() in img_ext:
        img.save('static/file1.jpg')

        imgg = cv2.imread('static/file1.jpg')
        
        gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)

        for x, y, w, h in faces:
            cv2.rectangle(imgg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            croppedd = imgg[y:y+h, x:x+w]

        cv2.imwrite('static/after.jpg', imgg)

        try:
            cv2.imwrite('static/cropped.jpg', croppedd)
        except:
            pass

        try:
            image = cv2.imread('static/cropped.jpg', 0)
        except:
            image = cv2.imread('static/file1.jpg')
        image = cv2.resize(image, (48, 48))
        image = image/255
        image = np.expand_dims(np.expand_dims(image, -1), 0)

        pred = emotion_model.predict(image)
        prediction_index = int(np.argmax(pred))
        emotion_prediction = emotion_dict[prediction_index]
        return render_template('result.html', data=emotion_prediction)
    elif im[1].lower() in video_ext:
        return Response(generate_frame(name), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)
