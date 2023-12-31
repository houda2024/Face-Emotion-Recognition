from flask import Flask, render_template, request
import cv2
import numpy as np 
from keras.models import model_from_json

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    image = request.files['select_file']

    image.save('static/file.jpg')

    image = cv2.imread('static/file.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        cropped = image[y:y+h, x:x+w]

    cv2.imwrite('static/after.jpg', image)

    try:
        cv2.imwrite('static/cropped.jpg', cropped)
    except:
        pass

    try:
        img = cv2.imread('static/cropped.jpg', 0)
    except:
        img = cv2.imread('static/file.jpg', 0)

    img = cv2.resize(img, (48,48))
    img = img/255

    img = img.reshape(1,48,48,1)

    # Load the model architecture from the JSON file
    with open('emotion_detection_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

    # Load the model weights
    model.load_weights('model_weights.h5')

    pred = model.predict(img)

    label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    pred = np.argmax(pred)
    final_pred = label_map[pred]

    return render_template('predict.html', data=final_pred)

if __name__ == "__main__":
    app.run(debug=True)
