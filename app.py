from flask import  Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data = 'Abhrajit Pal')

@app.route('/predction', methods = ['POST'])
def predction():
    model = load_model('model.h5')
    img = request.files['img']
    img.save('img.jpg')
    test_image = image.load_img('img.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'DOG'

    else:
        prediction = 'CAT'


    return  render_template('predection.html', data = prediction)


if __name__ == '__main__':
    app.run(debug = True)



