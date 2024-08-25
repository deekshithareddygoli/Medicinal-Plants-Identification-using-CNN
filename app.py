import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, static_url_path='/static')

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('leaf (1).h5')

# Ensure class labels match your model's output
class_labels = ['Alpinia Galanga (Rasna)', 
                'Amaranthus Viridis (Arive-Dantu)', 
                'Azadirachta Indica (Neem)', 
                'Basella Alba (Basale)', 
                'Carissa Carandas (Karanda)', 
                'Ficus Auriculata (Roxburgh fig)', 
                'Hibiscus Rosa-sinensis', 
                'Jasminum (Jasmine)', 
                'Mangifera Indica (Mango)', 
                'Mentha (Mint)', 
                'Murraya Koenigii (Curry)', 
                'Nerium Oleander (Oleander)', 
                'Nyctanthes Arbor-tristis (Parijata)', 
                'Ocimum Tenuiflorum (Tulsi)', 
                'Piper Betle (Betel)', 
                'Plectranthus Amboinicus (Mexican Mint)', 
                'Punica Granatum (Pomegranate)', 
                'Santalum Album (Sandalwood)']

def process_image(img_path):
    img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_plant_species(img_array):
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_labels[class_idx]

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/plant_species', methods=['POST'])
def plant_species():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        img_array = process_image(img_path)
        class_label = predict_plant_species(img_array)

        return render_template('prediction.html', prediction=class_label, image_file=filename)

if __name__ == '__main__':
    app.run(debug=False)

