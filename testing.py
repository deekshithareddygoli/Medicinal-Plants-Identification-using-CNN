from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')
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

@app.route('/', methods=['GET', 'POST'])
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
        img_path = os.path.join('static/images', file.filename)
        file.save(img_path)

        img = load_img(img_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]

        try:
            class_label = class_labels[class_idx]
        except IndexError as e:
            return f"Model prediction error: {e}"


        return render_template('prediction.html', prediction=class_label, image_file=file.filename)
    
if __name__ == '__main__':
    app.run(debug=True)

