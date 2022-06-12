from flask import (Flask,  render_template, request, jsonify
                   )
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import logging
import pickle
import pandas as pd
import tensorflow_hub as hub
from keras.utils import CustomObjectScope

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

with CustomObjectScope({'KerasLayer': hub.KerasLayer}):
    # model = load_model('model_ml/my_model.h5')
    model2 = load_model('model_ml/my_model_detect.h5')

api_key = "154c991c-1c07-492b-907c-6d3945759fd1"

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY='dev')

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass


@app.route('/')
def index():
    return render_template('sendImage.html')


def cleaning_data(data):

    data = data.to_dict(orient='dict')
    pesticide_name = list(data['nama'].values())
    pesticide_image = list(data['image_src'].values())
    pesticide_link = list(data['product_link'].values())
    pesticide_tempat = list(data['tempat'].values())
    # pesticede_image = data.image_src.to_json()

    clean_data = []
    for i in range(len(data)):
        temporary_data = {}
        temporary_data['nama'] = pesticide_name[i]
        temporary_data['image_src'] = pesticide_image[i]
        temporary_data['product_link'] = pesticide_link[i]
        temporary_data['tempat'] = pesticide_tempat[i]
        clean_data.append(temporary_data)
    return clean_data


df = pickle.load(open("model_ml/recommend_data_fix.pkl", "rb"))
similarity = pickle.load(open("model_ml/similarity_fix.pkl", "rb"))


@app.route('/recommend/<pesticide_name>')
def recommend_pestisida_name(pesticide_name):
    indices = pd.Series(df.index, index=df['nama'])

    idx = indices[pesticide_name]

    sig = list(enumerate(similarity[idx]))  # Sort the names

    # Scores of the 5 most similar name
    sig = sorted(sig, key=lambda x: x[1], reverse=True)

    sig = sig[1:10]  # pesticides indicies
    pesticide_indices = [i[0] for i in sig]

    #   # Top 5 pesticide recommendation
    rec = df[['nama', 'kegunaan', 'tempat',
              'berat', 'image_src', 'product_link']].iloc[pesticide_indices]

    cleaning = cleaning_data(rec)

    return jsonify(recommendations=cleaning)

    # return render_template('resultRecommend.html', recommend = rec)


def recommend_pestisida(disease):
    indices = pd.Series(df.index, index=df['nama'])

    if disease == 'Leaf Blast':
        pesticide_name = 'Filia 525Se 250Ml Obat Hawar Daun Dan Blast Original'
    elif disease == 'Brown Spot':
        pesticide_name = 'Fungisida REMAZOLE - P 490 EC - 250 ml'
    elif disease == 'Hispa':
        pesticide_name = 'nararel 550 EC 400 ML'
    elif disease == 'Tungro':
        pesticide_name = 'Pupuk Sidafur 3 GR'
    elif disease == 'Blight':
        pesticide_name = 'Bakterisida Agrept 20 WP 50 Gram'
    else:
        return ''
    # Get the pairwsie similarity scores
    idx = indices[pesticide_name]
    # return idx
    # return idx
    # print(idx)
    sig = list(enumerate(similarity[idx]))  # Sort the names
    # return sig
    # Scores of the 5 most similar name
    sig = sorted(sig, key=lambda x: x[1], reverse=True)
    # return sig

    sig = sig[1:10]  # Book indicies
    pesticide_indices = [i[0] for i in sig]

    # Top 5 pesticide recommendation
    rec = df[['nama', 'kegunaan', 'tempat',
              'berat', 'image_src', 'product_link']].iloc[pesticide_indices]

    cleaning = cleaning_data(rec)
    # print(rec)
    return cleaning


@app.route('/resultmodel', methods=['POST'])
def result_model():
    auth_key = request.form['api_key']
    images = request.files['img']
    
    if auth_key == api_key:
        # path file
        path = os.path.join(app.config['UPLOAD_PATH'], images.filename)

        # save file
        if images.filename != '':
            images.save(path)

        img = tf.keras.preprocessing.image.load_img(
            path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        img = np.vstack([x])
        classes = model2.predict(img, batch_size=10)
        result = np.argmax(classes)

        # menghapus file
        #os.remove(os.path.join(
        #    app.config['UPLOAD_PATH'], images.filename))

        # ['Tungro', 'Hispa', 'Healthy', 'LeafBlast','Blight , 'BrownSpot']
        if result == 0:
            disease = 'Tungro'
        elif result == 1:
            disease = 'Hispa'
        elif result == 2:
            disease = ''
        elif result == 3:
            disease = 'Leaf Blast'
        elif result == 4:
            disease = 'Blight'
        else:
            disease = 'Brown Spot'

        # get pesticide recommend
        rec = recommend_pestisida(disease)

        return jsonify(penyakit=disease, recommendations=rec)


        # return web
        # return render_template('resultModel.html', training=str(classes), hasil=str(result), nama=disease,recommend=rec )
    else:
        return "API Key incorrect"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
