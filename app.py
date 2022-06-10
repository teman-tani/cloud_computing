from flask import (Flask, render_template, request)
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd
from flask import jsonify

app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

model = load_model('model_ml/my_model.h5')

@app.route('/')
def index():
    return render_template('sendImage.html')

df = pickle.load(open("model_ml/recommend_data.pkl", "rb"))
similarity = pickle.load(open("model_ml/similarity.pkl", "rb"))
# list_movie = np.array(df["cleaned_desc"])

@app.route('/recommend/<pesticide_name>')
def recommend_pestisida_name(pesticide_name):
    indices = pd.Series(df.index, index=df['nama'])

    idx = indices[pesticide_name]

    sig = list(enumerate(similarity[idx]))  # Sort the names

    # Scores of the 5 most similar name
    sig = sorted(sig, key=lambda x: x[1], reverse=True)

    sig = sig[1:10]  # Book indicies
    pesticide_indices = [i[0] for i in sig]

    #   # Top 5 pesticide recommendation
    rec = df[['nama', 'kegunaan', 'tempat',
              'berat', 'image-src', 'product_link']].iloc[pesticide_indices]

    # response_json = {
        #     recommendations : rec
        # }
    # return jsonify(response_json)

    return render_template('resultRecommend.html', recommend = rec)

def recommend_pestisida(disease):
    indices = pd.Series(df.index, index=df['nama'])

    if disease == 'Leaf Blast':
        pesticide_name = 'Filia 525Se 250Ml Obat Hawar Daun Dan Blast Original'
    elif disease == 'Brown Spot':
        pesticide_name = 'Fungisida REMAZOLE - P 490 EC - 250 ml'
    elif disease == 'Hispa':
        pesticide_name = 'nararel 550 EC 400 ML'
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

    #   # Top 5 pesticide recommendation
    rec = df[['nama', 'kegunaan', 'tempat',
              'berat', 'image-src', 'product_link']].iloc[pesticide_indices]

    # print(rec)
    return rec

@app.route('/resultmodel', methods=['POST'])
def result_model():
        
        images = request.files['img']

        # save file
        if images.filename != '':
            images.save(os.path.join(
                app.config['UPLOAD_PATH'], images.filename))

        img = tf.keras.preprocessing.image.load_img(
            os.path.join(
                app.config['UPLOAD_PATH'], images.filename), target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        img = np.vstack([x])
        classes = model.predict(img, batch_size=10)
        result = np.argmax(classes[0])

        # menghapus file
        os.remove(os.path.join(
            app.config['UPLOAD_PATH'], images.filename))

        # ['LeafBlast', 'Healthy', 'BrownSpot', 'Hispa']
        if result == 0:
            disease = 'LeafBlast'
        elif result == 1:
            disease = 'Healthy'
        elif result == 2:
            disease = 'BrownSpot'
        else:
            disease = 'Hispa' 
            
        # get pesticide recommend
        # disease = disease
        rec = recommend_pestisida(disease = 'Brown Spot')

        # return json 
        # response_json = {
        #     disease : disease,
        #     recommendations : rec
        # }
        # return jsonify(response_json)

       
        # return web
        return render_template('resultModel.html', training=str(classes), hasil=str(result), nama=disease,recommend=rec )

if __name__ == '__main__':
    app.run(ssl_context='adhoc', host='0.0.0.0', port=8080, debug=True)
