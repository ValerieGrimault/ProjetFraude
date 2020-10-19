#import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame(np.array([int_features]))
    prediction = model.predict_proba(final_features)

    output = prediction[0]

    return render_template('index.html', donnees_prediction='Données en entrée :{}'.format(int_features), prediction_text='La prédiction pour la fraude est :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)