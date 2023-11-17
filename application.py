# kyunki hmare paas pickle file hai 
import pickle 
from flask import Flask ,request , jsonify , render_template
from flask import Flask
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import Ridge Regressor model and standard scaler pickle file
ridge_model = pickle.load(open('models/ridge.pkl' , 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl' , 'rb'))

# ROute for Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata' , methods = ['GET' , 'POST'])
# Jo bhi values mai home mein derha hu unhe read karunga
def predict_datapoint():
    # agar post hai toh pickle ke sath interact krega or prediction dega
    if request.method == 'POST':
        # jo bhi humne webpage se diya hoga use hum yha se extract karenge
        # order same hona chahiye jo bhi order aapke model ke time pe tha
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # scaling
        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        result = ridge_model.predict(new_data_scaled)

        # 1st element ko access krne ke liye we write result[0]
        return render_template('home.html' , result=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0" , port=5001)
