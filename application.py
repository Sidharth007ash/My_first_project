import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template
application=Flask(__name__)
app=application

##import ridge regression and standard scalar pickle
ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scalar=pickle.load(open("models/scaler.pkl","rb"))
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predictdata", methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=request.form['Temperature']
        RH=request.form['RH']
        Ws=request.form['Ws']
        Rain=request.form['Rain']
        FFMC=request.form['FFMC']
        DMC=request.form['DMC']
        ISI=request.form['ISI']
        Classes=request.form['Classes']
        Region=request.form.get('Region')

        new_data_scaled=standard_scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

        

    else:
        return render_template('home.html')
if __name__=='__main__':
    app.run(host="0.0.0.0")
    ##when we give (0.0.0.0) as our host it means that it maps to the local host of any  system we are working with