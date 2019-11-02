from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import Predictor
import numpy as np

from bigml.deepnet import Deepnet
from bigml.api import BigML
# Downloads and generates a local version of the DEEPNET,
# if it hasn't been downloaded previously.
deepnet = Deepnet('deepnet/5dba7bb15a21395ce200035a',
                  api=BigML("medias",
                            "1bbcaec3bdce36230a99d91fbf5597d0c5ea4fc4",
                            domain="bigml.io"))
pp = Predictor()
pp.load_pickle()
app = Flask(__name__)
CORS(app)
@app.route("/api/check",methods=["GET","POST"])
def give():
    return jsonify({"status":"OK","message":"Hello"})
@app.route("/api/predict",methods=['POST'])
def predict():
    #Get Request Data
    content = request.json
    age = content['Age'] #numeric
    km = content['KM'] #numerci
    fueltype = content['FuelType'] #string
    if fueltype=='CNG':
        fueltype = 0
    elif fueltype=='Diesel':
        fueltype = 1
    elif fueltype=='Petrol':
        fueltype = 2
    hp = content['HP'] #numeric
    metcolor = content['MetColor'] #binary
    auto = content['Automatic'] #binary
    cc = content['CC'] #numeric
    doors = content['Doors'] #numeric
    weight = content['Weight'] #weight

    #Standarized
    age = pp.scale_age(age)
    km = pp.scale_km(km)
    hp = pp.scale_hp(hp)
    cc = pp.scale_cc(cc)
    weight = pp.scale_weight(weight)
    #age,km,fueltype,hp,metcolor,automatic,cc,doors,weight
    d = np.array([[age,km,fueltype,hp,metcolor,auto,cc,doors,weight]])
    price = pp.price_prediction(d)
    price = pp.inverse_price(price)
    print(content)
    print(price.item(0))
    return jsonify({"data": content,"price":price.item(0)})

@app.route("/api/predict_big",methods=["POST"])
def predict_big():
    #Get Request Data
    content = request.json
    print(content)
    input_data = content
    price_pred = deepnet.predict(input_data,full=True)
    print(price_pred)
    return jsonify({"data":content,"prediction":price_pred})

..

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port=8080)