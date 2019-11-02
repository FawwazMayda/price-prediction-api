import lightgbm as lgb 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.externals import joblib

class Predictor:

    def __init__(self):
        self.le_color = None
        self.le_fuel = None
        self.le_trans = None
        self.st_price = None
        self.st_age  = None
        self.st_cc = None
        self.st_hp = None
        self.st_km = None
        self.st_price = None
        self.st_weight = None
        self.models = []
    def load_pickle(self):
        self.le_color = joblib.load("le_color.pkl")
        self.le_fuel = joblib.load("le_fuel.pkl")
        self.le_trans = joblib.load("le_trans.pkl")
        self.st_price = joblib.load("st_price.pkl")
        self.st_weight = joblib.load("st_weight.pkl")
        self.st_age = joblib.load("st_age.pkl")
        self.st_km = joblib.load("st_km.pkl")
        self.st_cc = joblib.load("st_cc.pkl")
        self.st_hp = joblib.load("st_hp.pkl")
        for i in range(4):
            m_path = "model_{i}.txt".format(i=i+1)
            print(m_path)
            m = lgb.Booster(model_file=m_path)
            print(type(m))
            self.models.append(m)
        
    def scale_weight(self,weight):
        weight = np.array(weight)
        return np.squeeze(self.st_weight.transform(weight.reshape(-1,1)))
    def scale_km(self,km):
        km = np.array(km)
        return np.squeeze(self.st_km.transform(km.reshape(-1,1)))
    def scale_cc(self,cc):
        cc = np.array(cc)
        return np.squeeze(self.st_cc.transform(cc.reshape(-1,1)))
    def scale_age(self,age):
        age = np.array(age)
        return np.squeeze(self.st_age.transform(age.reshape(-1,1)))
    def scale_hp(self,hp):
        hp = np.array(hp)
        return np.squeeze(self.st_hp.transform(hp.reshape(-1,1)))
    def inverse_price(self,price):
        price=np.array(price)
        return np.squeeze(self.st_price.inverse_transform(price.reshape(-1,1)))
    def price_prediction(self,data):
        #age,km,fueltype,hp,metcolor,automatic,cc,doors,weight
        print(data)
        price_pred = np.array([m.predict(data) for m in self.models])
        print("---------")
        print(price_pred)
        price_pred = np.mean(np.squeeze(price_pred))
        return price_pred




    




if __name__ == "__main__":
    print("BALLL")
    pp =Predictor()
    pp.load_pickle()
    print(pp.scale_age(14))
    print(pp.scale_hp(120))
    print(pp.scale_weight(1500))
    print(pp.inverse_price(0.763))


    data = np.array([[-1.771966,-0.574695,1,-0.768042,1,0,2.314976,3,1.758561]])
    print(pp.price_prediction(data))
