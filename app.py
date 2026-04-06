import uvicorn
from fastapi import FastAPI
from Banknote import BankNote
import pickle 


app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variaance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]

    prediction = classifier.predict([[variaance,skewness,curtosis,entropy]])
    if(prediction[0] > 0.5):
        prediction = "Fake Note"
    else:
        prediction = "Its a Bank Note"
    
    return{
        "prediction": prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host= "127.0.0.1",  port= 5001)