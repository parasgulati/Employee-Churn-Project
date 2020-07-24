from flask import Flask,request
from model import predict

app = Flask(__name__)

@app.route('/predictLogistic',methods=['GET','POST'])
def predict():
  if request.method=='POST':
    data=request.form['data']
    prediction = model.predict.predictLogisticModel(data)
    return json.dumps('success')
  
@app.route('/predictRandomForest',methods=['GET','POST'])
def predict():
  if request.method=='POST':
    data=request.form['data']
    prediction = model.predict.predictRandomForestModel(data) 
    return json.dumps(str(prediction))

if __name__ == "__main__":
  app.run(host='0.0.0.0',debug=True)
