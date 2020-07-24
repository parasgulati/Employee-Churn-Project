from flask import Flask,request
from model import predict

app = Flask(__name__)

class LogisticRegressionModel(Resource):
  def get(self):
    print('Hi in logisticRegressionModelGet')
    predictQuery=request.args.get('Data')
    answer=predictLogisticModel(predictQuery)
    return {'Attrition':answer}


app.add_resource(LogisticRegressionModel, '/predictLogisticRegression')

'''
@app.route('/')

def index():
    return "Index Page"

@app.route('/predictLogistic',methods=['GET','POST'])
def predict():
    data = request.form.get('data')
    if data == None:
        return 'Got None'
    else:
        prediction = model.predict.predictLogisticModel(data) 
    return json.dumps(str(prediction))

@app.route('/predictRandomForest',methods=['GET','POST'])
def predict():
    data = request.form.get('data')
    if data == None:
        return 'Got None'
    else:
        prediction = model.predict.predictRandomForestModel(data) 
    return json.dumps(str(prediction))
'''
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
