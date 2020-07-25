from flask import Flask,request
import predict
from flask_restful import Resource, Api
from json import dumps
import json 

app = Flask(__name__)
api = Api(app)


class LogisticRegressionModel(Resource):
	def get(self):
		
		Dict={
			'Age':0,
			'BusinessTravel':0,
			'DailyRate':0,
			'Department':0,
			'DistanceFromHome':0,
			'Education':0,
			'EducationField':0,
			'EmployeeCount':0,
			'EmployeeNumber':0,
			'EnvironmentSatisfaction':0,
			'Gender':0,
			'HourlyRate':0,
			'JobInvolvement':0,
			'JobLevel':0,
			'JobRole':0,
			'JobSatisfaction':0,
			'MaritalStatus':0,
			'MonthlyIncome':0,
			'MonthlyRate':0,
			'NumCompaniesWorked':0,
			'Over18':0,
			'OverTime':0,
			'PercentSalaryHike':0,
			'PerformanceRating':0,
			'RelationshipSatisfaction':0,
			'StandardHours':0, 
			'StockOptionLevel':0,
			'TotalWorkingYears':0,
			'TrainingTimesLastYear':0,
			'WorkLifeBalance':0,
			'YearsAtCompany':0,
			'YearsInCurrentRole':0,
			'YearsSinceLastPromotion':0,
			'YearsWithCurrManager':0
		}
		
		for keys,values in Dict.items():
			Dict[keys]=request.args.get(keys)
		answer=predict.predictLogisticModel(Dict)
		return {'Attrition':answer}

class RandomForestRegressorModel(Resource):
	def get(self):
		Dict={
			'Age':0,
			'BusinessTravel':0,
			'DailyRate':0,
			'Department':0,
			'DistanceFromHome':0,
			'Education':0,
			'EducationField':0,
			'EmployeeCount':0,
			'EmployeeNumber':0,
			'EnvironmentSatisfaction':0,
			'Gender':0,
			'HourlyRate':0,
			'JobInvolvement':0,
			'JobLevel':0,
			'JobRole':0,
			'JobSatisfaction':0,
			'MaritalStatus':0,
			'MonthlyIncome':0,
			'MonthlyRate':0,
			'NumCompaniesWorked':0,
			'Over18':0,
			'OverTime':0,
			'PercentSalaryHike':0,
			'PerformanceRating':0,
			'RelationshipSatisfaction':0,
			'StandardHours':0, 
			'StockOptionLevel':0,
			'TotalWorkingYears':0,
			'TrainingTimesLastYear':0,
			'WorkLifeBalance':0,
			'YearsAtCompany':0,
			'YearsInCurrentRole':0,
			'YearsSinceLastPromotion':0,
			'YearsWithCurrManager':0
		}
		for keys,values in Dict.items():
			Dict[keys]=request.args.get(keys)
		answer=predict.predictRandomForestModel(Dict)
		return {'Attrition':answer} 
	
api.add_resource(LogisticRegressionModel, '/predictLogisticRegression')
api.add_resource(RandomForestRegressorModel, '/predictRandomForest')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)