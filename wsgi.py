#from app.main import app

from flask import Flask,request,jsonify
import pickle
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.special import comb

app = Flask(__name__)
with open("model_randomForest.pkl","rb") as f1:
	reg_model=pickle.load(f1)
with open("model_logisticRegression.pkl","rb") as f2:
	log_model=pickle.load(f2)
@app.route('/predictRandom', methods=['POST'])
def predictRan():
	Dict={
			"Age":0,
			"DailyRate":0,
			"DistanceFromHome":0,
			"Education":0,
			"EmployeeCount":0,
			"EmployeeNumber":0,
			"EnvironmentSatisfaction":0,
			"Gender":0,
			"HourlyRate":0,
			"JobInvolvement":0,
			"JobLevel":0,
			"JobSatisfaction":0,
			"MonthlyIncome":0,
			"MonthlyRate":0,
			"NumCompaniesWorked":0,
			"Over18":0,
			"OverTime":0,
			"PercentSalaryHike":0,
			"PerformanceRating":0,
			"RelationshipSatisfaction":0,
			"StandardHours":0,
			"StockOptionLevel":0,
			"TotalWorkingYears":0,
			"TrainingTimesLastYear":0,
			"WorkLifeBalance":0, 
			"YearsAtCompany":0,
			"YearsInCurrentRole":0,
			"YearsSinceLastPromotion":0,
			"YearsWithCurrManager":0,
			"BusinessTravel_Travel_Frequently":0,
			"BusinessTravel_Travel_Rarely":0, 
			"Department_Research & Development":0,
			"Department_Sales":0,
			"EducationField_Life Sciences":0,
			"EducationField_Marketing":0,
			"EducationField_Medical":0,
			"EducationField_Other":0,
			"EducationField_Technical Degree":0,
			"JobRole_Human Resources":0,
			"JobRole_Laboratory Technician":0,
			"JobRole_Manager":0,
			"JobRole_Manufacturing Director":0,
			"JobRole_Research Director":0,
			"JobRole_Research Scientist":0,
			"JobRole_Sales Executive":0,
			"JobRole_Sales Representative":0,
			"MaritalStatus_Married":0,
			"MaritalStatus_Single":0
		}
	for keys,values in Dict.items():
		Dict[keys]=request.json.get(keys)
	res_array = numpy.array(list(Dict.items()))
	res_array = numpy.delete(res_array,0,axis=1)
	res_array = res_array.reshape(1,48)
	
	x=list()
	for i in res_array[0]:
		x.append(float(i))
	n = numpy.array(x)
	n = n.reshape(1,48)

	yp=reg_model.predict(n)
	return {'Attrition':str(yp[0])}

@app.route('/predictLogistic', methods=['POST'])
def predictLog():
	Dict={
			"Age":0,
			"DailyRate":0,
			"DistanceFromHome":0,
			"Education":0,
			"EmployeeCount":0,
			"EmployeeNumber":0,
			"EnvironmentSatisfaction":0,
			"Gender":0,
			"HourlyRate":0,
			"JobInvolvement":0,
			"JobLevel":0,
			"JobSatisfaction":0,
			"MonthlyIncome":0,
			"MonthlyRate":0,
			"NumCompaniesWorked":0,
			"Over18":0,
			"OverTime":0,
			"PercentSalaryHike":0,
			"PerformanceRating":0,
			"RelationshipSatisfaction":0,
			"StandardHours":0,
			"StockOptionLevel":0,
			"TotalWorkingYears":0,
			"TrainingTimesLastYear":0,
			"WorkLifeBalance":0, 
			"YearsAtCompany":0,
			"YearsInCurrentRole":0,
			"YearsSinceLastPromotion":0,
			"YearsWithCurrManager":0,
			"BusinessTravel_Travel_Frequently":0,
			"BusinessTravel_Travel_Rarely":0, 
			"Department_Research & Development":0,
			"Department_Sales":0,
			"EducationField_Life Sciences":0,
			"EducationField_Marketing":0,
			"EducationField_Medical":0,
			"EducationField_Other":0,
			"EducationField_Technical Degree":0,
			"JobRole_Human Resources":0,
			"JobRole_Laboratory Technician":0,
			"JobRole_Manager":0,
			"JobRole_Manufacturing Director":0,
			"JobRole_Research Director":0,
			"JobRole_Research Scientist":0,
			"JobRole_Sales Executive":0,
			"JobRole_Sales Representative":0,
			"MaritalStatus_Married":0,
			"MaritalStatus_Single":0
		}
	for keys,values in Dict.items():
		Dict[keys]=request.json.get(keys)
	res_array = numpy.array(list(Dict.items()))
	res_array = numpy.delete(res_array,0,axis=1)
	res_array = res_array.reshape(1,48)
	x=list()
	for i in res_array[0]:
		x.append(float(i))
	n = numpy.array(x)
	n = n.reshape(1,48)
	yp=log_model.predict(n)
	return {'Attrition':str(yp[0])}

if __name__ == "__main__":
	app.run()
	