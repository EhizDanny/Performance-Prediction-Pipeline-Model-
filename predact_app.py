from flask import Flask, jsonify, request, render_template
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import json

app = Flask(__name__)
model = joblib.load(open('pipeline_model.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def home():
	""" Home View """

	return (jsonify(message="Welcome Home Gees!"))

@app.route('/predact_predict', methods = ['GET', 'POST'])
def predict():
    print('Prediction page')
    
    # try:
    Course_code = request.args.get('Course')
    Age = request.args.get('Age at enrollment')
    First_Credited_Units = request.args.get('Curricular units 1st sem (credited)')
    First_grade = request.args.get('Curricular units 1st sem (grade)')
    Second_Credited_Units = request.args.get('Curricular units 2nd sem (credited)')
    Second_grade = request.args.get('Curricular units 2nd sem (grade)')
    Scholarship = request.args.get('Scholarship holder')
    Debtor = request.args.get('Debtor')
    Previous_Qua_Grade = request.args.get('Previous qualification (grade)')
    Gender = request.args.get('Gender')
    Attendance = request.args.get('Daytime/evening attendance\t')
    Marital_Status = request.args.get('Marital status')
    Owing_Tuition = request.args.get('Tuition fees up to date')
    CGPA = request.args.get('CGPA')
    DropOff = request.args.get('DropOff_Rate')
    Appreciation = request.args.get('Appreciation_Rate')

        # Transform Inputs to strings and integers as required

    Course_code = int(Course_code)
    Age = float(Age)
    First_Credited_Units = float(First_Credited_Units)
    First_grade = float(First_grade)
    Second_Credited_Units = float(Second_Credited_Units)
    Second_grade = float(Second_grade)
    Scholarship = int(Scholarship)
    Debtor = int(Debtor)
    Previous_Qua_Grade = float(Previous_Qua_Grade)
    Gender = int(Gender)
    Attendance = int(Attendance)
    Marital_Status = int(Marital_Status)
    Owing_Tuition = int(Owing_Tuition)
    CGPA = int(CGPA)
    DropOff = int(DropOff)
    Appreciation = int(Appreciation)


    input_list = [[Course_code, Age, First_Credited_Units, First_grade, Second_Credited_Units,
                     Second_grade, Scholarship, Debtor, Previous_Qua_Grade, Gender,
                     Attendance, Marital_Status, Owing_Tuition, CGPA, DropOff, Appreciation]]

        # Standardize as necessary
#     sc = StandardScaler()

#         # select features to standardize
#     numerical_feats = [Age,First_Credited_Units,
#                             First_grade, Second_Credited_Units,
#                             Second_grade, Debtor, Previous_Qua_Grade,
#                             Attendance]

# 9500,	-0.589960,	-0.376848,	-0.168547,	-0.360905,	-0.776905,	0,	0,	-0.267980,	0,	1,	1,	1

#     standardized = sc.fit_transform(numerical_feats) 

    # final_input = np.unique(standardized + input_list)   
    # final_input = jsonify(final_input)
   
    pred = model.predict(input_list).tolist()
    # pred = model.predict([input_list])
    return(jsonify(prediction=pred))    
        
        
        

    # except:
        # return {"message":"ERROR!"}, 400

    
if __name__=="__main__":
	#debug=False for production use
	app.run(debug=True, host='0.0.0.0', port=9001)