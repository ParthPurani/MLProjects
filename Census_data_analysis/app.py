import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# prediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 12)
	loaded_model = pickle.load(open("model.pkl", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

app = Flask(__name__) #Initialize the flask App

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    if request.method == 'POST':
		    to_predict_list = request.form.to_dict()
		    to_predict_list = list(to_predict_list.values())
		    to_predict_list = list(map(int, to_predict_list))
		    result = ValuePredictor(to_predict_list)		
		    if int(result)== 1:
		  	    prediction ='You will get the loan'
		    else:
		  	    prediction ='You will not get the loan'			
		    return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)