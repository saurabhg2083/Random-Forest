from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            CRIM=float(request.form.get('CRIM')),
            ZN = float(request.form.get('ZN')),
            INDUS = float(request.form.get('INDUS')),
            CHAS = float(request.form.get('CHAS')),
            NOX = float(request.form.get('NOX')),
            RM = float(request.form.get('RM')),
            AGE = float(request.form.get('AGE')),
            DIS = float(request.form.get('DIS')),
            RAD = float(request.form.get('RAD')),
            TAX = float(request.form.get('TAX')),
            PTRATIO = float(request.form.get('PTRATIO')),
            B = float(request.form.get('B')),
            LSTAT = float(request.form.get('LSTAT')),
            MEDV = float(request.form.get('MEDV'))
        )
       
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5005, debug=True)
    


