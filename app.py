import numpy as np
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import sklearn
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        #get form data
        to_predict_list = request.form.to_dict()
        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(to_predict_list)
            #pass prediction to template
            return render_template('/predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
    pass
    pass


def preprocessDataAndPredict(feature_dict):
    
    test_data = {k:[v] for k,v in feature_dict.items()}  
    test_data = pd.DataFrame(test_data)

    #file = open("saved_model.sav","rb")
    
     #load trained model
    #trained_model = pickle.load(file)
    
    #read data
    data = pd.read_csv('heart_2020_cleaned.csv')
    
    df = data.copy()
    # encoding data
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cols = ['HeartDisease','Smoking', 'AlcoholDrinking', 'Stroke',
            'DiffWalking', 'Sex', 'AgeCategory',
            'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',
        'Asthma', 'KidneyDisease', 'SkinCancer']

    for i in cols:
        df[i] = data[[i]].apply(le.fit_transform)
    # split data   
    from sklearn.model_selection import train_test_split
    X =df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test= train_test_split(X,y, train_size=.8, random_state=32)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_train)
    y = scaler.transform(X_test)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='warn', n_jobs=None, penalty='l2',
                    random_state=None, solver='warn', tol=0.0001, verbose=0,
                    warm_start=False)
    model.fit(X_train, y_train)
    
    
    predict = model.predict(test_data)

    return predict
pass

if __name__ == "__main__":
    app.run()

'''
    bmi = request.form['bmi']
    sleep = request.form['sleep']
    #alcohol = request.form['']
    
    arr = np.array([bmi, sleep])
    pred = model.predict(arr)
        
    return render_template('index.html', data=pred)'''