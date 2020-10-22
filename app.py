import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    
    
    
    final_features = np.array(int_features)
    final_features=final_features.reshape(1,-1)
    prediction = model.predict(final_features)
    
    if prediction[0]==1: 
        x='Yes, the account will default next month'
    else:
        x='No, the account will not default next month'
    
    return render_template('index.html', prediction_text=' {}'.format(x))


if __name__ == "__main__":
    app.run(debug=True)