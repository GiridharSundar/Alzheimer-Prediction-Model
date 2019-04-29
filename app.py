from flask import Flask, render_template, request
import numpy as np
import pickle

fileObject = open('kNeighbours.pickle','rb')
b = pickle.load(fileObject)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('hospital.html')

@app.route('/result', methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        ASF = request.form['ASF']
        Age = request.form['Age']
        EDUC = request.form['EDUC']
        Group = request.form['Group']
        Hand = request.form['Hand']
        M_F = request.form['M/F']
        MMSE = request.form['MMSE']
        MR_Delay = request.form['MR_Delay']
        SES = request.form['SES']
        eTIV = request.form['eTIV']
        nWBV = request.form['nWBV']

        arr= []
        arr.append(ASF)
        arr.append(Age)
        arr.append(EDUC)
        arr.append(Group)
        arr.append(Hand)
        arr.append(M_F)
        arr.append(MMSE)
        arr.append(MR_Delay)
        arr.append(SES)
        arr.append(eTIV)
        arr.append(nWBV)

        arr2 = np.array([arr])

        from sklearn.preprocessing import LabelEncoder
        f = LabelEncoder()
        arr2[0]= f.fit_transform(arr2[0])
        arr2.astype(np.float64)
        prediction = b.predict(arr2)

        return render_template('result.html' , predict = prediction)

if __name__ == '__main__':
    app.run(debug=True)
