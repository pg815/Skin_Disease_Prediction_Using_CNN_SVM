from flask import Flask,request,render_template,flash
from werkzeug.utils import redirect
from model import Disease
import os

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('home.html')


@app.route('/', methods=["POST"])
def upload_file():
    if request.method == 'POST':
        disease = Disease()
        f = request.files['file']
        f.save(f.filename)
        print(f.filename)
        file = f.filename
        prediction = disease.classify(f.filename)
        result = "Predicted skin disease is : " + prediction[0][0]
        f = open("result.txt", "w+")
        f.write(result)
        f.close()
        f= open("result.txt","r")
        ans = f.read()
        f.close()
        os.remove(file)
        return render_template('result.html',results=ans)


@app.route('/home',methods = ['GET','POST'])
def home():
    return render_template('home.html')



app.run(host= '0.0.0.0', port= 5000, debug= True)