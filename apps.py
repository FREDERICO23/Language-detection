import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

cv = CountVectorizer()
le = LabelEncoder()

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    txt = request.form.get('text')
    y = cv.fit_transform([txt]).toarray() # convert text to bag of words model (Vector)
    x = y.transform(y)
    language = model.predict(x) # predict the language
    language = le.inverse_transform(language) # find the language corresponding with the predicted value
    
    output = language[0]

    return render_template('index.html', prediction='Language is in {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)