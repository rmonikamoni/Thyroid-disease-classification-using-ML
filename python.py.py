from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('thyroid_model.pkl', 'wb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/pred', methods=['POST'])
def pred():
    age = request.form['Age']
    sex = request.form['Sex']
    bp = request.form['BP']
    cholesterol = request.form['Cholesterol']
    Na_to_k = request.form['Na_to_K']
    variables = [[int(age), int(sex), int(bp), int(cholesterol), np.log(float(Na_to_k))]]
    print(variables)
    output = model.predict(variables)
    prediction_text = label_encoder.inverse_transform(output)[0]
    return render_template('submit.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
