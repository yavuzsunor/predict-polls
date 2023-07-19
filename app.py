from flask import Flask, request, render_template
from src.pipeline.inference_pipeline import Inference, CustomData

application = Flask(__name__)

app = application

# Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(athlete=request.form.get('athlete'))

        predict_pipeline = Inference()
        json_response = predict_pipeline.predict_at_inference()
        return render_template('home.html', results = json_response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)