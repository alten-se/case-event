import os

from flask import Flask, render_template, request
from flask.helpers import redirect, url_for
from werkzeug.utils import secure_filename
app = Flask(__name__)

from lungai.force_cpu import force_CPU

force_CPU()

from lungai.load_model import load_trained_model
from lungai.paths import TRAINED_MODELS_PATH
from lungai.evaluate import eval_sound
from lungai.data_extraction import get_data




from lungai.model import AI


model = AI.load(os.path.join(TRAINED_MODELS_PATH, "dummy")).model
_, _, label_dict = get_data()



@app.route("/")
def index():
    return redirect(url_for("upload"))

@app.route('/upload')
def upload():
    return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        label, conf = eval_sound(f._file, model, label_dict)
        # f.save(secure_filename(f.filename))
        return """
AI Doctor: <br>
    My diagnosis is: "{label}". <br>
    im : {conf:.3%} confident of this""".format(label=label, conf=conf)

if __name__ == '__main__':
    app.run(debug = True)