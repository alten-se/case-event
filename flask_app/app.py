import os

from flask import Flask, render_template, request, jsonify
from flask.helpers import redirect, url_for
app = Flask(__name__)

from lungai.tf_config import force_CPU, silence_tf

force_CPU()
tf = silence_tf()


from lungai.paths import TRAINED_MODELS_PATH

from lungai.ai import AI


ai = AI.load(os.path.join(TRAINED_MODELS_PATH, "dummy"))

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
        label, conf = ai.predict_sound(f._file)
        # f.save(secure_filename(f.filename))
        return """
                AI Doctor: <br>
                    My diagnosis is: "{label}". <br>
                    I'm : {conf:.3%} confident of this
                """.format(label=label, conf=conf)

@app.route("/api", methods=["POST"])
def api():
    f = request.files["file"]
    label, conf = ai.predict_sound(f._file)
    res = {
        "label": label,
        "confidence": "{:.3}".format(conf)
    }
    return jsonify(res)

def create_app():
    return app


if __name__ == '__main__':
    app.run(debug = True)