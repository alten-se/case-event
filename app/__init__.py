from flask import Flask, render_template, request
from flask.helpers import redirect, url_for
from werkzeug.utils import secure_filename
app = Flask(__name__)


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
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'
		
if __name__ == '__main__':
   app.run(debug = True)