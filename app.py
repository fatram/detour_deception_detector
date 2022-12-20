# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request, Response, redirect, flash
from werkzeug.utils import secure_filename
import os
from static.lib.analyzer import Analyzer
 
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
app.secret_key = "very very secret key that how the heck do you even guess it?"

uploads_dir = os.path.join(app.root_path, 'static')
os.makedirs(uploads_dir, exist_ok=True)
verdict = ""
 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/', methods=["GET", "POST"])
# ‘/’ URL is bound with index() function.
def index():
    global verdict
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        file = request.files["file"]
        if file.filename == "":
            analyzer = Analyzer(app_root=uploads_dir)
            verdict = analyzer.predict()
        elif file:
            file.save(os.path.join(uploads_dir, secure_filename("clip.webm")))
            analyzer = Analyzer(with_file=True, app_root=uploads_dir)
            verdict = analyzer.predict()
            file.filename = ""
        return render_template("index.html", verdict=verdict)
    else:
        return render_template('index.html', verdict=verdict)
        

@app.route('/record_video', methods=['POST'])
def record_video():
    blob = request.data
    with open('static/clip.webm', 'wb') as f:
        f.write(blob)
    return Response(status=200)
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True, threaded=True)