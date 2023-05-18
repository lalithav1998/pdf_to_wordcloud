import os
from flask import Flask, render_template, request, send_file
from flask_cors import cross_origin, CORS
from utils.create_wordcloud import plot_wordcloud
from utils.make_upload_dir import make_upload_dir


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


@app.route("/")
@cross_origin()
def index():
    """
    displays the index.html page
    """
    result = '../static/white.png'
    if os.path.exists(os.path.join("static", "wordcloud.png")):
        os.remove(os.path.join("static", "wordcloud.png"))
    return render_template("index.html", result=result)


@app.route("/file_input", methods=["POST"])
@cross_origin()
def file_prediction():

    try:
        result = '../static/white.png'
        error = ""
        app.config['UPLOAD_FOLDER'] = os.path.join('.', 'uploads')
        make_upload_dir(app.config['UPLOAD_FOLDER'])
        if os.path.exists(os.path.join("static", "wordcloud.png")):
            os.remove(os.path.join("static", "wordcloud.png"))
        upload_file = request.files['fileinput']
        upload_file_path = ""

        upload_filename = upload_file.filename
        if upload_filename == "":
            raise Exception("Please Select a File")
        upload_file_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
        upload_file.save(upload_file_path)
        if os.path.getsize(upload_file_path) > 1000000:
            error = "File size greater than 1 MB"
        elif not upload_file_path.endswith(".pdf"):
            error = "Not a PDF"
        else:
            plot_wordcloud(upload_file_path)
            result = '../static/wordcloud.png'
        if os.path.exists(upload_file_path):
            os.remove(upload_file_path)

    except Exception as e:
        result = '../static/white.png'
        error = e
        if os.path.exists(upload_file_path):
            os.remove(upload_file_path)
        if os.path.exists(os.path.join("static", "wordcloud.png")):
            os.remove(os.path.join("static", "wordcloud.png"))

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)