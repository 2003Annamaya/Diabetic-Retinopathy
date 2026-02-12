"""
Diabetic Retinopathy – Flask Web Application (PyTorch)
=======================================================
Upload a retinal fundus image and get an AI-powered DR severity prediction.
"""

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import config as cfg
from predict import predict, severity_description

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload a JPG/PNG retinal image."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(cfg.UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        result = predict(filepath)
        result["description"] = severity_description(result["class_name"])
        result["image_url"] = f"/static/uploads/{filename}"
        return render_template("index.html", result=result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(cfg.UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        result = predict(filepath)
        result["description"] = severity_description(result["class_name"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  Diabetic Retinopathy Detection – Web App")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
