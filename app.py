from flask import Flask, render_template, request
from utils.predict import predict_digit
import os

app = Flask(__name__)

# Carpeta para guardar las imágenes subidas
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No se subió ningún archivo.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction="No se seleccionó archivo.")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Predicción usando el modelo CNN
    pred = predict_digit(filepath)
    return render_template("index.html", prediction=f"Predicción: {pred}", image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
import os
print(os.path.exists("model/cnn_model.h5"))
