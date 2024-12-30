import numpy as np
import argparse
import cv2
import os
import urllib.request
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set the relative directory to the models folder
DIR = os.path.dirname(os.path.realpath(__file__))  # Automatically gets the current directory
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")

# Dropbox direct download URL for the .caffemodel file
DROPBOX_MODEL_URL = "https://www.dropbox.com/scl/fi/m7u4z5v993fuzslegdveb/colorization_release_v2.caffemodel?rlkey=yb592p9mmo5l9wf794vgsyihs&st=ezrhcl14&dl=1"
MODEL_PATH = os.path.join(DIR, "models/colorization_release_v2.caffemodel")

# Check if the model file exists, and if not, download it from Dropbox
if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_PATH} from Dropbox...")
    try:
        urllib.request.urlretrieve(DROPBOX_MODEL_URL, MODEL_PATH)
        print(f"Download completed and saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_PATH)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Flask route for colorizing the image
@app.route('/colorize', methods=['POST'])
def colorize_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(image_file.filename)
    input_image_path = os.path.join(DIR, filename)
    image_file.save(input_image_path)

    # Load the input image
    image = cv2.imread(input_image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    # Save the output image (for server environments)
    output_image_path = os.path.join(DIR, "colorized_output.jpg")
    cv2.imwrite(output_image_path, colorized)

    return jsonify({"message": "Colorized image successfully!", "output_image": output_image_path})

if __name__ == "__main__":
    app.run(debug=True)
