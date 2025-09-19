import os
import io
import base64
import numpy as np
import tensorflow as tf
import cv2
import pydicom
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from tensorflow.keras.models import load_model 
from werkzeug.utils import secure_filename

# --- Flask setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.secret_key = os.urandom(24)  # Needed for sessions

USERNAME = "admin"
PASSWORD = "@Sonnaeun1"

# --- Load model ---
model = load_model("densenet121_fold5.keras")
last_conv_layer_name = "conv5_block16_concat"

# --- Preprocess image ---
def preprocess_image(image_path):
    ext = image_path.lower().rsplit('.', 1)[-1]
    if ext == 'dcm':
        # Load DICOM
        ds = pydicom.dcmread(image_path)
        img = ds.pixel_array.astype(np.float32)
        img -= img.min()
        img /= (img.max() + 1e-8)
        img = (img * 255).astype(np.uint8)

        # Convert grayscale to 3 channel BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (224, 224))

        # Convert to float32 tensor and normalize to [0,1]
        img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0

    else:
        # Normal image files (PNG/JPG)
        img_raw = tf.io.read_file(image_path)
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0

    return img


# --- Generate Grad-CAM heatmap ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[0][pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()

# --- Apply heatmap and save result ---
def save_and_superimpose(img_color, heatmap, result_path, alpha=0.5):
    # img_color: expected to be a BGR uint8 numpy array of any size
    
    img_color = cv2.resize(img_color, (224, 224))
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    mask_3ch = cv2.merge([mask, mask, mask])
    masked_heatmap = cv2.bitwise_and(heatmap_color, mask_3ch)
    masked_img = cv2.bitwise_and(img_color, mask_3ch)

    superimposed = cv2.addWeighted(masked_img, 1 - alpha, masked_heatmap, alpha, 0)
    cv2.imwrite(result_path, superimposed)

# --- Convert DICOM to PNG bytes ---
def dicom_to_png_bytes(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    pixel_array = ds.pixel_array
    # Normalize pixel_array to 0-255 uint8
    pixel_array = pixel_array.astype(np.float32)
    pixel_array -= pixel_array.min()
    pixel_array /= pixel_array.max()
    pixel_array = (pixel_array * 255).astype(np.uint8)

    # Convert grayscale to 3-channel BGR for consistent preview
    img_bgr = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
    img_bgr = cv2.resize(img_bgr, (224, 224))

    # Encode to PNG bytes
    success, encoded_image = cv2.imencode('.png', img_bgr)
    if not success:
        return None
    return encoded_image.tobytes()

# --- Login route ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

# --- Logout route ---
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- Protect routes ---
@app.before_request
def require_login():
    # Allow access to login, static files; block others if not logged in
    if request.endpoint not in ('login', 'static') and not session.get('logged_in'):
        return redirect(url_for('login'))

# --- Preview route for AJAX preview ---
@app.route('/preview', methods=['POST'])
def preview():
    file = request.files.get('file')
    if not file:
        return jsonify({'preview': None})

    filename = secure_filename(file.filename)
    ext = filename.lower().rsplit('.', 1)[-1]

    try:
        if ext == 'dcm':
            # Convert dicom to PNG bytes
            png_bytes = dicom_to_png_bytes(file)
            if png_bytes is None:
                return jsonify({'preview': None})

            b64 = base64.b64encode(png_bytes).decode('utf-8')
            data_uri = f"data:image/png;base64,{b64}"
            return jsonify({'preview': data_uri})

        else:
            # For normal images just read and return base64 preview
            img_bytes = file.read()
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            mime = file.mimetype
            data_uri = f"data:{mime};base64,{b64}"
            return jsonify({'preview': data_uri})

    except Exception as e:
        print("Preview error:", e)
        return jsonify({'preview': None})

# --- Main page ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            ext = filename.lower().rsplit('.', 1)[-1]

            # Preprocess image tensor for model input
            img_tensor = preprocess_image(file_path)
            img_array = np.expand_dims(img_tensor, axis=0)
            pred_prob = model.predict(img_array)[0][0]
            pred_class = int(pred_prob > 0.5)

            # Prepare image array for save_and_superimpose
            if ext == 'dcm':
                ds = pydicom.dcmread(file_path)
                img = ds.pixel_array.astype(np.float32)
                img -= img.min()
                img /= (img.max() + 1e-8)
                img = (img * 255).astype(np.uint8)
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Create PNG preview for result page
                dicom_preview_filename = None
                png_bytes = dicom_to_png_bytes(file_path)
                if png_bytes:
                    dicom_preview_filename = "preview_" + filename.rsplit('.', 1)[0] + ".png"
                    preview_path = os.path.join(app.config["RESULT_FOLDER"], dicom_preview_filename)
                    with open(preview_path, "wb") as f:
                        f.write(png_bytes)

            else:
                img_color = cv2.imread(file_path)
                dicom_preview_filename = None
                if img_color is None:
                    flash("Error loading image for visualization", "danger")
                    return redirect(url_for('index'))

            # Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            result_filename = "gradcam_" + filename.rsplit('.', 1)[0] + ".png"
            result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
            save_and_superimpose(img_color, heatmap, result_path)

            return render_template(
                "result.html",
                filename=filename,
                result_image=result_filename,
                prob=round(pred_prob, 4),
                pred_class=pred_class,
                dicom_preview=dicom_preview_filename
            )
    return render_template("index.html")

# --- Run the server ---
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
