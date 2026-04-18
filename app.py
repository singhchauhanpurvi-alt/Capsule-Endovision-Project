# Import basic libraries
import os
import datetime
import numpy as np

# Import Flask and related modules
from flask import Flask, render_template, redirect, url_for, flash, request

# Import database (SQLite)
from flask_sqlalchemy import SQLAlchemy

# Import login system
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Import password hashing
from flask_bcrypt import Bcrypt

# Secure file upload
from werkzeug.utils import secure_filename

# Import TensorFlow (for ML model)
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Create Flask app
app = Flask(__name__)

# App configuration
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize database, encryption, login manager
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Global variables for ML model
_model = None
MODEL_LOADED = False
CLASS_NAMES = ['AVM', 'Normal', 'Ulcer']


# Function to load model only once
def get_model():
    global _model, MODEL_LOADED

    if _model is None:
        try:
            _model = tf.keras.models.load_model('best_model.keras')
            MODEL_LOADED = True
            print("Model loaded successfully.")
        except Exception as e:
            MODEL_LOADED = False
            _model = None
            print(f"Error loading model: {e}")

    return _model


# ------------------ DATABASE MODELS ------------------

# User table
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    # Relationship with image history
    history = db.relationship('ImageHistory', backref='owner', lazy=True)


# Image history table
class ImageHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_file = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date_uploaded = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# Load user for login system
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ------------------ MODEL PREDICTION FUNCTION ------------------
def model_predict(img_path):
    model = get_model()
    if model is None:
        return "Model Not Loaded", 0.0

    # Load image and resize to 256x256
    img = image.load_img(img_path, target_size=(256, 256), interpolation='bilinear')

    # Convert image to array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Model trained on [0,255] values, so no normalization here
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    return CLASS_NAMES[class_idx], confidence


# ------------------ ROUTES ------------------

# Home page
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


# Register page
@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('predict'))

    if request.method == 'POST':h
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)

        db.session.add(user)
        db.session.commit()

        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


# Login page
@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('predict'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=request.form.get('remember'))
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('predict'))

        flash('Login Unsuccessful. Please check email and password', 'danger')

    return render_template('login.html')


# Logout
@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))


# Prediction page
@app.route("/predict", methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            unique_filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(file_path)
            pred_class, confidence = model_predict(file_path)

            history_item = ImageHistory(
                image_file=unique_filename,
                prediction=pred_class,
                confidence=confidence,
                user_id=current_user.id
            )
            db.session.add(history_item)
            db.session.commit()

            return render_template(
                'predict.html',
                title='Prediction Result',
                image_file=unique_filename,
                prediction=pred_class,
                confidence=f"{confidence * 100:.2f}%"
            )

    return render_template('predict.html', title='Predict Endoscopy Image')


# History page
@app.route("/history")
@login_required
def history():
    user_history = ImageHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(ImageHistory.date_uploaded.desc()).all()
    return render_template('history.html', history=user_history)


# ------------------ RUN APP ------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    app.run(debug=True)
