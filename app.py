import os
import datetime
import numpy as np
from flask import Flask, render_template, redirect, url_for, flash, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Global variables for model
_model = None
MODEL_LOADED = False
CLASS_NAMES = ['AVM', 'Normal', 'Ulcer']

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

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    history = db.relationship('ImageHistory', backref='owner', lazy=True)

class ImageHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_file = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date_uploaded = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility function for preprocessing and prediction
def model_predict(img_path):
    model = get_model()
    if model is None:
        return "Model Not Loaded", 0.0
    
    # Load image with bilinear interpolation to match image_dataset_from_directory
    img = image.load_img(img_path, target_size=(256, 256), interpolation='bilinear')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # The notebook loss (23.68) indicates the model was trained on [0, 255] range.
    # Therefore, we do NOT divide by 255.0 here.
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    
    return CLASS_NAMES[class_idx], confidence

# Routes
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('register'))
    if request.method == 'POST':
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

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))  # ✅ pehle se sahi hai
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=request.form.get('remember'))
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))  # ✅ 'predict' → 'home'
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

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
            # Create a unique filename to avoid collisions
            unique_filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Predict
            pred_class, confidence = model_predict(file_path)
            
            # Save to history
            history_item = ImageHistory(
                image_file=unique_filename,
                prediction=pred_class,
                confidence=confidence,
                user_id=current_user.id
            )
            db.session.add(history_item)
            db.session.commit()
            
            return render_template('predict.html', 
                                 title='Prediction Result', 
                                 image_file=unique_filename,
                                 prediction=pred_class, 
                                 confidence=f"{confidence*100:.2f}%")
                                 
    return render_template('predict.html', title='Predict Endoscopy Image')

@app.route("/history")
@login_required
def history():
    user_history = ImageHistory.query.filter_by(user_id=current_user.id).order_by(ImageHistory.date_uploaded.desc()).all()
    return render_template('history.html', history=user_history)

# @app.route('/dashboard')
# @login_required
# def dashboard():
#     return render_template('dashboard.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
