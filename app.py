

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd


app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev_secret")

database_url = os.environ.get("DATABASE_URL")

if database_url:
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
else:
    # ✅ Local fallback database
    database_url = "sqlite:///users.db"

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ==============================
# Load Model and Scaler ONCE
# ==============================

scaler_path = os.path.join("models", "scaler.pkl")
model_path = os.path.join("models", "model.keras")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

model = load_model(model_path)

car_model_path = os.path.join("models", "LinearRegressionModel.pkl")
car_data_path = os.path.join("models", "Cleaned_Car_data.csv")

with open(car_model_path, "rb") as f:
    car_model = pickle.load(f)

car_data = pd.read_csv(car_data_path)

# ==============================
# Database Model
# ==============================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ==============================
# Routes
# ==============================

@app.route("/")
def index():
    return render_template("home.html")

# ---------- SIGNUP ----------

@app.route("/signup", methods=["GET", "POST"])
def signup():

    # 🚀 If already logged in → go to home
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        # 🔴 Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template("signup.html", error="Email already registered!")

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("signup.html")
# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("predict_page"))
        else:
            return render_template("login.html", error="Invalid Email or Password")

    return render_template("login.html")
# ---------- LOGOUT ----------

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index")) 

@app.route("/predict")
@login_required
def predict_page():
    return render_template("predict.html")
# ==============================
# Prediction Route
# ==============================
@app.route("/result", methods=["POST"])
@login_required

def result():
    try:
        cylinders = int(request.form["cylinders"])
        displacement = float(request.form["displacement"])
        horsepower = float(request.form["horsepower"])
        weight = float(request.form["weight"])
        acceleration = float(request.form["acceleration"])
        model_year = int(request.form["model_year"])
        origin = int(request.form["origin"])

        # ==============================
        # Input Validation
        # ==============================

        if not (3 <= cylinders <= 12):
            return jsonify({"error": "Cylinders must be between 3 and 12"})

        if not (50 <= displacement <= 600):
            return jsonify({"error": "Displacement must be between 50 and 600"})

        if not (40 <= horsepower <= 500):
            return jsonify({"error": "Horsepower must be between 40 and 500"})

        if not (1000 <= weight <= 7000):
            return jsonify({"error": "Weight must be between 1000 and 7000"})

        if not (5 <= acceleration <= 30):
            return jsonify({"error": "Acceleration must be between 5 and 30 seconds"})

        if not (60 <= model_year <= 90):
            return jsonify({"error": "Model year must be between 60 and 90"})

        if not (1 <= origin <= 3):
            return jsonify({"error": "Origin must be 1 (USA), 2 (Europe), or 3 (Japan)"})

        # ==============================
        # Prepare Data
        # ==============================

        values = [[
            cylinders,
            displacement,
            horsepower,
            weight,
            acceleration,
            model_year,
            origin
        ]]

        # Scale
        values = scaler.transform(values)

        # Convert to numpy
        values = np.array(values)

        # Predict (NO reshape needed)
        prediction = model.predict(values)
        prediction = float(prediction[0][0])

        if prediction < 0:
            prediction = 0

        return jsonify({
            "prediction": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})



@app.route("/car_predict")
@login_required
def car_predict_page():
    companies = sorted(car_data['company'].unique())
    car_models = sorted(car_data['name'].unique())
    years = sorted(car_data['year'].unique(), reverse=True)
    fuel_types = car_data['fuel_type'].unique()

    return render_template(
        "car_predict.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )
    
@app.route("/car_result", methods=["POST"])
@login_required
def car_result():

    try:
        company = request.form["company"]
        car_name = request.form["car_models"]
        year = int(request.form["year"])
        fuel_type = request.form["fuel_type"]
        kms = int(request.form["kilo_driven"])

        input_df = pd.DataFrame([{
            'name': car_name,
            'company': company,
            'year': year,
            'kms_driven': kms,
            'fuel_type': fuel_type
        }])

        prediction = car_model.predict(input_df)
        prediction = float(prediction[0])

        return jsonify({
            "car_price_prediction": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run()