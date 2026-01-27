from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ================= LOAD DATA =================
data = pd.read_csv("healthcare_data_science_dataset.csv")

# Drop patient_id
data.drop("patient_id", axis=1, inplace=True)

X = data.drop("disease_outcome", axis=1)
y = data["disease_outcome"]

# ================= ENCODING =================
le_gender = LabelEncoder()
le_smoking = LabelEncoder()
le_alcohol = LabelEncoder()
le_treatment = LabelEncoder()

X["gender"] = le_gender.fit_transform(X["gender"])
X["smoking"] = le_smoking.fit_transform(X["smoking"])
X["alcohol"] = le_alcohol.fit_transform(X["alcohol"])
X["treatment_type"] = le_treatment.fit_transform(X["treatment_type"])

# ================= SPLIT =================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= SCALING =================
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ================= MODEL =================
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        input_df = pd.DataFrame([{
            "age": int(request.form["age"]),
            "gender": request.form["gender"],
            "bmi": float(request.form["bmi"]),
            "blood_pressure": int(request.form["blood_pressure"]),
            "sugar_level": int(request.form["sugar_level"]),
            "cholesterol": int(request.form["cholesterol"]),
            "smoking": request.form["smoking"],
            "alcohol": request.form["alcohol"],
            "hospital_days": int(request.form["hospital_days"]),
            "treatment_type": request.form["treatment_type"],
            "follow_up_visits": int(request.form["follow_up_visits"])
        }])

        # Encode
        input_df["gender"] = le_gender.transform(input_df["gender"])
        input_df["smoking"] = le_smoking.transform(input_df["smoking"])
        input_df["alcohol"] = le_alcohol.transform(input_df["alcohol"])
        input_df["treatment_type"] = le_treatment.transform(input_df["treatment_type"])

        # Scale
        scaled_input = scaler.transform(input_df)

        pred = model.predict(scaled_input)[0]

        result = "Patient Recovered ✅" if pred == 1 else "Patient Not Recovered ❌"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
