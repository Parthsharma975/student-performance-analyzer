import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- Create sample data (same logic as the notebook) ---
np.random.seed(42)
n = 50
study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(60, 100, n)
previous_score = np.random.uniform(40, 90, n)
final_score = (0.4 * study_hours + 0.3 * attendance + 0.3 * previous_score
               + np.random.normal(0, 5, n))

X = np.column_stack([study_hours, attendance, previous_score])
y = final_score

# --- Train Linear Regression model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

METRICS = {
    "mse": round(float(mean_squared_error(y_test, y_pred)), 2),
    "r2": round(float(r2_score(y_test, y_pred)), 2),
    "samples": n,
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    values = {"study_hours": "", "attendance": "", "previous_score": ""}

    if request.method == "POST":
        try:
            values = {k: request.form.get(k, "").strip() for k in values}
            sh = float(values["study_hours"])
            att = float(values["attendance"])
            prev = float(values["previous_score"])
            if not (0 <= sh <= 24 and 0 <= att <= 100 and 0 <= prev <= 100):
                raise ValueError
            prediction = round(float(model.predict([[sh, att, prev]])[0]), 2)
        except ValueError:
            error = "Please enter valid numbers (hours 0-24, attendance and score 0-100)."

    return render_template("index.html", prediction=prediction, error=error,
                           values=values, metrics=METRICS)


if __name__ == "__main__":
    app.run(debug=False)
