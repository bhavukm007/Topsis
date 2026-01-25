from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import smtplib
from email.message import EmailMessage
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ================= FILE READER (CSV / XLSX) =================
def read_input_file(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and XLSX files are supported")

# ================= TOPSIS FUNCTION =================
def run_topsis(input_file, weights, impacts, output_file):
    data = read_input_file(input_file)

    if data.shape[1] < 3:
        raise ValueError("Input file must contain at least 3 columns")

    matrix = data.iloc[:, 1:]

    if not np.all(matrix.applymap(np.isreal)):
        raise ValueError("From 2nd column onwards, values must be numeric")

    weights = np.array(weights, dtype=float)

    norm = np.sqrt((matrix ** 2).sum())
    norm_matrix = matrix / norm
    weighted = norm_matrix * weights

    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(weighted.iloc[:, i].max())
            ideal_worst.append(weighted.iloc[:, i].min())
        else:
            ideal_best.append(weighted.iloc[:, i].min())
            ideal_worst.append(weighted.iloc[:, i].max())

    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)

    data["Topsis Score"] = score
    data["Rank"] = score.rank(ascending=False).astype(int)

    data.to_csv(output_file, index=False)

# ================= EMAIL FUNCTION =================
def send_email(receiver_email, file_path):
    SENDER_EMAIL = "bhavukmahajan007@gmail.com"       # 🔁 replace once
    APP_PASSWORD = "alpjtomqazrytjno"  # replace once

    msg = EmailMessage()
    msg["Subject"] = "TOPSIS Result File"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg.set_content("Attached is your TOPSIS result file.")

    with open(file_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="octet-stream",
            filename=os.path.basename(file_path)
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        weights = request.form["weights"].split(",")
        impacts = request.form["impacts"].split(",")
        email = request.form["email"]

        if len(weights) != len(impacts):
            return "Error: Number of weights must equal number of impacts"

        for i in impacts:
            if i not in ['+', '-']:
                return "Error: Impacts must be + or -"

        if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
            return "Error: Only CSV and XLSX files allowed"

        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Unique result file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"topsis_result_{timestamp}.csv"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        # Run TOPSIS
        run_topsis(input_path, weights, impacts, output_path)

        # Send Email
        send_email(email, output_path)

        return f"Result generated and sent successfully! File saved as {output_filename}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
