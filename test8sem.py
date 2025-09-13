import streamlit as st
import pandas as pd
import hashlib
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.express as px
import base64

# ---------- Background Helper ----------
def local_css_with_background(css_file, image_file):
    with open(image_file, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode()
    with open(css_file, "r") as f:
        css = f.read().replace("{{BACKGROUND_IMAGE}}", base64_img)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---------- Session State Initialization ----------
for key in ['logged_in', 'username', 'result', 'pdf_file', 'input_data', 'predicted']:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- Constants ----------
USER_DATA_FILE = 'users_details.csv'
HISTORY_FILE = 'history_details.csv'

# ---------- Authentication System ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_DATA_FILE):
        return pd.read_csv(USER_DATA_FILE)
    return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    users = load_users()
    if username in users['username'].values:
        return False
    new_user = pd.DataFrame([[username, hash_password(password)]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DATA_FILE, index=False)
    return True

def authenticate(username, password):
    users = load_users()
    user = users[users['username'] == username]
    return not user.empty and user.iloc[0]['password'] == hash_password(password)

# ---------- Save Prediction and Generate PDF ----------
def save_prediction(username, data_dict, result):
    df = pd.DataFrame([{
        "username": username,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **data_dict,
        "result": "Diabetic" if result == 1 else "Not Diabetic"
    }])
    if os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)

def generate_pdf_report(user_data, prediction, username):
    filename = f"{username}_diabetes_report.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"Diabetes Prediction Report for {username}")
    c.drawString(50, 735, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 710, "Input Data:")
    y = 695
    for key, val in user_data.items():
        c.drawString(60, y, f"{key}: {val}")
        y -= 15
    c.drawString(50, y - 10, f"Prediction Result: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")
    c.save()
    return filename

# ---------- Login UI ----------
def login_ui():
    st.markdown(
        """
        <h1 class='custom-title'>Welcome to the Diabetes Predictor</h1>
        <h3 style='text-align:center; color:#555;'>Predict your health with confidence</h3>
        """,
        unsafe_allow_html=True
    )
    st.subheader("🔐 Login or Register to Continue")
    auth_choice = st.radio("Select Action", ["Login", "Register"], key="auth_choice")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if auth_choice == "Register":
        if st.button("Register"):
            if save_user(username, password):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists.")
    else:
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

# ---------- Main Logic ----------
if not st.session_state.logged_in:
    local_css_with_background("style.css", "background.jpg")
    login_ui()
else:
    local_css_with_background("style.css", "diabetes.jpg")

    st.markdown("""
        <div style='text-align: right;'>
            <form action="#" method="post">
                <button type="submit" name="logout" style='background:#4a47a3;color:white;padding:8px 16px;border:none;border-radius:8px;font-weight:bold;'>Logout</button>
            </form>
        </div>
    """, unsafe_allow_html=True)

    st.title("🩺 Diabetes Prediction App")
    st.markdown("<p style='font-weight:bold;'>This app predicts the likelihood of diabetes using Machine Learning.</p>", unsafe_allow_html=True)

    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    st.sidebar.title("📊 Patient Health Info")
    def user_input_features():
        return pd.DataFrame({
            "Pregnancies": [st.sidebar.slider("Pregnancies", 0, 17, 3)],
            "Glucose": [st.sidebar.slider("Glucose", 50, 200, 120)],
            "BloodPressure": [st.sidebar.slider("Blood Pressure", 50, 130, 70)],
            "SkinThickness": [st.sidebar.slider("Skin Thickness", 0, 100, 20)],
            "Insulin": [st.sidebar.slider("Insulin", 0, 900, 79)],
            "BMI": [st.sidebar.slider("BMI", 10.0, 67.0, 24.0)],
            "DiabetesPedigreeFunction": [st.sidebar.slider("DPF", 0.0, 2.4, 0.47)],
            "Age": [st.sidebar.slider("Age", 21, 88, 33)]
        })

    user_data = user_input_features()

    if st.sidebar.button("🔍 Predict"):
        st.session_state.predicted = True
        st.session_state.input_data = user_data.to_dict('records')[0]
        prediction = rf.predict(scaler.transform(user_data))[0]
        st.session_state.result = prediction
        save_prediction(st.session_state.username, st.session_state.input_data, prediction)
        st.session_state.pdf_file = generate_pdf_report(st.session_state.input_data, prediction, st.session_state.username)

    if st.session_state.predicted and st.session_state.result is not None:
        result = st.session_state.result
        st.subheader("📌 Patient Data")
        st.write(pd.DataFrame([st.session_state.input_data]))

        st.subheader("🔍 Prediction Result")
        if result == 0:
            st.markdown("<div class='healthy-msg'>✅ You are not diabetic</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='diabetic-msg'>⚠️ You are diabetic. Please consult a doctor.</div>", unsafe_allow_html=True)

        with open(st.session_state.pdf_file, "rb") as f:
            st.download_button("📥 Download PDF Report", f, file_name=st.session_state.pdf_file)

        st.markdown("---")
        st.subheader("📜 Prediction History")
        if os.path.exists(HISTORY_FILE):
            history_df = pd.read_csv(HISTORY_FILE)
            user_history = history_df[history_df["username"] == st.session_state.username]
            if not user_history.empty:
                user_history = user_history.sort_values("datetime", ascending=False).reset_index(drop=True)
                user_history.index += 1
                st.dataframe(user_history, use_container_width=True)
            else:
                st.info("No history found.")
        else:
            st.info("No history file found.")

        st.sidebar.subheader(f"✅ Accuracy: {accuracy_score(y_test, rf.predict(X_test)) * 100:.2f}%")

        def create_plot(x, y, title):
            fig = px.scatter(df, x=x, y=y, color=df["Outcome"].map({0: "Healthy", 1: "Diabetic"}),
                             title=title, labels={"Outcome": "Status"})
            fig.add_scatter(x=[st.session_state.input_data[x]], y=[st.session_state.input_data[y]],
                            mode='markers', marker=dict(size=12, color="red" if result == 1 else "green"), name='You')
            st.plotly_chart(fig)

        st.subheader("📊 Health Metrics Comparison")
        for metric in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]:
            create_plot("Age", metric, f"{metric} Comparison")

        if result == 1:
            st.subheader("🏥 Find Nearest Hospital")
            state = st.text_input("Enter State")
            city = st.text_input("Enter City")
            if st.button("Search Hospitals"):
                if state and city:
                    query = f"hospitals in {city}, {state}".replace(" ", "+")
                    st.markdown(f"<a href='https://www.google.com/search?q={query}' target='_blank'>🔍 Click here to view hospitals</a>", unsafe_allow_html=True)
                else:
                    st.warning("Please enter both state and city.")
