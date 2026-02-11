import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Insurance Pricing Demo", page_icon="💼")

st.title("💼 Insurance Pricing Demo")
st.write("Enter customer details below to generate a price prediction.")

# --- Input fields ---
age = st.slider("Age", 18, 80, 40)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 
                14.0, 
                45.0, 
                28.5,
                help="Typical BMI range in the training data is 16–45. Values below 16 may lead to less accurate predictions.")
if bmi < 16:
    st.warning("Predictions may be less accurate for BMI values below 16, as the training data contains very few samples in this range.")
elif bmi < 18.5:
    st.info(" Predictions may be slightly less stable for BMI values below 18.5.")

children = st.slider("Number of Children", 0, 5, 2)
smoker = st.selectbox("Smoker", ["no", "yes"])

if st.button("Predict Price"):
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()

            st.subheader("📊 Prediction Results")
            st.metric("Final Price", f"${data['final_price']:.2f}")
            st.write(f"**Confidence Interval:** ${data['confidence_interval'][0]:.2f} — ${data['confidence_interval'][1]:.2f}")

            st.write("---")
            st.subheader("🔍 Model Breakdown")
            st.write(f"**XGBoost Price:** ${data['xgb_price']:.2f}")
            st.write(f"**Frontier Price:** ${data['frontier_price']:.2f}")

            st.write("---")
            st.subheader("🧠 Explanation")
            st.write(data["frontier_explanation"])
            st.caption(data["ensemble_explanation"])

        else:
            st.error(f"API Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"Request failed: {e}")
