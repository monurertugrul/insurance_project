import streamlit as st
from core.predictor import InsurancePredictor

# This ensures the heavy load only happens once, and NOT during startup
@st.cache_resource
def get_predictor():
    return InsurancePredictor()

st.set_page_config(page_title="Medical Insurance Pricing Demo", page_icon="💼")
st.title("💼 Medical Insurance Pricing Demo")

# --- Input fields ---
age = st.slider("Age", 18, 80, 40)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 14.0, 45.0, 28.5)
children = st.slider("Number of Children", 0, 5, 2)
smoker = st.selectbox("Smoker", ["no", "yes"])

# --- Prediction button ---
if st.button("Predict Price", type="primary"):
    # The app is already "live" and working at this point
    with st.spinner("Analyzing cases and generating prediction..."):
        try:
            predictor = get_predictor()
            result = predictor.predict(age, sex, bmi, children, smoker)

            st.subheader("📊 Prediction Results")
            st.metric("Final Price", f"${result['final_price']:.2f}")
            
            col1, col2 = st.columns(2)
            col1.write(f"**XGBoost:** ${result['xgb_price']:.2f}")
            col2.write(f"**Gemini:** ${result['frontier_price']:.2f}")

            st.write("---")
            st.subheader("🧠 Explanation")
            
            # Escape dollar signs to prevent LaTeX "squashing"
            explanation = result["frontier_explanation"].replace("$", "\\$")
            st.markdown(explanation) 
            
            st.caption(result["ensemble_explanation"])

        except Exception as e:
            st.error(f"Prediction failed: {e}")