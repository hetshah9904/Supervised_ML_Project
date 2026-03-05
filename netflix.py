import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set Page Config for a professional Netflix-style look
st.set_page_config(page_title="Netflix Rating Predictor", page_icon="🍿", layout="centered")

# Custom CSS to make it look nice
st.markdown("<h1>Header</h1>", unsafe_allow_html=True)

# 1. Load the Model
# Important: Ensure netflix_model.pkl was trained with exactly 6 features
try:
    with open('netflix_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🍿 Netflix AI: Content Success Predictor")
st.write("Predict the IMDB score of your favorite titles using Random Forest Regression.")

# 2. User Inputs
user_title = st.text_input("Project / Movie Title", placeholder="e.g. Stranger Things")

col1, col2 = st.columns(2)

with col1:
    content_type = st.selectbox("Content Type", ["MOVIE", "SHOW"])
    release_year = st.number_input("Release Year", min_value=1950, max_value=2026, value=2024)

with col2:
    runtime = st.slider("Runtime (in minutes)", 10, 250, 90)
    expected_votes = st.number_input("Expected Popularity (Votes)", min_value=0, value=5000)

# 3. Feature Engineering (The 6 Clues)
# Feature 1: Type Encoded
type_val = 0 if content_type == "MOVIE" else 1

# Feature 2: Runtime (Direct)
# Feature 3: Log of Votes (To normalize data)
votes_log = np.log1p(expected_votes)

# Feature 4: Age of Content
content_age = 2024 - release_year

# Feature 5: Era (Binned Year)
if release_year <= 1980: era = 1
elif release_year <= 2000: era = 2
elif release_year <= 2010: era = 3
elif release_year <= 2020: era = 4
else: era = 5

# Feature 6: Title Length
title_len = len(user_title)

# 4. Prediction Logic
if st.button("Generate Prediction Report"):
    if not user_title:
        st.warning("Please provide a title to generate the report.")
    else:
        # Prepare the array with EXACTLY 6 features in the correct order
        # Order: [type, runtime, votes_log, age, era, title_len]
        input_data = np.array([[type_val, runtime, votes_log, content_age, era, title_len]])
        
        # Make the prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # --- SHOWCASE THE RESULT ---
            st.divider()
            st.subheader(f"Analysis for: :red[{user_title.upper()}]")
            
            # Display metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Score", f"{prediction:.1f}/10")
            m2.metric("Release Era", f"Era {era}")
            m3.metric("Title Length", f"{title_len} Chars")
            
            # Progress bar for visual score representation
            st.write("**Rating Confidence:**")
            st.progress(min(max(prediction/10, 0.0), 1.0))
            
            # Detailed Conclusion
            if prediction >= 7.5:
                st.success("🎯 **Conclusion:** High potential for critical acclaim and 'Top 10' trending status.")
            elif prediction >= 6.0:
                st.info("📈 **Conclusion:** Likely to be a solid performer with balanced audience reviews.")
            else:
                st.error("📉 **Conclusion:** May struggle with ratings; consider improving production value or runtime.")
                
        except ValueError as ve:
            st.error(f"Feature Mismatch Error: {ve}")
            st.info("Tip: If the model expects more/less features, ensure your 'input_data' array")