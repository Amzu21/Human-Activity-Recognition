#GUI

import joblib
import streamlit as st

joblib.dump(svm, 'svm_model.pkl')
joblib.dump(sc, 'scaler.pkl')

model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

class_labels = ['LAYING' 'SITTING' 'STANDING' 'WALKING' 'WALKING_DOWNSTAIRS'
 'WALKING_UPSTAIRS']

st.title("Activity Prediction")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read and validate the input
        df = pd.read_csv(uploaded_file, header=None)
        st.write("Uploaded Data:", df)

        if df.shape[0] != 1:
            st.warning("Please upload a file with only **one row** of feature data.")
        else:
            input_data = df.values
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            activity = class_labels[int(prediction[0])]

            st.success(f"Predicted Activity: **{activity}**")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
