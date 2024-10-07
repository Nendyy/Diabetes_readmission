import joblib
import streamlit as st
import pandas as pd
import json

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

model_path = config['model_path']
encoder_path = config['encoder_path']

@st.cache_resource
def load_model():
    model = joblib.load(model_path)  
    return model

@st.cache_resource
def load_encoder():
    encoder = joblib.load(encoder_path)  
    return encoder

def get_number_input(label, default=1, min_value=0):
    value = st.number_input(label, value=default, min_value=min_value)
    if value < min_value:
        st.error(f"{label} cannot be negative.")
    return value

def preprocess_input(input_df, encoder):
    # Encode gender
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    
    # Encode hba1c
    input_df['hba1c'] = input_df['hba1c'].map({'>7': 1, '>8': 1, 'Normal': 0, 'Unknown': -99})
    
    # Encode max_glu_serum
    input_df['max_glu_serum'] = input_df['max_glu_serum'].map({'>200': 1, '>300': 1, 'Normal': 0, 'Unknown': -99})

    # Function to map Yes/No to 1/0 for specified columns
    def encode_binary_columns(input_df, columns):
        mapping = {'Yes': 1, 'No': 0}
        for column in columns:
            input_df[column] = input_df[column].map(mapping)
        return input_df

    # List of columns to encode
    columns_to_encode = ['diabetesMed', 'change', 'metformin', 'glipizide', 
                         'glyburide', 'pioglitazone', 'rosiglitazone']

    # Apply the encoding function
    input_df = encode_binary_columns(input_df, columns_to_encode)

    # Define the categorical columns for encoding
    cat_columns = ['race', 'age', 'admission_type', 'discharge_disposition',
                   'admission_source', 'primary_diagnosis', 'insulin']
    
    # Apply the encoder on categorical columns
    input_encoded = encoder.transform(input_df[cat_columns])

    # Create a DataFrame from the encoded new data with proper column names
    encoded_new_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(cat_columns))
    
    # Merge encoded categorical columns with non-categorical ones
    non_cat_columns = input_df.drop(columns=cat_columns)
    final_input = pd.concat([non_cat_columns.reset_index(drop=True), encoded_new_df.reset_index(drop=True)], axis=1)
    
    # Display the final input data in the Streamlit app
    st.write("Final input data:")
    st.dataframe(final_input)

    return final_input

def main():
    st.title('Diabetes Readmission Prediction')

    model = load_model()
    encoder = load_encoder()

    with st.form(key='my_form'):
        st.write("""
            ### Input Instructions
            Please fill in the patient details to predict the likelihood of readmission.
            """)
        
        # Categorical Inputs
        gender = st.selectbox('Gender', ('Male', 'Female'))
        age = st.selectbox('Age Group', ('[0-40]', '[40-50]', '[50-60]', '[60-70]', '[70-80]', '[80-100]'))
        race = st.selectbox('Race', ('Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'))
        admission_type = st.selectbox('Admission Type', ('Emergency', 'Elective', 'Other'))
        admission_source = st.selectbox('Admission Source', ('Emergency room', 'Physician referral', 'Other'))
        discharge_disposition = st.selectbox('Discharge Disposition', ('Home', 'Transfer to another facility', 'Other'))
        primary_diagnosis = st.selectbox('Primary Diagnosis', ('Circulatory', 'Diabetes', 'Respiratory', 'Digestive',
                                        'Genitourinary', 'Neoplasms', 'Musculoskeletal', 'Injury', 'Other'))
        insulin = st.selectbox('Insulin', ('No', 'Steady', 'Up', 'Down'))

        # Numerical Inputs using the new function
        time_in_hospital = get_number_input('Time in Hospital (days)')
        num_lab_procedures = get_number_input('Number of Lab Procedures')
        num_procedures = get_number_input('Number of Procedures')
        num_medications = get_number_input('Number of Medications')
        number_inpatient = get_number_input('Number of Inpatient Visits')
        number_outpatient = get_number_input('Number of Outpatient Visits')
        number_emergency = get_number_input('Number of Emergency Visits')
        number_diagnoses = get_number_input('Number of Diagnoses')

        # Medications
        diabetesMed = st.selectbox('Diabetes Medications', ('Yes', 'No'))
        metformin = st.selectbox('Metformin', ('Yes', 'No'))
        glipizide = st.selectbox('Glipizide', ('Yes', 'No'))
        glyburide = st.selectbox('Glyburide', ('Yes', 'No'))
        pioglitazone = st.selectbox('Pioglitazone', ('Yes', 'No'))
        rosiglitazone = st.selectbox('Rosiglitazone', ('Yes', 'No'))
        change = st.selectbox('Medication Change', ('Yes', 'No'))

        # HbA1c and Max Glucose Serum
        hba1c = st.selectbox('HbA1c Result', ('>7', '>8', 'Normal', 'Unknown'))
        max_glu_serum = st.selectbox('Max Glucose Serum Result', ('>200', '>300', 'Normal', 'Unknown'))

        # Form submission button
        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Collect input data in a DataFrame
        input_data = pd.DataFrame({
            'race': [race],
            'gender': [gender],
            'age': [age],
            'admission_type': [admission_type],
            'discharge_disposition': [discharge_disposition],
            'admission_source': [admission_source],
            'time_in_hospital': [time_in_hospital],
            'num_lab_procedures': [num_lab_procedures],
            'num_procedures': [num_procedures],
            'num_medications': [num_medications],
            'number_inpatient': [number_inpatient],
            'number_outpatient': [number_outpatient],
            'number_emergency': [number_emergency],
            'primary_diagnosis': [primary_diagnosis],
            'insulin': [insulin],
            'number_diagnoses': [number_diagnoses],
            'diabetesMed': [diabetesMed],
            'metformin': [metformin],
            'glipizide': [glipizide],
            'glyburide': [glyburide],
            'pioglitazone': [pioglitazone],
            'rosiglitazone': [rosiglitazone],
            'change': [change],
            'hba1c': [hba1c],
            'max_glu_serum': [max_glu_serum]
        })

        # Preprocess user input
        input_data_preprocessed = preprocess_input(input_data, encoder)

        # Predict readmission
        prediction_proba = model.predict_proba(input_data_preprocessed.values)[:, 1]
        output = round(prediction_proba[0] * 100, 2)

        # Display prediction result
        st.success(f"Patient has {output} % chance of being readmitted.")

if __name__ == '__main__':
    main()
