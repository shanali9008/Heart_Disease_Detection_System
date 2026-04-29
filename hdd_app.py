# Import required libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to train the selected model
def train_model(data, model_type):
    # Ensure the model_type is valid
    valid_models = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    if model_type not in valid_models:
        raise ValueError(f"Unsupported model type: {model_type}. Please select a valid model.")
    
    # Splitting the dataset into training and testing sets (70% train, 30% test)
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Select model based on user input
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, accuracy, precision, recall, f1, model_type  # Return model_type as well

# Set custom background image or color
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://t4.ftcdn.net/jpg/01/76/67/25/360_F_176672598_cJ4yPCFhxvDXm9Cu7vDLIcXpvTMQJ9zm.jpg');
            background-size: cover;
            background-position:center;
            background-repeat: no-repeat:
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the background function
set_background()

# Streamlit application
st.title("Heart Disease Detection App")

# Sidebar for user options
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose an action", ["Load Existing Model", "Train a New Model"])

if option == "Train a New Model":
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        # Load the dataset
        data = pd.read_csv(uploaded_file)
        
        # Ensure 'target' column exists
        if "target" not in data.columns:
            st.error("The dataset must contain a 'target' column with 0 for No Disease and 1 for Disease.")
        else:
            st.write("Dataset Preview:")
            st.write(data.head())
            
            # Model selection
            model_type = st.selectbox("Select Model to Train", ["Random Forest", "Logistic Regression", "Gradient Boosting"])
            st.write(f"Selected model: {model_type}")  # Debugging the selected model type
            
            if st.button("Train Model"):
                try:
                    model, accuracy, precision, recall, f1, model_type = train_model(data, model_type)
                    st.write(f"Model trained successfully!")
                    st.write(f"Accuracy: {accuracy*100:.2f}%")
                    st.write(f"Precision: {precision*100:.2f}%")
                    st.write(f"Recall: {recall*100:.2f}%")
                    st.write(f"F1 Score: {f1*100:.2f}%")
                    
                    # Save the trained model and its name
                    with open('trained_model.pkl', 'wb') as file:
                        pickle.dump((model, model_type), file)  # Save both model and model_type
                    st.write(f"{model_type} model saved as 'trained_model.pkl'")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
else:
    st.subheader("Load an Existing Model")
    try:
        with open('trained_model.pkl', 'rb') as file:
            model, model_type = pickle.load(file)  # Load both model and model_type
        st.write(f"{model_type} loaded successfully!")  # Modified message
    except FileNotFoundError:
        st.warning("No saved model found. Train a new model or ensure 'trained_model.pkl' exists in the application directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")

# Input parameters with classy descriptions
st.sidebar.header("User Input Parameters")
def user_input_features():
    st.sidebar.markdown("### Personal Information")
    age = st.sidebar.number_input("Age", 29, 77, 50)
    st.sidebar.markdown("**Sex**  \n1 = Male, 0 = Female")
    sex = st.sidebar.selectbox("Select Sex", [0, 1])

    st.sidebar.markdown("### Medical Information")
    st.sidebar.markdown("**Chest Pain Type**  \n0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic")
    cp = st.sidebar.selectbox("Select Chest Pain Type", [0, 1, 2, 3])

    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 250)

    st.sidebar.markdown("**Fasting Blood Sugar > 120 mg/dl**  \n0 = False, 1 = True")
    fbs = st.sidebar.selectbox("Select Fasting Blood Sugar", [0, 1])

    st.sidebar.markdown("**Resting ECG Results**  \n0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy")
    restecg = st.sidebar.selectbox("Select Resting ECG Results", [0, 1, 2])

    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)

    st.sidebar.markdown("**Exercise Induced Angina**  \n0 = No, 1 = Yes")
    exang = st.sidebar.selectbox("Select Exercise Induced Angina", [0, 1])

    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise (Oldpeak)", 0.0, 6.2, 1.0)

    st.sidebar.markdown("**Slope of the Peak Exercise ST Segment**  \n0 = Upsloping, 1 = Flat, 2 = Downsloping")
    slope = st.sidebar.selectbox("Select Slope", [0, 1, 2])

    st.sidebar.markdown("**Number of Major Vessels Colored by Fluoroscopy**  \n0 = None, 1 = One, 2 = Two, 3 = Three")
    ca = st.sidebar.selectbox("Select Number of Major Vessels", [0, 1, 2, 3])

    st.sidebar.markdown("**Thalassemia**  \n0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect")
    thal = st.sidebar.selectbox("Select Thalassemia", [0, 1, 2])

    st.sidebar.markdown("**Smoking**  \n0 = No, 1 = Yes")
    Smoking = st.sidebar.selectbox("Select Smoking", [0, 1])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'Smoking': Smoking  # Case-sensitive match
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Loading saved column order (update this part to match your feature order from training)
expected_columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'Smoking'
]

# Reorder input_df to match the order used during training
input_df = input_df[expected_columns]

st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction")
        st.write("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")

        st.subheader("Prediction Probability")
        st.write(f"Probability of Heart Disease: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]*100:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")