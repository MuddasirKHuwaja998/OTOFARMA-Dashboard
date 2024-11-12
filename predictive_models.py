import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

def predictive_models_page(df, target_columns):
    # Check if target columns exist in the dataframe
    missing_columns = [col for col in target_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing target columns: {', '.join(missing_columns)}")

    # Split the dataset into left ear and right ear feature sets
    left_ear_features = [col for col in df.columns if 'sx' in col]
    right_ear_features = [col for col in df.columns if 'dx' in col]

    # Handle missing values in the dataset
    st.write("Handling missing values...")

    # Fill missing values in the features with the median (for numerical data)
    imputer = SimpleImputer(strategy='median')
    left_ear_data = pd.DataFrame(imputer.fit_transform(df[left_ear_features]), columns=left_ear_features)
    right_ear_data = pd.DataFrame(imputer.fit_transform(df[right_ear_features]), columns=right_ear_features)

    # Fill missing values in the target columns with the mode (most frequent value)
    for target_column in target_columns:
        left_ear_data[target_column] = df[target_column].fillna(df[target_column].mode()[0])
        right_ear_data[target_column] = df[target_column].fillna(df[target_column].mode()[0])

    # Perform prediction for each target column for both left and right ear data
    st.subheader("Predictive Modeling for Left Ear Data")

    for target_column in target_columns:
        st.write(f"Training model for {target_column} (Left Ear)...")

        # Prepare features and target for left ear
        X_left = left_ear_data.drop(columns=target_columns)
        y_left = left_ear_data[target_column]

        # Split data into training and test sets
        X_train_left, X_test_left, y_train_left, y_test_left = train_test_split(X_left, y_left, test_size=0.3, random_state=42)

        # Initialize and train the model for left ear
        model_left = RandomForestClassifier(n_estimators=100, random_state=42)
        model_left.fit(X_train_left, y_train_left)

        # Make predictions
        y_pred_left = model_left.predict(X_test_left)

        # Evaluate the model
        accuracy_left = accuracy_score(y_test_left, y_pred_left)
        classification_rep_left = classification_report(y_test_left, y_pred_left)

        # Display results for left ear
        st.write(f"Model for {target_column} (Left Ear) - Accuracy: {accuracy_left:.2f}")
        st.text(f"Classification Report for {target_column} (Left Ear):\n{classification_rep_left}")

        # Show the true vs predicted values
        predictions_df_left = pd.DataFrame({
            'True Values': y_test_left,
            'Predicted Values': y_pred_left
        })

        st.subheader(f"True vs Predicted values for {target_column} (Left Ear)")
        st.write(predictions_df_left)

    st.subheader("Predictive Modeling for Right Ear Data")

    for target_column in target_columns:
        st.write(f"Training model for {target_column} (Right Ear)...")

        # Prepare features and target for right ear
        X_right = right_ear_data.drop(columns=target_columns)
        y_right = right_ear_data[target_column]

        # Split data into training and test sets
        X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(X_right, y_right, test_size=0.3, random_state=42)

        # Initialize and train the model for right ear
        model_right = RandomForestClassifier(n_estimators=100, random_state=42)
        model_right.fit(X_train_right, y_train_right)

        # Make predictions
        y_pred_right = model_right.predict(X_test_right)

        # Evaluate the model
        accuracy_right = accuracy_score(y_test_right, y_pred_right)
        classification_rep_right = classification_report(y_test_right, y_pred_right)

        # Display results for right ear
        st.write(f"Model for {target_column} (Right Ear) - Accuracy: {accuracy_right:.2f}")
        st.text(f"Classification Report for {target_column} (Right Ear):\n{classification_rep_right}")

        # Show the true vs predicted values
        predictions_df_right = pd.DataFrame({
            'True Values': y_test_right,
            'Predicted Values': y_pred_right
        })

        st.subheader(f"True vs Predicted values for {target_column} (Right Ear)")
        st.write(predictions_df_right)
