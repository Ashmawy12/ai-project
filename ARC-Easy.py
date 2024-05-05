import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Read the dataset from CSV
def read_csv_file(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    # Provide the file paths of the CSV files
    train_csv_file_path = r"C:\Users\osama_g47d7h5\Downloads\ARC-Easy-Train.csv"
    dev_csv_file_path = r"C:\Users\osama_g47d7h5\Downloads\ARC-Easy-Dev.csv"
    test_csv_file_path = r"C:\Users\osama_g47d7h5\Downloads\ARC-Easy-Test.csv"

    # Call the function to read the CSV files
    train_data_frame = read_csv_file(train_csv_file_path)
    dev_data_frame = read_csv_file(dev_csv_file_path)
    test_data_frame = read_csv_file(test_csv_file_path)

    if train_data_frame is not None and dev_data_frame is not None and test_data_frame is not None:
        print("Training Dataset loaded successfully:")
        print(train_data_frame.head())  # Display the first few rows of the training DataFrame

        print("\nDevelopment Dataset loaded successfully:")
        print(dev_data_frame.head())  # Display the first few rows of the development DataFrame

        print("\nTest Dataset loaded successfully:")
        print(test_data_frame.head())  # Display the first few rows of the test DataFrame

        # Concatenate train and dev dataframes
        train_data_frame = pd.concat([train_data_frame, dev_data_frame])

        # Choosing target and features
        target_column = 'totalPossiblePoint'
        feature_columns = ['questionID', 'originalQuestionID', 'isMultipleChoiceQuestion', 'includesDiagram',
                           'schoolGrade', 'year', 'subject', 'category']

        # Convert numerical columns to numeric data type for training data
        numeric_columns = ['originalQuestionID', 'isMultipleChoiceQuestion', 'includesDiagram', 'schoolGrade', 'year']
        for col in numeric_columns:
            train_data_frame[col] = pd.to_numeric(train_data_frame[col], errors='coerce')

        # Splitting training data into features and target
        X_train = train_data_frame[feature_columns]
        y_train = train_data_frame[target_column]

        # Convert numerical columns to numeric data type for test data
        for col in numeric_columns:
            test_data_frame[col] = pd.to_numeric(test_data_frame[col], errors='coerce')

        # Splitting test data into features and target
        X_test = test_data_frame[feature_columns]
        y_test = test_data_frame[target_column]

        # Preprocessing pipeline
        numeric_features = ['originalQuestionID', 'isMultipleChoiceQuestion', 'includesDiagram', 'schoolGrade', 'year']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        categorical_features = ['subject', 'category']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Creating a pipeline with preprocessing and linear regression model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # Training the pipeline
        pipeline.fit(X_train, y_train)

        # Making predictions on the test data
        y_pred = pipeline.predict(X_test)

        # Evaluating the model
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)
