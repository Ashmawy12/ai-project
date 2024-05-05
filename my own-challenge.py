import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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
    train_csv_file_path = r"C:\Users\osama_g47d7h5\Downloads\ARC-Challenge-Train.csv"
    dev_csv_file_path = r"C:\Users\osama_g47d7h5\Downloads\ARC-Challenge-Dev.csv"
    test_csv_file_path = r"C:\Users\osama_g47d7h5\Downloads\ARC-Challenge-Test.csv"

    # Call the function to read the CSV files
    train_data = read_csv_file(train_csv_file_path)
    dev_data = read_csv_file(dev_csv_file_path)
    test_data = read_csv_file(test_csv_file_path)

    if train_data is not None and dev_data is not None and test_data is not None:
        print("Datasets loaded successfully:")
        print("Train dataset:")
        print(train_data.head())  # Display the first few rows of the train dataset
        print("\nDev dataset:")
        print(dev_data.head())  # Display the first few rows of the dev dataset
        print("\nTest dataset:")
        print(test_data.head())  # Display the first few rows of the test dataset

        # Concatenate train, dev, and test datasets
        data_frame = pd.concat([train_data, dev_data, test_data], ignore_index=True)

        # Choosing target and features
        target_column = 'totalPossiblePoint'
        feature_columns = ['isMultipleChoiceQuestion', 'includesDiagram', 'schoolGrade', 'year',
                           'subject', 'category']  # Updated feature columns

        # Splitting into features and target
        X = data_frame[feature_columns]
        y = data_frame[target_column]

        # Preprocessing pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        categorical_features = X.select_dtypes(include=['object']).columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Creating a pipeline with preprocessing and Ridge regression model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0))  # You can adjust the alpha parameter for the level of regularization
        ])

        # Splitting the dataset into the training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the pipeline
        pipeline.fit(X_train, y_train)

        # Making predictions
        y_pred = pipeline.predict(X_test)

        # Evaluating the model
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)
