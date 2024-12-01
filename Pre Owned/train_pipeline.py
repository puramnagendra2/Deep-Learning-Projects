import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from scikeras.wrappers import KerasRegressor
import joblib

# Importing Dataset
df = pd.read_csv('vehiclesFinal.csv')

df = df.drop(['id', 'manufacturer', 'model', 'region', 'lat', 'long', 'paint_color'], axis=1)

# Removing Cylinders text from column

df['cylinders'] = df['cylinders'].str.extract('(\d+)')
df['cylinders'] = df['cylinders'].ffill()
df['cylinders'] = df['cylinders'].astype(int)


df['year'] = df['year'].astype(int)
df['odometer'] = df['odometer'].astype(int)

# Extracting Numerical Features and Categorical Features
num_features = df.select_dtypes('number').columns
cat_features = df.select_dtypes(exclude='number').columns

# Handling Duplicates
df_duplicates = df.duplicated()
duplicate_rows = df[df.duplicated()]
all_duplicates = df[df.duplicated(keep=False)]
df_no_duplicates = df.drop_duplicates()


# Handling Outliers
for col in num_features:
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    threshold = 3
    df = df[z_scores < threshold]



# Split features and target
X = df.drop('price', axis=1)
y = df['price']


# Define the Keras model
def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# Extracting Features
numerical_features = X.select_dtypes('number').columns
categorical_features = X.select_dtypes(exclude='number').columns

# Creating Preprocessing Pipeline
preprocessing_pipeline = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('cat', Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('encoding', OneHotEncoder())
    ]), categorical_features)
])


# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessing_pipeline', preprocessing_pipeline),
    ('model', KerasRegressor(model=create_model(43), epochs=20, batch_size=32, verbose=0))
])

# Train pipeline
pipeline.fit(X_train, y_train)


# Save the trained pipeline
joblib.dump(pipeline, 'price_prediction_pipeline.pkl')
print("Pipeline saved!")