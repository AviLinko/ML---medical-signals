import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv("insurance.csv")

# One-hot encode the categorical variables
encoder = OneHotEncoder(sparse=False)
encoded_sex = encoder.transform(data[["sex"]])
encoded_sex_df = pd.DataFrame(encoded_sex, columns=encoder.get_feature_names_out())
data = pd.concat([data, encoded_sex_df], axis=1)
encoded_sex = encoder.fit_transform(data[["sex"]])
encoded_smoker = encoder.fit_transform(data[["smoker"]])
encoded_region = encoder.fit_transform(data[["region"]])

# Concatenate the encoded variables with the numerical variables
data = pd.concat([data, pd.DataFrame(encoded_sex, columns=encoder.get_feature_names(["sex"]))], axis=1)
data = pd.concat([data, pd.DataFrame(encoded_smoker, columns=encoder.get_feature_names(["smoker"]))], axis=1)
data = pd.concat([data, pd.DataFrame(encoded_region, columns=encoder.get_feature_names(["region"]))], axis=1)

# Drop the original categorical variables
data = data.drop(["sex", "smoker", "region"], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("charges", axis=1), data["charges"], test_size=0.2)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the R^2 score for the model on the test set
print("R^2:", model.score(X_test, y_test))