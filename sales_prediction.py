#  Iam Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('Advertising.csv')

# Then We Will Display the first few rows
print("Dataset Preview:")
print(df.head())

#  Now It  will Show Dataset summary
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

#  IT Will Visualize relationships between variables
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')
plt.suptitle("Advertising Spend vs Sales", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Prepare features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Splitinng  the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# NOW we Will Make predictions on the test set
y_pred = model.predict(X_test)

#  This Will Evaluate the model
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared Score (R²):", r2_score(y_test, y_pred))

# Optional:if we want to Predict on new input we can do it
new_data = pd.DataFrame({'TV': [150], 'Radio': [30], 'Newspaper': [20]})
predicted_sales = model.predict(new_data)
print("\nPredicted Sales for new data:", predicted_sales[0])
