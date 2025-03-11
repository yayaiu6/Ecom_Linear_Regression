
# !pip install pandas matplotlib seaborn scikit-learn

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
df = pd.read_csv('Ecommerce Customers')

# Data visualization (relationship between time on app and yearly amount spent)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df)
plt.show()

# Splitting data into features (X) and target (y)
X = df[['Time on App', 'Time on Website', 'Length of Membership']]
y = df[['Yearly Amount Spent']]

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Linear Regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Display model coefficients
coef_df = pd.DataFrame(lm.coef_[0], X.columns, columns=['Coefficient'])
print(coef_df)

# Make predictions using the test set
predictions = lm.predict(X_test)

# Calculate errors to evaluate model performance
mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error
mse = mean_squared_error(y_test, predictions)    # Mean Squared Error
rmse = mse ** 0.5                                # Root Mean Squared Error

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Visualizing the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Ideal line
plt.xlabel('Actual Yearly Amount Spent')
plt.ylabel('Predicted Yearly Amount Spent')
plt.title('Actual vs Predicted')
plt.show()
