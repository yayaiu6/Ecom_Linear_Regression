# Ecom_Linear_Regression

## Project Overview
This project is a machine learning model that predicts the **Yearly Amount Spent** by e-commerce customers based on their online behavior. By analyzing features such as **Time on App**, **Time on Website**, and **Length of Membership**, we build a linear regression model to gain insights into customer spending habits.



## Project Objectives
- **Predict** yearly customer spending using a linear regression model.
- **Visualize** the relationship between different features and spending.
- **Evaluate** the model's performance using error metrics.



## Dataset
The dataset used in this project (`Ecommerce Customers.csv`) contains the following features:
- **Email**: Customer's email address.
- **Address**: Customer's physical address.
- **Avatar**: Customer's avatar or profile image.
- **Time on App**: Minutes spent on the mobile application.
- **Time on Website**: Minutes spent on the website.
- **Length of Membership**: Duration of the customer's membership in years.
- **Yearly Amount Spent**: The target variable representing the yearly spending in USD.



## Project Steps
1. **Data Loading and Visualization**:
   - The dataset is loaded using `pandas`.
   - We use `seaborn` to create joint plots to visualize the relationship between `Time on App` and `Yearly Amount Spent`.

2. **Data Preparation**:
   - Selecting relevant features (`Time on App`, `Time on Website`, `Length of Membership`) and the target variable (`Yearly Amount Spent`).
   - Splitting the data into training (70%) and testing (30%) sets using `train_test_split` from `sklearn`.

3. **Model Training**:
   - Using `LinearRegression` from `sklearn.linear_model` to train the model.
   - Extracting the coefficients of the linear model to understand the impact of each feature.

4. **Making Predictions**:
   - Predicting `Yearly Amount Spent` using the test set.
   - Comparing predicted values with actual values.

5. Model Evaluation :
   - Calculating Mean Absolute Error (MAE) , Mean Squared Error (MSE) , and  Root Mean Squared Error (RMSE) to measure model performance.
   - Plotting actual vs. predicted values to visualize the model's accuracy.


