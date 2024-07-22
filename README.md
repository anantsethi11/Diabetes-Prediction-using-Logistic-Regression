# Diabetes-Prediction-using-Logistic-Regression
Overview
This Python script demonstrates the use of logistic regression for predicting diabetes outcomes based on various health metrics. The dataset used (diabetes.csv) contains information about patients, including their glucose levels, blood pressure, BMI, etc., and whether they have diabetes or not (binary outcome).
Steps Performed
1.	Data Loading: The dataset is loaded from the local file (diabetes.csv) using Pandas.
2.	Data Preprocessing: 
o	Separate the features (X) and the target variable (y).
o	Split the dataset into training and testing sets using train_test_split from sklearn.model_selection.
o	Perform feature scaling using Standard Scaler from sklearn.preprocessing to standardize the features.
3.	Model Training:
o	Train a logistic regression model (Logistic Regression from sklearn.linear_model) on the training data (X_train, y_train).
4.	Model Evaluation:
o	Predict outcomes on the test data (X_test) and evaluate the model's accuracy using accuracy_score from sklearn.metrics.
o	Visualize performance using a confusion matrix (confusion_matrix from sklearn.metrics) to show predicted versus actual outcomes.
5.	ROC Curve Analysis:
o	Compute ROC curve and calculate AUC (Area Under Curve) using roc_curve and auc from sklearn.metrics.
o	Visualize the ROC curve to assess the model's discrimination ability between diabetes and non-diabetes cases.

Conclusion
The logistic regression model achieved an accuracy of Model Accuracy 75% on the test set. The confusion matrix heatmap visually represents the model's performance in predicting diabetes outcomes. Additionally, the ROC curve illustrates the trade-off between sensitivity (true positive rate) and specificity (true negative rate) across different threshold settings.
This script serves as a practical example of using logistic regression for binary classification tasks and demonstrates essential steps in model evaluation and performance visualization.
