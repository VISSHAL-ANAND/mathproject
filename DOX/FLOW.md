# Data and Prediction Flow
1. App creates synthetic demo data or reads an uploaded CSV.
2. Validate target and supported predictors.
3. Split rows into training and test subsets.
4. Fit imputation, encoding, scaling, and model using training data only.
5. Predict on the held-out test set and calculate MAE, RMSE, R-squared.
6. Display comparison and diagnostic plots.
7. User enters feature values and chooses a fitted model.
8. Show an estimated response time in minutes and a reliability warning.
