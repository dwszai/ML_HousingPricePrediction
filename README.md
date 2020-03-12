# Machine Learning Pipeline
- Prepare data
- Select algorithm(s)
- Train model
- Package model
- Validate model
- Deploy model
- Monitor model

## Choice of model(s):
* Ridge
* Lasso
* ElasticNet
* Support Vector Regressor (SVR)
* LightGBM
* XGBoost
* Stack Regressor
* Blended Models (Combination of all models)

### Evaluation of the model(s):
The results of the trained model produced quite a close prediction to the actual train target variable. It has an accuracy score based on performance metrics RMSLE = 0.0292 and RMSE = 0.155. However, the best prediction score actually does not comes from our blended models. Instead, it comes from other algorithms trained models. We aim to achieve a test score that is close or even better than the train set score. If the difference is too far off, there could be overfitting in our train set which render the model to be less reliable for other datasets or future deployment. The following points will explain the differences of each trained models and their final prediction score (rounded).

##### 1. Ridge
A regularization model to prevent model from overfitting. It reduces large coefficients but prevent it from reaching zero.
- RidgeCV regression with built-in cross-validation
- Robust scaler by scaling features using statistics that are robust to outliers
- Decrease model complexity while keeping all features in the model
- Penalizes the sum of squared coefficients
- Choice of Regularization Parameter by allocating list of alphas value to find the optimal value

Final prediction score:
- RMSLE: 0.781
- RMSE: 17094.348

##### 2. Lasso
A regularization model to prevent model from overfitting. It reduces large coefficients and can bring it to 0.
- It can be use for feature selection
- Penalizes the sum of their absolute values
- Correlated features have a larger coefficient, while others are close to zero or zero
- Choice of Regularization Parameter by allocating list of alphas value to find the optimal value

Final prediction score:
- RMSLE: 0.782
- RMSE: 17204.302

##### 3. ElasticNet
It is a combination of both Ridge and Lasso regularization models.
- Mix of both sum of square and absolute values
- Tunable on the ratio of penalty
- Choice of Regularization Parameter and ratio values by allocating list to find the optimal value

Final prediction score:
- RMSLE: 0.782
- RMSE: 17200.410

##### 4. SVR
SVR is different than other regression models. It uses the Support Vector Machine(SVM, a classification algorithm) to predict a continuous variable. 
- Fit the best line within a predefined or threshold error value
- Hard to scale to datasets with more than a couple of 10,000 samples

Final prediction score:
- RMSLE: 0.687
- RMSE: 6839.904

##### 5. Light GBM
The size of dataset ranges from small to extremely large (big data) and is becoming available on many platforms. It becomes difficult for traditional data science algorithms to give faster results. Light GBM is prefixed as ‘Light’ because of its high speed and efficiency.
* Faster training speed and higher efficiency
* Handle large size of data using lower memory to run
* Focuses on accuracy of results
* Supports parallel and GPU learning, good for data science application development
* Performed accurately on the model resulting in overall high accuracy in predicted target

Final prediction score:
- RMSLE: 0.140
- RMSE: 563.641

##### 6. XGBoost
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
- Provides a parallel tree boosting
- Runs on major distributed environment 
- One of the best algorithm for Execution Speed and Model Performance
Final prediction score:
- RMSLE: 0.028
- RMSE: 134.136

##### 7. Stack Regressor
Stacking is an ensemble method to combine multiple regression models via a meta-regressor. StackingCVRegressor extends the standard stacking algorithm to a level-2 regressor.
* Combines all models listed above to get the best results among them
* Best accuracy score among all other models, including the blended model

Final prediction score:
- RMSLE: 0.020
- RMSE: 89.751

##### 8. Blended Models
Blending is similar to stacking but uses only a validation set from the train set to make predictions. The validation set and the predictions are used to build a model which is used for test set. It consists of following percentage of models prediction:
* Ridge: 0.05
* Lasso: 0.05
* ElasticNet: 0.05
* SVR: 0.05
* LGB: 0.1
* XGB: 0.3
* StackReg: 0.4

Final prediction score:
- RMSLE: 0.151
- RMSE: 952.618

### Summary
The model selection process is based on several factors such as training performance, performance metrics, validation performance, final results and whether it solves the problem. If it solves the business problem in the most efficient and consistent manner, it will be highly favourable as the model to deploy. 
For this business case, the scooter rental forecasting is trained on given dataset and requires the model to predict accurate results to predict their total users. The model that predicts the best score is the Stack Regressor model. It has the best RMSLE and RMSE scores. XGBoost is not far behind and could be chosen as the prefered model in other circumstances. For now the deployed model will be the Stack Regressor model. 