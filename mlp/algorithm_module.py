import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Split train data into train/test sets, k num of folds, shuffle data before split, set constant random generator 
k_folds = KFold(n_splits=10, shuffle=True, random_state=0)

# Assign diff alphas values to find best fit for model
r_alphas = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
l_alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]     # values close to 1 better choice based on documentation

class Algorithms:
    
    def __init__(self):
        pass

    def rmseCV(self, X, y, model):
        """Cross validation training root mean squared error"""
        rmse = np.sqrt(-cross_val_score(model, X, y, 
                                        scoring="neg_mean_squared_error", cv=k_folds))
        return rmse
    
    def rmsle(self, y, y_pred):
        """Root mean squared log error(RMSLE) chosen to scale down outliers, nullify their effects (robust)"""
        assert len(y) == len(y_pred)
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y))**2))
    
    def rmse(self, y, y_pred):
        """Root mean squared error(RMSE), prone to outliers"""
        return np.sqrt(mean_squared_error(y, y_pred))
    
    def regularization_models(self):
        """Regularization models (prevent overfitting using penalty on coeff), use pipelines to combine with Robust scaler"""
        # Ridge model using pipeline to add robustscaler (scale feature based on percentiles, wont be affected by outliers)
        self.ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=r_alphas, cv=k_folds))
        # Lasso model
        self.lasso = make_pipeline(RobustScaler(), LassoCV(alphas=l_alphas, max_iter=1e7, cv=k_folds, random_state=0))
        # ElasticNet model
        self.elastic_net = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=e_l1ratio, alphas=e_alphas, max_iter=1e7, cv=k_folds))
        
    def regression_models(self):  
        """Regression models"""
        # Support Vector Regression (SVR) used for working with continuous values (tune para for diff results)
        self.svr = make_pipeline(RobustScaler(), SVR(gamma=0.1, C=20, epsilon=0.008))    # Small gamma value define a Gaussian function with a large variance
        
        # Light GBM
        self.lightgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000,
                                 max_bin=200, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7,
                                 feature_fraction=0.2, feature_fraction_seed=7, verbose=-1)
        # XGBoost
        self.xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0,
                               gamma=0, subsample=0.7, colsample_bytree=0.7, objective='reg:linear', nthread=-1, 
                               scale_pos_weight=1, seed=27, reg_alpha=0.00006)
        # Emsemble learning, using multiple regressors to predict 
        self.stack_reg = StackingCVRegressor(regressors=(self.ridge, self.lasso, self.elastic_net, self.svr, self.lightgbm, self.xgboost),
                                        meta_regressor=self.xgboost, use_features_in_secondary=True)\
    
    def fit_models(self, X, y):
        """Fitting models"""
        print('Fitting Model...')   
        print('Ridge') 
        self.ridge_model = self.ridge.fit(X, y)
        print('Lasso')
        self.lasso_model = self.lasso.fit(X, y)
        print('ElasticNet')
        self.elastic_model = self.elastic_net.fit(X, y)
        print('SVR')
        self.svr_model = self.svr.fit(X, y)
        print('LightGBM')
        self.lgb_model = self.lightgbm.fit(X, y)
        print('XGBoost')
        self.xgb_model = self.xgboost.fit(X, y)
        print('StackRegressor')
        self.stack_reg_models = self.stack_reg.fit(np.array(X), np.array(y))
        
    def blend_models_predict(self, X):
        """Tunable Ensemble technique to combine percentage of each model to possibly obtain the best overall prediction value"""
        return ((0.05 * self.ridge_model.predict(X)) +
                (0.05 * self.lasso_model.predict(X)) + 
                (0.05 * self.elastic_model.predict(X)) +    
                (0.05 * self.svr_model.predict(X)) + 
                (0.1 * self.lgb_model.predict(X)) + 
                (0.3 * self.xgb_model.predict(X)) +             
                (0.4 * self.stack_reg_models.predict(np.array(X))))
    