# -*- coding: utf-8 -*-
import numpy as np
def RF_feature_selector(X,y,test_size=0.8,n_search=30,n_repeats=10,limits={}):
    '''
    Feature importance selector based on random forest. The optimal set of hyperparameters is optimized through a random search.
    Importance is evaluated through the permutation method, which gives higher scores to fatures whose error metrics drops more after reshuffling.
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from scipy.stats import randint
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import mean_absolute_error

    #build train/test datasets
    data = np.hstack((X, y.reshape(-1, 1)))

    data = data[~np.isnan(data).any(axis=1)]
    train_set, test_set = train_test_split(data, random_state=42, test_size=test_size)

    X_train = train_set[:,0:-1]
    y_train = train_set[:,-1]

    X_test = test_set[:,0:-1]
    y_test = test_set[:,-1]
    
    #default grid of hyperparamters (Bodini and Optis, 2020)
    if limits=={}:
        p_grid = {'n_estimators': randint(low=10, high=100), # number of trees
                  'max_features': randint(low=1,high= 6), # number of features to consider when looking for the best split
                  'min_samples_split' : randint(low=2, high=11),
                  'max_depth' : randint(low=4, high=10),
                  'min_samples_leaf' : randint(low=1, high=15)
            }
        
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    forest_reg = RandomForestRegressor()
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions = p_grid, n_jobs = -1,
                                    n_iter=n_search, cv=5, scoring='neg_mean_squared_error')
    rnd_search.fit(X_train, y_train)
    print('Best set of hyperparameters found:')
    print(rnd_search.best_estimator_)

    predicted_test = rnd_search.best_estimator_.predict(X_test)
    test_mae = mean_absolute_error(y_test, predicted_test)
    print("Average testing MAE:", test_mae)

    predicted_train = rnd_search.best_estimator_.predict(X_train)
    train_mae = mean_absolute_error(y_train, predicted_train)
    print("Average training MAE:", train_mae)

    best_params=rnd_search.best_estimator_.get_params()    
        
    #random forest prediction with optimized hyperparameters
    reals=np.sum(np.isnan(np.hstack((X, y.reshape(-1, 1)))),axis=1)==0
    rnd_search.best_estimator_.fit(X[reals,:], y[reals])
        
    y_pred=y+np.nan
    y_pred[reals] = rnd_search.best_estimator_.predict(X[reals])
       
    reals=~np.isnan(y+y_pred)
    result = permutation_importance(rnd_search.best_estimator_, X[reals], y[reals], n_repeats=n_repeats, random_state=42, n_jobs=2)

    importance=result.importances_mean
    importance_std=result.importances_std
    
    return importance,importance_std,y_pred,test_mae,train_mae,best_params,y_test,predicted_test
