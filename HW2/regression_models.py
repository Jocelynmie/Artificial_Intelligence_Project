import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from PIL import Image
from sklearn.metrics import r2_score


#Load image data and corresponding prices
def load_regression_data(base_directory):
    images = []
    prices = []
    
    for filename in os.listdir(base_directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(base_directory, filename)
            img = Image.open(img_path)
            #Resize images to a consistent size
            img = img.resize((64, 64))
            #Normalize pixel values
            img_arr = np.array(img) / 255.0
            images.append(img_arr)
            # price_xxx
            #Extract price from filename
            price = float(filename.split('_')[1].split('.')[0])
            prices.append(price)
    
    X = np.array(images)
    y = np.array(prices)
    
    return X, y
#Split data into train, validation, and test sets
def split_regression_data(X, y):
    X_flat = X.reshape(X.shape[0], -1)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

#Perform k-fold cross-validation and train the model
def perform_kfcv_and_train(X_train, y_train, X_val, y_val, model_class, param_grid,n_jobs=1):
    best_score = float('inf')
    best_params = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    #Grid search over parameter combinations
    for params in param_grid:
        model = model_class(**params)
        scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
    
    #Train the model with best parameters from cross-validation
    initial_model = model_class(**best_params)
    initial_model.fit(X_train, y_train)
    
    #Evaluate on validation set
    val_score = mean_squared_error(y_val, initial_model.predict(X_val))
    
    #Fine-tune hyperparameters based on validation set performance
    fine_tuned_params = fine_tune_hyperparameters(initial_model, X_val, y_val, best_params)
    
    #Train final model with fine-tuned parameters
    final_model = model_class(**fine_tuned_params)
    final_model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    
    return final_model, fine_tuned_params

def fine_tune_hyperparameters(model, X_val, y_val, initial_params):
    # This function will depend on the specific model and hyperparameters
    if 'alpha' in initial_params:
        alpha_values = [initial_params['alpha'] * 0.5, initial_params['alpha'], initial_params['alpha'] * 1.5]
        best_alpha = min(alpha_values, key=lambda a: mean_squared_error(y_val, model.__class__(alpha=a, **{k: v for k, v in initial_params.items() if k != 'alpha'}).fit(X_val, y_val).predict(X_val)))
        initial_params['alpha'] = best_alpha
    
    return initial_params

#Evaluate regression model performance
def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse, y_pred

#Plot scatter plot of true vs predicted prices
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
#i.	Linear regression
def linear_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    lr_param_grid = [{'fit_intercept': True}, {'fit_intercept': False}]
    lr_model, lr_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, LinearRegression, lr_param_grid)
    mse, rmse,y_pred = evaluate_regression_model(lr_model, X_test, y_test)
    
    print("Linear Regression Results:")
    print(f"Best parameters after fine-tuning: {lr_best_params}")
    print(f"Mean Squared Error: {mse:.4f}")
    # print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # additional_analysis(y_test, y_pred)
    plot_scatter(y_test, y_pred, "Linear Regression: True vs Predicted Price")
#ii.	Polynomial regression
def polynomial_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    degrees = [2]
    poly_param_grid = [{'degree': d} for d in degrees]
    
    best_mse = float('inf')
    best_model = None
    best_degree = None
    
    for params in poly_param_grid: 
        poly_features = PolynomialFeatures(degree=params['degree'])
        X_train_poly = poly_features.fit_transform(X_train)
        X_val_poly = poly_features.transform(X_val)
        
        lr_model, _ = perform_kfcv_and_train(X_train_poly, y_train, X_val_poly, y_val, LinearRegression, [{'fit_intercept': True}])
        mse, _, _ = evaluate_regression_model(lr_model, poly_features.transform(X_val), y_val)
        
        if mse < best_mse:
            best_mse = mse
            best_model = lr_model
            best_degree = params['degree']
    
    X_test_poly = PolynomialFeatures(degree=best_degree).fit_transform(X_test)
    mse, rmse, y_pred = evaluate_regression_model(best_model, X_test_poly, y_test)
    
    print("Polynomial Regression Results:")
    print(f"Best degree: {best_degree}")
    print(f"Mean Squared Error: {mse:.4f}")
    # print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # additional_analysis(y_test, y_pred)
    plot_scatter(y_test, y_pred, f"Polynomial Regression (degree={best_degree}): True vs Predicted Price")
#Support Vector Regression model
def svr_model(X_train, y_train, X_val, y_val, X_test, y_test):
    svr_param_grid = [
        {'kernel': 'rbf', 'C': 0.1, 'epsilon': 0.1},
        {'kernel': 'rbf', 'C': 1, 'epsilon': 0.1},
        {'kernel': 'rbf', 'C': 10, 'epsilon': 0.1},
        {'kernel': 'linear', 'C': 0.1, 'epsilon': 0.1},
        {'kernel': 'linear', 'C': 1, 'epsilon': 0.1},
        {'kernel': 'linear', 'C': 10, 'epsilon': 0.1}
    ]
    svr_model, svr_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, SVR, svr_param_grid)
    mse, rmse, y_pred = evaluate_regression_model(svr_model, X_test, y_test)
    
    print("Support Vector Regression Results:")
    print(f"Best parameters: {svr_best_params}")
    print(f"Mean Squared Error: {mse:.4f}")

    
    # additional_analysis(y_test, y_pred)
    plot_scatter(y_test, y_pred, "Support Vector Regression: True vs Predicted Price")

#Multi-layer Perceptron Regression model
def mlp_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    mlp_param_grid = [
        {'hidden_layer_sizes': (50,), 'activation': 'relu'},
        {'hidden_layer_sizes': (100,), 'activation': 'relu'},
        {'hidden_layer_sizes': (50, 50), 'activation': 'relu'},
        {'hidden_layer_sizes': (50,), 'activation': 'tanh'},
        {'hidden_layer_sizes': (100,), 'activation': 'tanh'},
        {'hidden_layer_sizes': (50, 50), 'activation': 'tanh'}
    ]
    mlp_model, mlp_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, MLPRegressor, mlp_param_grid)
    mse, rmse, y_pred = evaluate_regression_model(mlp_model, X_test, y_test)
    
    print("Multi-layer Perceptron Regression Results:")
    print(f"Best parameters: {mlp_best_params}")
    print(f"Mean Squared Error: {mse:.4f}")
    # print(f"Root Mean Squared Error: {rmse:.4f}")
    
    #additional_analysis(y_test, y_pred)
    plot_scatter(y_test, y_pred, "Multi-layer Perceptron Regression: True vs Predicted Price")

#Random Forest Regression model
def random_forest_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    rf_param_grid = [
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': None},
        {'n_estimators': 200, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 200, 'max_depth': None}
    ]
    rf_model, rf_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, RandomForestRegressor, rf_param_grid)
    mse, rmse, y_pred = evaluate_regression_model(rf_model, X_test, y_test)
    
    print("Random Forest Regression Results:")
    print(f"Best parameters: {rf_best_params}")
    print(f"Mean Squared Error: {mse:.4f}")
    # print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # additional_analysis(y_test, y_pred)
    plot_scatter(y_test, y_pred, "Random Forest Regression: True vs Predicted Price")
    
    
if __name__ == "__main__":
    base_dir = "/Users/yangwenyu/Desktop/CS5100 /HW2/Dataset2"
    X, y = load_regression_data(base_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_regression_data(X, y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    
    # Run different regression models
    linear_regression_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    polynomial_regression_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    svr_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    mlp_regression_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    random_forest_regression_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)