import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
#Load image data and corresponding labels
def load_images(base_directory):
    images = []
    labels = []
    
    for class_name in os.listdir(base_directory):
        class_dir = os.path.join(base_directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(class_dir, filename)
                img = Image.open(img_path)
                img = img.resize((64, 64))
                img_arr = np.array(img) / 255.0
                images.append(img_arr)
                labels.append(class_name)
    return np.array(images), np.array(labels)

def split_data(base_dir):
    X, y = load_images(base_dir)
    
    #Flatten the image data
    X_flat = X.reshape(X.shape[0], -1)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def perform_kfcv_and_train(X_train, y_train, X_val, y_val, model_class, param_grid):
    best_score = -float('inf')
    best_params = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for params in param_grid:
        model = model_class(**params)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    #Train the model with best parameters from cross-validation
    initial_model = model_class(**best_params)
    initial_model.fit(X_train, y_train)
    
    #Evaluate on validation set
    val_score = initial_model.score(X_val, y_val)
    
    #Fine-tune hyperparameters based on validation set performance
    fine_tuned_params = fine_tune_hyperparameters(initial_model, X_val, y_val, best_params)
    
    #Train final model with fine-tuned parameters
    final_model = model_class(**fine_tuned_params)
    final_model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    
    return final_model, fine_tuned_params

def fine_tune_hyperparameters(model, X_val, y_val, initial_params):
    # This function will depend on the specific model and hyperparameters
    if 'C' in initial_params:
        c_values = [initial_params['C'] * 0.5, initial_params['C'], initial_params['C'] * 1.5]
        best_c = max(c_values, key=lambda c: model.__class__(C=c, **{k: v for k, v in initial_params.items() if k != 'C'}).fit(X_val, y_val).score(X_val, y_val))
        initial_params['C'] = best_c
    
    return initial_params

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, conf_matrix

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#i.	Logistic regression
def logistic_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    lr_param_grid = [
        {'C': 0.1, 'penalty': 'l2'},
        {'C': 1.0, 'penalty': 'l2'},
        {'C': 10.0, 'penalty': 'l2'}
    ]
    lr_model, lr_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, LogisticRegression, lr_param_grid)
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(lr_model, X_test, y_test)
    
    print("Logistic Regression Results:")
    print(f"Best parameters after fine-tuning: {lr_best_params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plot_confusion_matrix(conf_matrix, np.unique(y_test))

#ii.	Support vector machine
def svm_model(X_train, y_train, X_val, y_val, X_test, y_test):
    svm_param_grid = [
        {'C': 0.1, 'kernel': 'rbf'},
        {'C': 1.0, 'kernel': 'rbf'},
        {'C': 10.0, 'kernel': 'rbf'}
    ]
    svm_model, svm_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, SVC, svm_param_grid)
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(svm_model, X_test, y_test)
    
    print("Support Vector Machine Results:")
    print(f"Best parameters: {svm_best_params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plot_confusion_matrix(conf_matrix, np.unique(y_test))
#iii.	Decision tree
def decision_tree_model(X_train, y_train, X_val, y_val, X_test, y_test):
    dt_param_grid = [
        {'max_depth': 5},
        {'max_depth': 10},
        {'max_depth': 15}
    ]
    dt_model, dt_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, DecisionTreeClassifier, dt_param_grid)
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(dt_model, X_test, y_test)
    
    print("Decision Tree Results:")
    print(f"Best parameters: {dt_best_params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plot_confusion_matrix(conf_matrix, np.unique(y_test))
    
#iv.	Multi-layer perceptron
def mlp_model(X_train, y_train, X_val, y_val, X_test, y_test):
    mlp_param_grid = [
        {'hidden_layer_sizes': (50,), 'activation': 'relu'},
        {'hidden_layer_sizes': (100,), 'activation': 'relu'},
        {'hidden_layer_sizes': (50, 50), 'activation': 'relu'}
    ]
    mlp_model, mlp_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, MLPClassifier, mlp_param_grid)
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(mlp_model, X_test, y_test)
    
    print("Multi-layer Perceptron Results:")
    print(f"Best parameters: {mlp_best_params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plot_confusion_matrix(conf_matrix, np.unique(y_test))
# Scikit-learn’s available models
def random_forest_model(X_train, y_train, X_val, y_val, X_test, y_test):
    rf_param_grid = [
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 10},
        {'n_estimators': 300, 'max_depth': 15}
    ]
    rf_model, rf_best_params = perform_kfcv_and_train(X_train, y_train, X_val, y_val, RandomForestClassifier, rf_param_grid)
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(rf_model, X_test, y_test)
    
    print("Random Forest Results:")
    print(f"Best parameters: {rf_best_params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plot_confusion_matrix(conf_matrix, np.unique(y_test))

if __name__ == "__main__":
    base_dir = "/Users/yangwenyu/Desktop/CS5100 /HW2/Dataset1"
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(base_dir)
    
    logistic_regression_model(X_train, y_train, X_val, y_val, X_test, y_test)
    svm_model(X_train, y_train, X_val, y_val, X_test, y_test)
    decision_tree_model(X_train, y_train, X_val, y_val, X_test, y_test)
    mlp_model(X_train, y_train, X_val, y_val, X_test, y_test)
    random_forest_model(X_train, y_train, X_val, y_val, X_test, y_test)