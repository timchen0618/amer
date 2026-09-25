import torch
import numpy as np
from models import LinearProbe, LinearRegressionProbe, MLPProbe, MLPRegressionProbe
from data import CustomDataset
from linear_probe import evaluate, evaluate_mse
from data import load_data

import torch.nn as nn

def load_test_data(batch_size=256):
    # Load test embeddings and labels
    test_embeddings = np.load('qembs_test.npy')
    test_lengths = np.load('test_lengths.npy')  # For classification
    test_lengths_actual = np.load('test_lengths_actual.npy')  # For regression
    
    # Create datasets
    test_dataset_cls = CustomDataset(test_embeddings, test_lengths)
    test_dataset_reg = CustomDataset(test_embeddings, test_lengths_actual)
    
    # Create dataloaders
    test_loader_cls = torch.utils.data.DataLoader(test_dataset_cls, batch_size=batch_size, shuffle=False)
    test_loader_reg = torch.utils.data.DataLoader(test_dataset_reg, batch_size=batch_size, shuffle=False)
    
    return test_loader_cls, test_loader_reg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the models
    classification_model = MLPProbe().to(device)
    regression_model = MLPRegressionProbe().to(device)
    
    # Load the saved model weights
    classification_model.load_state_dict(torch.load('checkpoint/best_mlp_probe.pt'))
    regression_model.load_state_dict(torch.load('checkpoint/best_mlp_probe_regression.pt'))
    
    # Load test data
    # test_loader_cls, test_loader_reg = load_test_data()
    data_cls = load_data()
    data_reg = load_data(regression=True)
    test_loader_cls = torch.utils.data.DataLoader(data_cls['test']['dataset'], batch_size=256, shuffle=False)
    dev_loader_cls = torch.utils.data.DataLoader(data_cls['dev']['dataset'], batch_size=256, shuffle=False)
    test_loader_reg = torch.utils.data.DataLoader(data_reg['test']['dataset'], batch_size=256, shuffle=False)
    dev_loader_reg = torch.utils.data.DataLoader(data_reg['dev']['dataset'], batch_size=256, shuffle=False)
    
    print('start')
    # Evaluate classification model
    classification_accuracy = evaluate(classification_model, test_loader_cls, device)
    print(f"Classification Test Accuracy: {classification_accuracy:.2f}%")
    
    # Evaluate regression model
    criterion = nn.MSELoss()
    regression_mse = evaluate_mse(regression_model, test_loader_reg, device)
    print(f"Regression Test MSE: {regression_mse:.4f}")
    
    
    # Add random baseline for classification
    # Load training labels to get class distribution
    train_lengths = np.load('train_lengths.npy')
    unique_classes, class_counts = np.unique(train_lengths, return_counts=True)
    class_probs = class_counts / len(train_lengths)

    # Make random predictions based on class distribution
    # test_lengths = np.load('test_lengths.npy')
    test_lengths = data_cls['test']['labels']
    random_predictions = np.random.choice(unique_classes, size=len(test_lengths), p=class_probs)
    random_accuracy = np.mean(random_predictions == test_lengths)
    print(f"Random Baseline Accuracy: {random_accuracy * 100:.2f}%")
    
    # Most frequent label baseline
    most_frequent_label = unique_classes[np.argmax(class_counts)]
    most_frequent_predictions = np.full_like(test_lengths, most_frequent_label)
    most_frequent_accuracy = np.mean(most_frequent_predictions == test_lengths)
    print(f"Most Frequent Label Baseline Accuracy: {most_frequent_accuracy * 100:.2f}%")
    
    
    # Add random baseline for regression
    # Random uniform baseline for regression
    # test_lengths_actual = np.load('test_lengths_actual.npy')
    test_lengths_actual = data_reg['test']['labels']
    random_uniform_predictions = np.random.uniform(0, 30, size=len(test_lengths_actual))
    random_uniform_mse = np.mean((random_uniform_predictions - test_lengths_actual) ** 2)
    print(f"Random Uniform Baseline MSE: {random_uniform_mse:.4f}")

    # Random Gaussian baseline for regression
    random_gaussian_predictions = np.random.normal(15, 3, size=len(test_lengths_actual))
    random_gaussian_mse = np.mean((random_gaussian_predictions - test_lengths_actual) ** 2)
    print(f"Random Gaussian Baseline MSE: {random_gaussian_mse:.4f}")
    
    random_gaussian_predictions = np.random.normal(14.5, 1, size=len(test_lengths_actual))
    random_gaussian_mse = np.mean((random_gaussian_predictions - test_lengths_actual) ** 2)
    print(f"Random Gaussian Baseline MSE: {random_gaussian_mse:.4f}")
    
    
    print('-------------------')
    print('repeat for test set')
    print('--------------------')
    
    # Evaluate classification model
    classification_accuracy = evaluate(classification_model, dev_loader_cls, device)
    print(f"Classification Dev Accuracy: {classification_accuracy:.2f}%")
    
    # Evaluate regression model
    criterion = nn.MSELoss()
    regression_mse = evaluate_mse(regression_model, dev_loader_reg, device)
    print(f"Regression Dev MSE: {regression_mse:.4f}")
    
    
    # Add random baseline for classification
    # Load training labels to get class distribution
    train_lengths = np.load('train_lengths.npy')
    unique_classes, class_counts = np.unique(train_lengths, return_counts=True)
    class_probs = class_counts / len(train_lengths)

    # Make random predictions based on class distribution
    # dev_lengths = np.load('dev_lengths.npy')
    dev_lengths = data_cls['dev']['labels']
    random_predictions = np.random.choice(unique_classes, size=len(dev_lengths), p=class_probs)
    random_accuracy = np.mean(random_predictions == dev_lengths)
    print(f"Random Baseline Accuracy: {random_accuracy * 100:.2f}%")
    
    
    # Add random baseline for regression
    # Random uniform baseline for regression
    # dev_lengths_actual = np.load('dev_lengths_actual.npy')
    dev_lengths_actual = data_reg['dev']['labels']
    random_uniform_predictions = np.random.uniform(0, 30, size=len(dev_lengths_actual))
    random_uniform_mse = np.mean((random_uniform_predictions - dev_lengths_actual) ** 2)
    print(f"Random Uniform Baseline MSE: {random_uniform_mse:.4f}")

    # Random Gaussian baseline for regression
    random_gaussian_predictions = np.random.normal(15, 3, size=len(dev_lengths_actual))
    random_gaussian_mse = np.mean((random_gaussian_predictions - dev_lengths_actual) ** 2)
    print(f"Random Gaussian Baseline MSE: {random_gaussian_mse:.4f}")
    
    random_gaussian_predictions = np.random.normal(14.5, 1, size=len(dev_lengths_actual))
    random_gaussian_mse = np.mean((random_gaussian_predictions - dev_lengths_actual) ** 2)
    print(f"Random Gaussian Baseline MSE: {random_gaussian_mse:.4f}")

if __name__ == "__main__":
    main()