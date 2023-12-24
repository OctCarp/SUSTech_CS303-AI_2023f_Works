import pickle
import numpy as np
import matplotlib.pyplot as plt


def save_data(file_name, data):
    """
    Save ndarray to file_name.pkl 
    
    Parameters:
    - file_name: *.pkl
    - data: numpy array
    """
    try:
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        print("Saved successfully")
    except Exception as e:
        print(f"Error saving data to {file_name}: {str(e)}")

def load_data(file_name):
    """
    Load ndarray from file_name.pkl
    
    Parameters:
    - file_name: *.pkl

    Returns:
    - data: numpy array
    """
    try:
        with open(file_name, 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data
    except FileNotFoundError:
        print(f"File {file_name} not found")
    except Exception as e:
        print(f"Error loading data from {file_name}: {str(e)}")


def split_train_validation(data, labels, train_ratio=0.8, random_seed=None):
    """
    Split a dataset into training and validation sets.

    Parameters:
    - data: numpy array, input data samples
    - labels: numpy array, labels corresponding to the data samples
    - train_ratio: float, ratio of training data (default: 0.8)
    - random_seed: int, random seed for reproducibility (default: None)

    Returns:
    - train_data, train_labels: numpy arrays, training data and labels
    - validation_data, validation_labels: numpy arrays, validation data and labels
    """
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError("Train ratio must be between 0 and 1")

    if data.shape[0] != labels.shape[0]:
        raise ValueError("Data and labels must have the same number of samples")

    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    num_train_samples = int(train_ratio * num_samples)

    train_indices = indices[:num_train_samples]
    validation_indices = indices[num_train_samples:]

    train_data, train_labels = data[train_indices], labels[train_indices]
    validation_data, validation_labels = data[validation_indices], labels[validation_indices]

    return (train_data, train_labels), (validation_data, validation_labels)



def plot_loss_curves(train_losses, val_losses):
    """
    Plot train loss and validation loss curve.

    Parameter:
    - train_losses: train losses history
    - val_losses: validation losses history
    """
    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Iterations')
    plt.grid(True)
    plt.show()
    
def plot_acc_curves(train_acc, val_acc):
    """
    Plot train loss and validation loss curve.

    Parameter:
    - train_acc: train accuracies history
    - val_acc: validation accuracies history
    """
    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(train_acc, label='Training Accuracy', linewidth=2)
    plt.plot(val_acc, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies Over Iterations')
    plt.grid(True)
    plt.show()
    