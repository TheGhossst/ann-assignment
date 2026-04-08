# ANN Assignment

This project trains and evaluates a LeNet-5 style convolutional neural network on the MNIST handwritten digits dataset using TensorFlow.

## What it does

- Loads the MNIST dataset.
- Normalizes pixel values from `0-255` to `0-1`.
- Expands the image shape to include a channel dimension.
- Pads `28x28` images to `32x32` for LeNet-5 compatibility.
- Builds a LeNet-5 inspired CNN with convolution, average pooling, and dense layers.
- Trains the model, evaluates it on the test set, and prints sample predictions.

## Requirements

- [Python 3.10.11](https://www.python.org/downloads/windows/)
- Windows PowerShell or Command Prompt

## Setup

Run the Windows setup script from the project root:

```powershell
.\setup.ps1
```

The script will:

- Create a virtual environment in `venv` if it does not already exist.
- Upgrade `pip`.
- Install the packages listed in `requirements.txt`.

## Run

After setup, run the training script with:

```powershell
.\venv\Scripts\python.exe .\main.py
```

## Notes

- The first run may download the MNIST dataset automatically.
- TensorFlow may print deprecation warnings from internal libraries. These do not stop the program from running.
