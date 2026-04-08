# ANN Assignment

This project trains a LeNet-5 style convolutional neural network on MNIST with TensorFlow, saves the trained model, and predicts custom handwritten digit images.

## Requirements

- Windows PowerShell 5.1 or PowerShell 7+
- Python 3.10+

## Setup

Run the PowerShell bootstrap script from the repository root:

```powershell
.\setup.ps1
```

If your execution policy blocks scripts, run it like this instead:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The script creates `venv`, upgrades `pip`, installs the pinned packages from `requirements.txt`, and activates the environment in the current session when possible.

## Train The Model

Train the model, evaluate it on the MNIST test split, and save the trained artifacts:

```powershell
python main.py
```

You can force retraining with custom hyperparameters:

```powershell
python main.py --train --epochs 10 --batch-size 128
```

Saved artifacts are written to:

- `artifacts\lenet5_mnist.keras`
- `artifacts\lenet5_mnist.weights.h5`

## Evaluate A Saved Model

If the saved artifacts already exist, evaluate them without retraining:

```powershell
python main.py --evaluate
```

## Predict Custom Digits

Pass one image, a folder, or a glob pattern to classify custom handwritten digits:

```powershell
python main.py .\images\7.png
python main.py .\images\*.png
```

The inference pipeline converts the image to grayscale, applies autocontrast, centers the digit, pads it to the LeNet-5 input size, and uses a simple inversion heuristic for white-background handwriting.

For best results, use a single centered digit per image with minimal background clutter.

## Project Files

- `main.py` contains training, saving, evaluation, and custom-image prediction.
- `setup.ps1` bootstraps the Windows virtual environment and installs dependencies.
