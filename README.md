# ANN Assignment

This project implements a LeNet-5 style convolutional neural network in PyTorch for MNIST digit recognition. The script can train the model, evaluate saved weights, and run inference on a custom handwritten digit image.

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

The script creates `venv`, upgrades `pip`, installs the packages from `requirements.txt`, and activates the environment in the current session when possible.

## Train The Model

Train LeNet-5 on MNIST and save the learned weights:

```powershell
python main.py train
```

Training uses the defaults defined in `main.py`, including the MNIST normalization constants, Adam, a step learning-rate scheduler, and 30 epochs.

When training finishes, the model weights are saved to:

- `lenet5_weights.pth`
- `lenet5_weights.json`

The JSON file also includes the training history for each epoch.

## Evaluate Saved Weights

Evaluate a saved checkpoint on the MNIST test split:

```powershell
python main.py evaluate
```

To load the JSON snapshot instead of the PyTorch checkpoint, add `--weights json`:

```powershell
python main.py evaluate --weights json
```

## Predict A Custom Digit

Run inference on a single image file:

```powershell
python main.py test --image .\images\9.png
```

If needed, you can also load the JSON snapshot during inference:

```powershell
python main.py test --image .\images\9.png --weights json
```

The inference pipeline converts the image to grayscale, inverts light backgrounds automatically, crops the foreground digit, adds padding, resizes to 28x28, and normalizes using the MNIST statistics expected by the model.

For best results, use a single centered digit with minimal background clutter.

## Project Files

- `main.py` contains the model definition, training loop, evaluation helper, checkpoint saving/loading, and custom image prediction.
- `setup.ps1` bootstraps the Windows virtual environment and installs dependencies.
- `requirements.txt` lists the Python packages used by the project.
- `data/MNIST` is where the MNIST dataset is downloaded.
