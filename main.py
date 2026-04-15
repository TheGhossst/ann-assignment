IN_CHANNELS       = 1        # Greyscale images  → 1 channel
NUM_CLASSES       = 10       # Digits 0-9

# LeNet-5 layer sizes
CONV1_OUT         = 6        # C1 : feature maps after 1st conv
CONV2_OUT         = 16       # C3 : feature maps after 2nd conv
CONV3_OUT         = 120      # C5 : fully-connected conv layer
FC1_OUT           = 84       # F6 : fully-connected hidden layer

KERNEL_SIZE       = 5        # All convolution kernels are 5×5
POOL_SIZE         = 2        # Average-pooling window  2×2
POOL_STRIDE       = 2        # Non-overlapping pooling


BATCH_SIZE        = 64       # Mini-batch size
LEARNING_RATE     = 1e-3     # Adam initial learning rate
NUM_EPOCHS        = 30       # Training epochs
WEIGHT_DECAY      = 1e-4     # L2 regularisation coefficient
LR_STEP_SIZE      = 5        # Decay LR every N epochs
LR_GAMMA          = 0.5      # LR multiplier at each step

NORM_MEAN         = (0.1307,)
NORM_STD          = (0.3081,)


WEIGHTS_PT_PATH   = "lenet5_weights.pth"   # PyTorch checkpoint
WEIGHTS_JSON_PATH = "lenet5_weights.json"

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


class LeNet5(nn.Module):
    """
    Classic LeNet-5 adapted for MNIST (28×28 input, padding on C1
    so spatial size is preserved before the first pooling step).

    Layer flow
    ----------
    Input  : (N, 1, 28, 28)
    C1     : Conv 5×5, pad=2  → (N,  6, 28, 28)  + Tanh
    S2     : AvgPool 2×2      → (N,  6, 14, 14)
    C3     : Conv 5×5         → (N, 16, 10, 10)  + Tanh
    S4     : AvgPool 2×2      → (N, 16,  5,  5)
    C5     : Conv 5×5         → (N, 120,  1,  1) + Tanh  (= FC)
    Flatten: (N, 120)
    F6     : Linear 120→84    + Tanh
    Output : Linear 84→10
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, CONV1_OUT,
                      kernel_size=KERNEL_SIZE, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(POOL_SIZE, POOL_STRIDE),
            nn.Conv2d(CONV1_OUT, CONV2_OUT,
                      kernel_size=KERNEL_SIZE),
            nn.Tanh(),
            nn.AvgPool2d(POOL_SIZE, POOL_STRIDE),

            nn.Conv2d(CONV2_OUT, CONV3_OUT,
                      kernel_size=KERNEL_SIZE),
            nn.Tanh(),
        )


        self.classifier = nn.Sequential(
            nn.Linear(CONV3_OUT, FC1_OUT),
            nn.Tanh(),

            nn.Linear(FC1_OUT, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x


def get_data_loaders():
    """Download MNIST and return (train_loader, test_loader)."""
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    train_set = datasets.MNIST(root="./data", train=True,
                               download=True, transform=transform)
    test_set  = datasets.MNIST(root="./data", train=False,
                               download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2,
                              pin_memory=True)
    return train_loader, test_loader


def train(model, device, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=LR_STEP_SIZE,
                                          gamma=LR_GAMMA)

    print(f"\n{'='*55}")
    print(f"  Training LeNet-5 on {device}")
    print(f"  Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}"
          f"  |  LR: {LEARNING_RATE}")
    print(f"{'='*55}\n")

    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss   = 0.0
        running_correct = 0

        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss    += loss.item() * images.size(0)
            preds            = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()

        scheduler.step()

        train_loss = running_loss    / len(train_loader.dataset)
        train_acc  = running_correct / len(train_loader.dataset)

        val_loss, val_acc = evaluate(model, device,
                                     test_loader, criterion)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%  |  "
              f"LR: {lr_now:.6f}  ({elapsed:.1f}s)")

        history.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "train_acc" : round(train_acc,  6),
            "val_loss"  : round(val_loss,   6),
            "val_acc"   : round(val_acc,    6),
        })

    print(f"\n  Final validation accuracy: {val_acc*100:.2f}%\n")
    return history

def evaluate(model, device, loader, criterion=None):
    """Return (avg_loss, accuracy)."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss    = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss    / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy



def save_weights_pt(model):
    """Save full model state-dict as a .pth file."""
    torch.save(model.state_dict(), WEIGHTS_PT_PATH)
    print(f"  [✓] PyTorch weights saved → {WEIGHTS_PT_PATH}")


def save_weights_json(model, history=None):
    """
    Save every parameter tensor as a nested JSON file.
    Large tensors are stored as lists of floats (rounded to 6 dp).
    This is primarily for inspection / portability.
    """
    data = {"parameters": {}, "training_history": history or []}

    for name, tensor in model.state_dict().items():
        arr = tensor.cpu().numpy()
        # Round to keep file size manageable
        data["parameters"][name] = np.round(arr, 6).tolist()

    with open(WEIGHTS_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

    size_kb = os.path.getsize(WEIGHTS_JSON_PATH) / 1024
    print(f"  [✓] JSON weights saved  → {WEIGHTS_JSON_PATH}"
          f"  ({size_kb:.0f} KB)")


# ─────────────────────────────────────────────────────────
#  Loading Weights
# ─────────────────────────────────────────────────────────
def load_weights(model, device, source="pt"):
    """Load weights from .pth (default) or reconstructed from .json."""
    if source == "json":
        print(f"  Loading weights from {WEIGHTS_JSON_PATH} …")
        with open(WEIGHTS_JSON_PATH, "r") as f:
            data = json.load(f)
        state = {k: torch.tensor(np.array(v))
                 for k, v in data["parameters"].items()}
        model.load_state_dict(state)
    else:
        print(f"  Loading weights from {WEIGHTS_PT_PATH} …")
        model.load_state_dict(
            torch.load(WEIGHTS_PT_PATH, map_location=device))

    model.to(device)
    print("  [✓] Weights loaded successfully.\n")
    return model


# ─────────────────────────────────────────────────────────
#  Single-Image Inference
# ─────────────────────────────────────────────────────────
def predict_image(model, device, image_path):
    if not os.path.isfile(image_path):
        print(f"  [✗] File not found: {image_path}")
        return

    img = Image.open(image_path).convert("L")  # greyscale

    # ── Auto-invert if background is light (MNIST = dark bg) ──
    pixel_mean = np.array(img).mean()
    if pixel_mean > 127:          # light background → invert
        img = Image.fromarray(255 - np.array(img))

    # ── Center the digit with padding (mimics MNIST style) ────
    img_arr = np.array(img)
    rows    = np.any(img_arr > 30, axis=1)
    cols    = np.any(img_arr > 30, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img_arr    = img_arr[rmin:rmax+1, cmin:cmax+1]  # tight crop

    # Add 20% padding on each side, then resize to 28×28
    h, w    = img_arr.shape
    pad     = int(max(h, w) * 0.2)
    padded  = np.pad(img_arr, pad, constant_values=0)
    img     = Image.fromarray(padded)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits     = model(tensor)
        probs      = torch.softmax(logits, dim=1).squeeze()
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item() * 100
        
    print(f"\n  Image : {image_path}")
    print(f"  ┌─────────────────────────────┐")
    print(f"  │  Predicted digit : {pred_class}          │")
    print(f"  │  Confidence      : {confidence:6.2f}%  │")
    print(f"  └─────────────────────────────┘")
    print(f"\n  Per-class probabilities:")
    for digit, p in enumerate(probs.tolist()):
        bar = "█" * int(p * 30)
        print(f"    {digit}  {bar:<30}  {p*100:5.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="LeNet-5 MNIST – Train, Evaluate, or Test")

    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "test"],
        help=(
            "train    – train the model and save weights\n"
            "evaluate – load saved weights and run on the test set\n"
            "test     – predict a digit from a custom image file"
        ),
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to image file (required for 'test' mode)",
    )
    parser.add_argument(
        "--weights", "-w",
        choices=["pt", "json"],
        default="pt",
        help="Weight format to load: 'pt' (default) or 'json'",
    )

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps"  if torch.backends.mps.is_available()
                          else "cpu")

    model = LeNet5().to(device)

    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"\n  Device        : {device}")
    print(f"  Model         : LeNet-5")
    print(f"  Total params  : {total_params:,}")

    if args.mode == "train":
        train_loader, test_loader = get_data_loaders()
        history = train(model, device, train_loader, test_loader)
        save_weights_pt(model)
        save_weights_json(model, history)

    elif args.mode == "evaluate":
        model  = load_weights(model, device, source=args.weights)
        _, test_loader = get_data_loaders()
        _, acc = evaluate(model, device, test_loader)
        print(f"  Test Accuracy : {acc*100:.2f}%")

    elif args.mode == "test":
        if args.image is None:
            parser.error("--image/-i is required for 'test' mode")
        model = load_weights(model, device, source=args.weights)
        predict_image(model, device, args.image)


if __name__ == "__main__":
    main()