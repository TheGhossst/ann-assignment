from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "lenet5_mnist.keras"
DEFAULT_WEIGHTS_PATH = ARTIFACTS_DIR / "lenet5_mnist.weights.h5"
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}


def set_random_seed(seed: int = 42) -> None:
	np.random.seed(seed)
	tf.random.set_seed(seed)


def load_mnist_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode="constant")
	x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode="constant")

	x_train = x_train.astype(np.float32) / 255.0
	x_test = x_test.astype(np.float32) / 255.0

	x_train = x_train[..., np.newaxis]
	x_test = x_test[..., np.newaxis]

	return x_train, y_train, x_test, y_test


def build_lenet5_model(
	input_shape: tuple[int, int, int] = (32, 32, 1),
	num_classes: int = 10,
	learning_rate: float = 1e-3,
) -> tf.keras.Model:
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Input(shape=input_shape),
			tf.keras.layers.Conv2D(6, kernel_size=5, activation="tanh", padding="valid"),
			tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
			tf.keras.layers.Conv2D(16, kernel_size=5, activation="tanh", padding="valid"),
			tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(120, activation="tanh"),
			tf.keras.layers.Dense(84, activation="tanh"),
			tf.keras.layers.Dense(num_classes, activation="softmax"),
		]
	)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def train_model(
	model: tf.keras.Model,
	x_train: np.ndarray,
	y_train: np.ndarray,
	epochs: int,
	batch_size: int,
) -> tf.keras.callbacks.History:
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_accuracy",
			patience=3,
			restore_best_weights=True,
		)
	]

	history = model.fit(
		x_train,
		y_train,
		validation_split=0.1,
		epochs=epochs,
		batch_size=batch_size,
		verbose=2,
		callbacks=callbacks,
	)
	return history


def save_model_artifacts(
	model: tf.keras.Model,
	model_path: Path = DEFAULT_MODEL_PATH,
	weights_path: Path = DEFAULT_WEIGHTS_PATH,
) -> None:
	model_path.parent.mkdir(parents=True, exist_ok=True)
	model.save(model_path)
	model.save_weights(weights_path)


def load_saved_model(
	model_path: Path = DEFAULT_MODEL_PATH,
	weights_path: Path = DEFAULT_WEIGHTS_PATH,
) -> tf.keras.Model | None:
	if model_path.exists():
		return tf.keras.models.load_model(model_path)

	if weights_path.exists():
		model = build_lenet5_model()
		model.load_weights(weights_path)
		return model

	return None


def preprocess_custom_image(image_path: str | Path) -> np.ndarray:
	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"Image not found: {path}")

	image = Image.open(path).convert("L")
	image = ImageOps.autocontrast(image)
	image = ImageOps.contain(image, (28, 28), method=Image.Resampling.LANCZOS)

	canvas = Image.new("L", (28, 28), color=0)
	left = (28 - image.width) // 2
	top = (28 - image.height) // 2
	canvas.paste(image, (left, top))

	array = np.asarray(canvas, dtype=np.float32) / 255.0
	border_pixels = np.concatenate(
		[
			array[:3, :].ravel(),
			array[-3:, :].ravel(),
			array[:, :3].ravel(),
			array[:, -3:].ravel(),
		]
	)

	if border_pixels.mean() > array.mean():
		array = 1.0 - array

	array = np.pad(array, ((2, 2), (2, 2)), mode="constant")
	return array[..., np.newaxis]


def expand_image_inputs(image_inputs: Sequence[str]) -> list[Path]:
	resolved_paths: list[Path] = []
	seen: set[str] = set()

	for item in image_inputs:
		matches: list[Path] = []
		if any(character in item for character in "*?[]"):
			matches = [Path(match) for match in glob.glob(item, recursive=True)]
		else:
			candidate = Path(item)
			if candidate.is_dir():
				matches = [
					child
					for child in sorted(candidate.iterdir())
					if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
				]
			else:
				matches = [candidate]

		for path in matches:
			key = str(path.resolve())
			if path.is_file() and key not in seen:
				seen.add(key)
				resolved_paths.append(path)

	return resolved_paths


def predict_image(model: tf.keras.Model, image_path: str | Path) -> tuple[int, float, np.ndarray]:
	sample = preprocess_custom_image(image_path)
	probabilities = model.predict(sample[np.newaxis, ...], verbose=0)[0]
	digit = int(np.argmax(probabilities))
	confidence = float(probabilities[digit])
	return digit, confidence, probabilities


def ensure_model(
	force_train: bool,
	default_train: bool,
	epochs: int,
	batch_size: int,
	model_path: Path,
	weights_path: Path,
	x_train: np.ndarray,
	y_train: np.ndarray,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History | None, bool]:
	should_train = force_train or default_train or not (model_path.exists() or weights_path.exists())

	if should_train:
		model = build_lenet5_model()
		history = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
		save_model_artifacts(model, model_path=model_path, weights_path=weights_path)
		return model, history, True

	model = load_saved_model(model_path=model_path, weights_path=weights_path)
	if model is None:
		model = build_lenet5_model()
		history = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
		save_model_artifacts(model, model_path=model_path, weights_path=weights_path)
		return model, history, True

	return model, None, False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train a LeNet-5 style CNN on MNIST and predict custom handwritten digits."
	)
	parser.add_argument("images", nargs="*", help="Custom digit image paths or glob patterns to predict.")
	parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
	parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
	parser.add_argument("--train", action="store_true", help="Force retraining even if a saved model exists.")
	parser.add_argument("--evaluate", action="store_true", help="Evaluate on the MNIST test split.")
	parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved model file.")
	parser.add_argument(
		"--weights-path",
		type=Path,
		default=DEFAULT_WEIGHTS_PATH,
		help="Path to the saved model weights file.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	set_random_seed()

	saved_model_exists = args.model_path.exists() or args.weights_path.exists()
	default_train = not args.images and not args.evaluate
	should_train = args.train or default_train or not saved_model_exists
	should_evaluate = args.evaluate or default_train

	x_train: np.ndarray | None = None
	y_train: np.ndarray | None = None
	x_test: np.ndarray | None = None
	y_test: np.ndarray | None = None

	if should_train or should_evaluate:
		x_train, y_train, x_test, y_test = load_mnist_dataset()

	if should_train:
		if x_train is None or y_train is None:
			raise RuntimeError("Training data could not be loaded.")

		model, history, trained = ensure_model(
			force_train=True,
			default_train=False,
			epochs=args.epochs,
			batch_size=args.batch_size,
			model_path=args.model_path,
			weights_path=args.weights_path,
			x_train=x_train,
			y_train=y_train,
		)
	else:
		model = load_saved_model(model_path=args.model_path, weights_path=args.weights_path)
		if model is None:
			raise RuntimeError(
				"No saved model was found. Run with --train first to create artifacts."
			)
		history = None
		trained = False

	if trained and history is not None:
		final_val_accuracy = history.history.get("val_accuracy", [])
		final_accuracy = history.history.get("accuracy", [])
		if final_accuracy:
			print(f"Training accuracy: {final_accuracy[-1]:.4f}")
		if final_val_accuracy:
			print(f"Validation accuracy: {max(final_val_accuracy):.4f}")
		print(f"Saved model to: {args.model_path}")
		print(f"Saved weights to: {args.weights_path}")

	if should_evaluate:
		if x_test is None or y_test is None:
			raise RuntimeError("Test data could not be loaded.")

		loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
		print(f"Test loss: {loss:.4f}")
		print(f"Test accuracy: {accuracy:.4f}")

	image_paths = expand_image_inputs(args.images)
	if image_paths:
		print("Custom image predictions:")
		for image_path in image_paths:
			digit, confidence, probabilities = predict_image(model, image_path)
			top_indices = np.argsort(probabilities)[-3:][::-1]
			top_summary = ", ".join(
				f"{index}={probabilities[index]:.2%}" for index in top_indices
			)
			print(f"{image_path}: predicted {digit} ({confidence:.2%}) | top 3: {top_summary}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
