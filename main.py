from __future__ import annotations

import numpy as np
import tensorflow as tf


NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 1)
BATCH_SIZE = 128
EPOCHS = 50


def load_and_preprocess_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
	"""Load MNIST, normalize to [0, 1], add channel dim, and pad to 32x32."""
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	x_train = x_train.astype("float32") / 255.0
	x_test = x_test.astype("float32") / 255.0

	# CNNs expect an explicit channel dimension.
	x_train = np.expand_dims(x_train, axis=-1)
	x_test = np.expand_dims(x_test, axis=-1)

	# LeNet-5 expects 32x32 inputs, so pad MNIST's 28x28 images by 2 pixels per side.
	x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")
	x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")

	return (x_train, y_train), (x_test, y_test)


def build_lenet5() -> tf.keras.Model:
	"""Build a LeNet-5 inspired architecture for digit classification."""
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Conv2D(6, kernel_size=5, activation="tanh", input_shape=INPUT_SHAPE),
			tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
			tf.keras.layers.Conv2D(16, kernel_size=5, activation="tanh"),
			tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(120, activation="tanh"),
			tf.keras.layers.Dense(84, activation="tanh"),
			tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
		]
	)

	model.compile(
		optimizer="adam",
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def preview_predictions(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray, count: int = 10) -> None:
	"""Print predicted and actual labels for a few test samples."""
	sample_images = x_test[:count]
	sample_labels = y_test[:count]

	probabilities = model.predict(sample_images, verbose=0)
	predicted_labels = np.argmax(probabilities, axis=1)

	print("\nSample predictions:")
	for idx in range(count):
		print(
			f"Index {idx:2d} | predicted: {predicted_labels[idx]} "
			f"| actual: {sample_labels[idx]} | confidence: {probabilities[idx][predicted_labels[idx]]:.4f}"
		)


def main() -> None:
	tf.random.set_seed(42)
	np.random.seed(42)

	(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

	print(f"Training data shape: {x_train.shape}")
	print(f"Test data shape: {x_test.shape}")

	model = build_lenet5()
	model.summary()

	model.fit(
		x_train,
		y_train,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		validation_split=0.1,
		verbose=2,
	)

	test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
	print(f"\nTest loss: {test_loss:.4f}")
	print(f"Test accuracy: {test_accuracy:.4f}")

	preview_predictions(model, x_test, y_test, count=10)


if __name__ == "__main__":
	main()
