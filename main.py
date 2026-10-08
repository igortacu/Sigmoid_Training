import numpy as np

from formulas import (
	binary_cross_entropy,
	mean_squared_error,
	normal_distribution,
	sigmoid,
	update_weights,
)

def main() -> None:
	x = 1.0
	mu = 0.0
	sigma = 1.0
	print("Normal distribution:", normal_distribution(x, mu, sigma))

	logit = 0.5
	print("Sigmoid:", sigmoid(logit))

	weights = np.array([0.1, -0.2])
	features = np.array([
		[1.0, 0.5],
		[1.0, 1.5],
	])
	targets = np.array([0.0, 1.0])
	predictions = np.array([0.4, 0.7])
	updated_weights = update_weights(weights, features, targets, predictions)
	print("Updated weights:", updated_weights)

	print("Mean squared error:", mean_squared_error(targets, predictions))
	print("Binary cross entropy:", binary_cross_entropy(targets, predictions))


if __name__ == '__main__':
	main()
