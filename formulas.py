import numpy as np

def normal_distribution(x: float, mu: float, sigma: float) -> float:
	"""Return the normal probability density at ``x``."""
	if sigma <= 0:
		raise ValueError("sigma must be greater than zero")

	coefficient = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
	exponent = -0.5 * ((x - mu) / sigma) ** 2
	return float(coefficient * np.exp(exponent))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
	"""Map real-valued inputs to probabilities between zero and one."""
	values = np.asarray(x, dtype=float)
	# This equivalent form avoids overflow for large negative inputs.
	result = np.empty_like(values)
	positive = values >= 0
	result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
	exp_values = np.exp(values[~positive])
	result[~positive] = exp_values / (1.0 + exp_values)
	return float(result) if result.ndim == 0 else result


def update_weights(
	w: np.ndarray,
	X: np.ndarray,
	y: np.ndarray,
	y_hat: np.ndarray,
	alpha: float = 0.0005,
) -> np.ndarray:
	"""Apply one batch gradient update using the supplied predictions."""
	weights = np.asarray(w, dtype=float)
	features = np.asarray(X, dtype=float)
	targets = np.asarray(y, dtype=float)
	predictions = np.asarray(y_hat, dtype=float)

	if weights.ndim != 1 or features.ndim != 2:
		raise ValueError("w must be 1-D and X must be 2-D")
	if features.shape[1] != weights.size:
		raise ValueError("X must have one column per weight")
	if targets.ndim != 1 or predictions.ndim != 1:
		raise ValueError("y and y_hat must be 1-D arrays")
	if features.shape[0] == 0 or targets.size != features.shape[0] or predictions.size != targets.size:
		raise ValueError("X, y, and y_hat must contain the same nonzero number of samples")
	if not 0 <= alpha <= 1:
		raise ValueError("alpha must be between 0 and 1")

	gradient = features.T @ (predictions - targets) / targets.size
	return weights - alpha * gradient


def mean_squared_error(y: np.ndarray, y_hat: np.ndarray) -> float:
	"""Return the mean squared difference between targets and predictions."""
	targets, predictions = _matching_vectors(y, y_hat)
	return float(np.mean((targets - predictions) ** 2))


def binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
	"""Return mean binary cross entropy for binary targets and probabilities."""
	targets, predictions = _matching_vectors(y, y_hat)
	if not np.all((targets == 0) | (targets == 1)):
		raise ValueError("y must contain only 0 and 1")
	if np.any((predictions < 0) | (predictions > 1)):
		raise ValueError("y_hat values must be probabilities between 0 and 1")

	epsilon = np.finfo(float).eps
	probabilities = np.clip(predictions, epsilon, 1.0 - epsilon)
	loss = -targets * np.log(probabilities) - (1.0 - targets) * np.log1p(-probabilities)
	return float(np.mean(loss))


def _matching_vectors(y: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	targets = np.asarray(y, dtype=float)
	predictions = np.asarray(y_hat, dtype=float)
	if targets.ndim != 1 or predictions.ndim != 1:
		raise ValueError("y and y_hat must be 1-D arrays")
	if targets.size == 0 or targets.shape != predictions.shape:
		raise ValueError("y and y_hat must have the same nonzero length")
	return targets, predictions
