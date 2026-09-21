import pandas as pd

DATASET_FILE = 'credit_risk_dataset.csv'


def calculate_income_by_age(file_path : str) -> None:
	'''
	Calculates and prints income statistics grouped by person age.
	:param file_path: str
	This parameter is used for the dataset path.
	:return:
	None.
	'''
	credit_risk_data = pd.read_csv(file_path)
	print(credit_risk_data.groupby(['person_age'])['person_income'].agg(['max', 'min', 'mean']))


if __name__ == '__main__':
	calculate_income_by_age(DATASET_FILE)