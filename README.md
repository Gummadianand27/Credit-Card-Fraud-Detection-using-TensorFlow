
# Credit Card Fraud Detection

This project is focused on detecting fraudulent transactions using machine learning techniques. The dataset used contains anonymized credit card transactions labeled as either fraudulent or genuine.

## Dataset

- **Source**: [Kaggle](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)
- **Description**: The dataset contains transactions made by European cardholders in September 2013. The dataset includes 284,807 transactions, of which 492 are fraudulent. Due to the highly imbalanced nature of the data, special care is needed in evaluating model performance.

## Project Overview

The main objective of this project is to develop a model that can effectively identify fraudulent credit card transactions. The steps involved in this project include:

1. **Data Loading**: Loading the raw data from a CSV file.
2. **Data Preprocessing**: Handling missing values, scaling features, and splitting the data into training and testing sets.
3. **Modeling**: Building a machine learning model using TensorFlow/Keras.
4. **Evaluation**: Evaluating the model performance using metrics like Precision-Recall Curve and Confusion Matrix.
5. **Visualization**: Visualizing the data distribution, model performance, and important features.

## Installation

To run this project, you need the following libraries:

- Python 3.x
- NumPy
- Pandas
- Seaborn
- Matplotlib
- TensorFlow

You can install the required libraries using pip:

```bash
pip install numpy pandas seaborn matplotlib tensorflow
```

## Usage

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Open the Jupyter notebook:

```bash
jupyter notebook credit-card-fraud-detection-tensorflow.ipynb
```

Follow the steps in the notebook to load the data, train the model, and evaluate its performance.

## Results

The model's performance is evaluated using various metrics, with a particular focus on the Area Under the Precision-Recall Curve (AUPRC) due to the imbalanced nature of the data.

## Contributing

Feel free to fork the repository and make contributions. Pull requests are welcome.


## Acknowledgments

- The dataset was sourced from Kaggle.

