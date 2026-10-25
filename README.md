# Iris Species Classification using Logistic Regression

A machine learning mini project that predicts the species of an iris flower (**setosa**, **versicolor** or **virginica**) from four measurements, using logistic regression in scikit-learn. Built and run in Google Colab.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **96.67%** (58 of 60 test flowers correct) |
| Train / test split | 90 / 60 samples |

The classifier separates setosa perfectly. The only mistakes are between versicolor and virginica, which overlap slightly in petal size.

## Dataset

The classic [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset), loaded with `sklearn.datasets.load_iris()`.

- 150 samples, 50 per species (balanced)
- 4 features, all in cm: sepal length, sepal width, petal length, petal width
- 3 classes: setosa, versicolor, virginica

## Method

1. Load the dataset
2. Split into training and test sets (`test_size=0.4`, `random_state=1`)
3. Train `LogisticRegression(max_iter=200)` on the training set
4. Predict on the test set and compute accuracy with `accuracy_score`
5. Predict species for new sample inputs and plot the prediction probabilities

## Visualizations

- **Predicted species on test data**: scatter plot of petal length vs petal width, coloured by predicted species
- **Prediction probabilities**: bar chart of the predicted class probabilities for new samples

## Sample prediction

```python
sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
preds = log_reg.predict(sample)
print([iris.target_names[p] for p in preds])   # ['virginica', 'virginica']
```

Both samples are classified as virginica, but with different confidence: sample 1 is about 55% virginica (and 42% setosa), while sample 2 is about 99.99% virginica.

**Note:** both samples are far outside the range of real iris measurements (for example a sepal length of 2 to 3 cm, while the dataset ranges from 4.3 to 7.9 cm). The model is extrapolating, so a high probability here does not mean a trustworthy prediction.

## Tech stack

- Python 3
- scikit-learn
- NumPy
- Matplotlib
- Google Colab

## How to run

**On Google Colab**

1. Open the notebook in Colab (click the *Open in Colab* badge on GitHub, or upload the `.ipynb` file).
2. Run all cells from top to bottom (`Runtime > Run all`).

**Locally**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install scikit-learn numpy matplotlib jupyter
jupyter notebook
```

Then open the notebook and run all cells.

## Project structure

```
.
├── model_iris.ipynb   # main notebook
└── README.md
```

## Possible improvements

- Add a confusion matrix and a per-class precision/recall report
- Use k-fold cross-validation instead of a single train/test split
- Compare with k-nearest neighbours, SVM and decision trees
- Apply feature scaling and tune the regularisation strength
- Warn when a new sample falls outside the training data range

## Author

**Aashi Srivastava**
B.Tech Computer Science Engineering, United College of Engineering and Research, Prayagraj



