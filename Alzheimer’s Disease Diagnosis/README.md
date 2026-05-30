# Alzheimer’s Disease Diagnosis Using Machine Learning

A machine learning project for predicting Alzheimer’s Disease diagnosis from patient-related clinical, lifestyle, cognitive, and health features. The project compares two supervised classification algorithms, **Decision Tree** and **K-Nearest Neighbors (KNN)**, and evaluates their performance using standard classification metrics.

## Project Overview

The goal of this project is to explore how machine learning can support early Alzheimer’s Disease diagnosis by classifying patient data into:

- **No Alzheimer**
- **Alzheimer**

The project includes data loading, exploration, preprocessing, model training, hyperparameter tuning, evaluation, and comparison across different train/test splits.

> **Note:** This project is for educational and experimental purposes only. It is not intended for real clinical diagnosis without validation by medical professionals and larger real-world datasets.

## Team

**Team Name:** Finding Memo

**Team Members:**
- Athanasios Karletsos
- Aristea Patrikaki

## Dataset

The dataset is downloaded from Kaggle using `kagglehub`:

```python
kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")
```

The dataset contains patient information such as:

- Demographic features
- Lifestyle and health-related features
- Cognitive test scores
- Functional assessment features
- Alzheimer’s diagnosis label

Some example columns include:

- `Age`
- `Gender`
- `Ethnicity`
- `EducationLevel`
- `BMI`
- `Smoking`
- `AlcoholConsumption`
- `PhysicalActivity`
- `SleepQuality`
- `FamilyHistoryAlzheimers`
- `CardiovascularDisease`
- `Diabetes`
- `Depression`
- `Hypertension`
- `MMSE`
- `FunctionalAssessment`
- `MemoryComplaints`
- `BehavioralProblems`
- `ADL`
- `Diagnosis`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub
- Jupyter Notebook

## Machine Learning Models

### 1. Decision Tree Classifier

A Decision Tree is a supervised machine learning model that classifies data by creating a tree-like structure of decision rules. It is easy to interpret and useful for understanding which features influence the prediction.

In this project, the Decision Tree model was trained and later fine-tuned using `GridSearchCV`.

Fine-tuned parameters included:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `criterion`

The best fine-tuned Decision Tree model used:

```python
criterion = "entropy"
max_depth = 5
min_samples_split = 10
min_samples_leaf = 1
random_state = 42
```

### 2. K-Nearest Neighbors (KNN)

KNN is an instance-based classifier that predicts the class of a new sample based on the majority class of its nearest neighbors.

Since KNN is sensitive to feature scale, the features were standardized using `StandardScaler`.

The project tested different values of `k` from 1 to 19. The final KNN model used a tuned value around `k = 15` to `k = 17`, depending on the experiment.

Fine-tuned KNN parameters included:

```python
n_neighbors = 15
weights = "uniform"
metric = "euclidean"
algorithm = "kd_tree"
```

## Methodology

The project follows these main steps:

1. Import required libraries
2. Download and load the dataset
3. Explore the dataset
4. Check for missing values
5. Drop unnecessary columns
6. Split the data into training and testing sets
7. Scale features for KNN
8. Train Decision Tree and KNN models
9. Tune hyperparameters
10. Evaluate the models
11. Compare performance using multiple train/test splits
12. Visualize confusion matrices and Decision Tree structure

## Preprocessing

The preprocessing stage included:

- Checking for missing values
- Removing unnecessary columns such as identifiers or non-predictive fields
- Splitting the dataset into training and testing sets
- Applying feature scaling for KNN

The dataset did not require major missing-value handling because no missing/null values were detected.

## Evaluation Metrics

The models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics were used because accuracy alone is not enough for medical-related classification problems. In Alzheimer’s Disease detection, recall is especially important because false negatives can mean missing patients who may actually have the disease.

## Results

### Decision Tree

The original Decision Tree achieved approximately:

| Metric | Score |
|---|---:|
| Accuracy | 0.9023 |
| Precision | 0.8725 |
| Recall | 0.8497 |
| F1-score | 0.8609 |

After hyperparameter tuning, the Decision Tree improved to approximately:

| Metric | Score |
|---|---:|
| Accuracy | 0.9488 |
| Precision | 0.9396 |
| Recall | 0.9150 |
| F1-score | 0.9272 |

### KNN

The KNN model achieved approximately:

| Metric | Score |
|---|---:|
| Accuracy | 0.8581 |
| Precision | 0.8239 |
| Recall | 0.7647 |
| F1-score | 0.7932 |

The fine-tuned KNN model produced very similar results, indicating that the initially selected value of `k` was already close to optimal.

## Multiple Train/Test Split Comparison

The models were also tested using multiple train/test splits:

| Split | KNN Accuracy | KNN F1 | Decision Tree Accuracy | Decision Tree F1 |
|---|---:|---:|---:|---:|
| 60/40 | 0.8337 | 0.7572 | 0.9349 | 0.9082 |
| 70/30 | 0.8326 | 0.7632 | 0.9256 | 0.8974 |
| 67/33 | 0.8338 | 0.7640 | 0.9282 | 0.8998 |
| 80/20 | 0.8581 | 0.7932 | 0.9279 | 0.8956 |
| 82/18 | 0.8579 | 0.7985 | 0.9302 | 0.9018 |

Overall, the Decision Tree model achieved higher performance than KNN in this project, while also offering better interpretability.

## Key Findings

- Decision Tree performed better than KNN in most evaluation metrics.
- Fine-tuning significantly improved the Decision Tree model.
- KNN required feature scaling to perform properly.
- Testing different train/test splits helped evaluate model stability.
- Recall is very important in this type of healthcare classification problem because false negatives should be minimized.
- The models provide a strong educational baseline, but more validation would be needed before any real-world medical use.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/patrickaria/Machine-Learning.git
cd "Machine-Learning/Alzheimer’s Disease Diagnosis"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
```

### 4. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
ML_FINAL_PROJECT_KARLETSOS_PATRIKAKI.ipynb
```

### 5. Run all cells

Run the notebook from top to bottom to download the dataset, train the models, and view the results.


## Future Improvements

Possible improvements for this project include:

- Testing more machine learning models such as Random Forest, SVM, Logistic Regression, and XGBoost
- Applying cross-validation more extensively
- Performing deeper feature importance analysis
- Handling possible class imbalance
- Testing the model on larger and more diverse real-world datasets
- Using more advanced explainability tools such as SHAP or LIME
- Saving the trained model using `joblib` or `pickle`
- Building a simple web app interface for predictions

## Conclusion

This project demonstrates how machine learning can be applied to healthcare-related data for Alzheimer’s Disease prediction. The Decision Tree model achieved the best overall performance and provided interpretability, while KNN offered a simple and useful baseline. Although the results are promising, the project should be considered an educational machine learning application rather than a clinical diagnostic tool.
