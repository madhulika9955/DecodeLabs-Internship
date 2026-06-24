# DecodeLabs-Internship

# Task 1

# House Price Prediction

## Project Overview

This project aims to predict house prices using Machine Learning techniques based on various property features such as area, number of bedrooms, bathrooms, stories, parking spaces, furnishing status, and preferred location. The objective is to build predictive models, evaluate their performance, and identify the factors that have the greatest impact on house prices.

## Dataset

* Dataset Name: Housing Prices Dataset
* Source: Kaggle
* File Used: `Housing.csv`

## Project Tasks

### 1. Data Loading & Exploration

* Loaded the dataset using Pandas.
* Examined dataset structure and dimensions.
* Identified target and feature variables.
* Checked for missing values and data types.

### 2. Data Cleaning

* Removed duplicate records.
* Handled missing values.
* Converted categorical variables into numerical format using one-hot encoding.
* Prepared the dataset for model training.

### 3. Model Building

Two regression models were trained and evaluated:

#### Linear Regression

* MAE: 970,043
* RMSE: 1,324,507
* R² Score: 0.653

#### Random Forest Regressor

* MAE: 1,021,546
* RMSE: 1,400,566
* R² Score: 0.612

### 4. Data Visualization

The following visualizations were created:

* House Price Distribution Histogram
* Correlation Heatmap
* Actual vs Predicted Price Scatter Plot

All chart images are available in the `charts/` folder.

## Results

The Linear Regression model performed better than the Random Forest model on this dataset, achieving a higher R² score and lower prediction error. The model was able to explain approximately 65% of the variation in house prices.

## Key Insights

* Area is one of the strongest factors affecting house price.
* Houses with more bathrooms, parking spaces, and stories generally have higher prices.
* Preferred location significantly increases property value.
* Furnishing status and air conditioning also contribute to pricing.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook (VS Code)

## Project Structure

```text
HousePricePrediction_MadhulikaSingh/
│
├── analysis.ipynb
├── Housing.csv
├── summary.pdf
├── README.md
│
└── charts/
    ├── price_distribution.png
    ├── correlation_heatmap.png
    └── actual_vs_predicted.png
```

## Conclusion

This project demonstrates how machine learning can be used to estimate house prices based on property characteristics. The findings can help buyers, sellers, and real estate businesses make more informed decisions regarding property valuation.



# Task 2

# Fraud Detection Pipeline using Machine Learning

## Project Overview

This project aims to detect fraudulent credit card transactions using Machine Learning techniques. Since fraud transactions represent a very small percentage of all transactions, the dataset is highly imbalanced. To address this challenge, SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset before training classification models.

The project compares multiple machine learning algorithms and evaluates them using Precision, Recall, F1-Score, and ROC-AUC instead of Accuracy, which can be misleading for imbalanced datasets.

---

## Dataset

Dataset Used: Credit Card Fraud Detection Dataset

* Total Transactions: 284,807
* Fraud Transactions: 492
* Genuine Transactions: 284,315
* Fraud Percentage: 0.172%

Target Variable:

* Class = 0 → Genuine Transaction
* Class = 1 → Fraudulent Transaction

Dataset Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Project Workflow

1. Data Collection
2. Data Exploration and Analysis (EDA)
3. Data Cleaning
4. Train-Test Split
5. Handling Class Imbalance using SMOTE
6. Logistic Regression Model Training
7. Random Forest Model Training
8. Hyperparameter Tuning using GridSearchCV
9. Model Evaluation
10. Best Model Selection
11. Model Saving using Joblib

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Joblib

---

## Machine Learning Models

### Logistic Regression

Used as a baseline classification model to identify fraudulent transactions.

### Random Forest Classifier

An ensemble learning method used to improve prediction performance and handle complex patterns in the dataset.

---

## Evaluation Metrics

Since the dataset is highly imbalanced, model performance is evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

Accuracy is not considered a reliable metric for this project.

---

## Results

The models were trained on a balanced dataset generated using SMOTE.

Performance comparison was carried out between:

* Logistic Regression
* Random Forest Classifier

The best-performing model was selected based on Recall and ROC-AUC Score.

---

## Project Structure

```text
Fraud_Detection_Project/
│
├── creditcard.csv
├── fraud_detection.ipynb
├── fraud_model.pkl
├── README.md
│
├── graphs/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── reports/
    └── project_report.pdf
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Fraud-Detection-Pipeline.git
```

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

Run the notebook:

```bash
jupyter notebook
```

---

## Model Saving

The trained model is saved using Joblib:

```python
joblib.dump(best_model, "fraud_model.pkl")
```

---

## Future Improvements

* XGBoost Implementation
* LightGBM Implementation
* Deep Learning Approaches
* Real-Time Fraud Detection System
* Deployment using Flask or Streamlit

# Task 3

# Customer Segmentation using K-Means Clustering and PCA

##  Project Overview

This project focuses on **Customer Segmentation** using **Unsupervised Machine Learning** techniques. The goal is to identify hidden customer groups based on their purchasing behavior and demographic information. By applying **Principal Component Analysis (PCA)** and **K-Means Clustering**, customers are segmented into meaningful groups that can be used for targeted marketing and business decision-making.

This project was completed as part of the **DecodeLabs Data Science Industrial Training Program (Project 3)**.

---

##  Objectives

- Perform data preprocessing and feature engineering.
- Standardize customer data for clustering.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Determine the optimal number of clusters using:
  - Elbow Method
  - Silhouette Score
- Build a K-Means clustering model.
- Visualize customer segments.
- Generate actionable customer personas and business insights.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Jupyter Notebook
- VS Code

---

##  Project Structure

```text
Customer-Segmentation-Project/
│
├── data/
│   └── Mall_Customers.csv
│
├── images/
│   ├── elbow_curve.png
│   ├── silhouette_curve.png
│   └── customer_clusters.png
│
├── models/
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── kmeans.pkl
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── Customer_Segmentation_Output.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Dataset

The dataset contains customer demographic and spending information.

### Features

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

Dataset Source:

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

##  Workflow

### 1. Data Collection
- Load customer dataset.
- Explore dataset structure and statistics.

### 2. Data Preprocessing
- Handle missing values.
- Encode categorical variables.
- Select relevant features.

### 3. Feature Scaling
- Standardize data using StandardScaler.

### 4. Dimensionality Reduction
- Apply PCA.
- Retain 95% of variance.

### 5. Optimal Cluster Selection
- Elbow Method
- Silhouette Score Analysis

### 6. K-Means Clustering
- Train K-Means model.
- Assign cluster labels to customers.

### 7. Data Visualization
- PCA 2D Scatter Plot
- Cluster Distribution Charts
- Elbow Curve
- Silhouette Score Curve

### 8. Customer Persona Generation
- Analyze cluster characteristics.
- Create business-friendly customer segments.

---

##  Results

The K-Means algorithm successfully grouped customers into distinct segments based on income, age, and spending behavior.

Example Customer Personas:

### Cluster 0 – Conservative Wealthy Customers
- High Income
- Low Spending
- Prefer savings over luxury purchases

### Cluster 1 – Premium Customers
- High Income
- High Spending
- Most valuable customers

### Cluster 2 – Young Trend Shoppers
- Low Income
- High Spending
- Influenced by trends and promotions

### Cluster 3 – Budget Customers
- Low Income
- Low Spending
- Price-sensitive customers

### Cluster 4 – Regular Customers
- Moderate Income
- Moderate Spending
- Stable customer base

---

##  Sample Visualizations

### Elbow Method
Determines the optimal number of clusters.

### Silhouette Score
Measures clustering quality.

### PCA Cluster Visualization
Displays customer groups in reduced dimensions.

---

##  How to Run the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/customer-segmentation-project.git
```

### Navigate to Project Folder

```bash
cd customer-segmentation-project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/customer_segmentation.ipynb
```

---

##  Required Libraries

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

##  Business Applications

- Customer Targeting
- Personalized Marketing
- Customer Retention
- Loyalty Programs
- Product Recommendations
- Revenue Optimization

---

##  Learning Outcomes

Through this project, I gained hands-on experience with:

- Unsupervised Learning
- Customer Segmentation
- Principal Component Analysis (PCA)
- K-Means Clustering
- Data Visualization
- Business Intelligence
- Model Evaluation

---

## Task 4

# NLP & Sentiment Analysis using Machine Learning

##  Project Overview

This project focuses on **Natural Language Processing (NLP)** and **Sentiment Analysis**. The objective is to build a machine learning model that can analyze textual reviews and classify them as **Positive** or **Negative**.

The project demonstrates the complete NLP workflow, including text preprocessing, feature extraction using TF-IDF, model training, evaluation, and prediction on new reviews.

---

##  Objectives

* Clean and preprocess unstructured text data.
* Perform tokenization, stop-word removal, and lemmatization.
* Convert text into numerical features using TF-IDF.
* Train a sentiment classification model.
* Evaluate model performance using various metrics.
* Predict sentiment for unseen reviews.

---

##  Dataset

Dataset Used: **IMDB Movie Reviews Dataset**

* Total Reviews: 50,000
* Classes:

  * Positive
  * Negative

Dataset Link:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Matplotlib
* Seaborn
* WordCloud
* Joblib

---

##  Project Workflow

### 1. Data Collection

* Download IMDB Reviews Dataset from Kaggle.
* Load dataset using Pandas.

### 2. Data Preprocessing

* Convert text to lowercase.
* Remove special characters and numbers.
* Tokenization.
* Stop-word removal.
* Lemmatization.

### 3. Feature Engineering

* Apply TF-IDF Vectorization.
* Convert text into numerical vectors.

### 4. Model Building

* Train Multinomial Naive Bayes Classifier.

### 5. Model Evaluation

* Accuracy Score
* Classification Report
* Confusion Matrix

### 6. Prediction

* Predict sentiment of custom reviews.

---

##  Project Structure

```text
NLP-Sentiment-Analysis/
│
├── data/
│   └── IMDB Dataset.csv
│
├── notebook/
│   └── NLP_Sentiment_Analysis.ipynb
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── reports/
│   └── project_report.pdf
│
├── screenshots/
│   ├── dataset_head.png
│   ├── preprocessing.png
│   ├── confusion_matrix.png
│   └── wordcloud.png
│
├── README.md
│
└── requirements.txt
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/NLP-Sentiment-Analysis.git
```

Move into the project directory:

```bash
cd NLP-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Run the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run:

```text
NLP_Sentiment_Analysis.ipynb
```

---

##  Results

Model Used:

* Multinomial Naive Bayes

Performance:

* Accuracy: ~85% to 90%

Evaluation Metrics:

* Precision
* Recall
* F1-Score
* Confusion Matrix

---

##  Sample Prediction

Input Review:

```text
This movie was absolutely fantastic and enjoyable.
```

Output:

```text
Positive Review
```

Input Review:

```text
The movie was boring and a complete waste of time.
```

Output:

```text
Negative Review
```

---

##  Screenshots

Include the following screenshots in the repository:

* Dataset Overview
* Cleaned Reviews
* TF-IDF Features
* Confusion Matrix
* Word Cloud
* Prediction Results

---

##  Future Enhancements

* Deploy model using Flask or Streamlit.
* Use advanced NLP techniques.
* Train using SVM and compare performance.
* Add real-time review analysis.
* Integrate deep learning models such as LSTM and BERT.

---

## Author

Madhulika Singh

B.Tech CSE (AI)

ABES Institute of Technology

GitHub: https://github.com/madhulika9955

LinkedIn: https://www.linkedin.com/in/madhulika-singh-a34b9a28b/

---

## License

This project is developed for educational and internship purposes.
