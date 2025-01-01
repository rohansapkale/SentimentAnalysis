
```markdown
# Sentiment Analysis of Tweets

This repository showcases a Machine Learning pipeline for sentiment analysis of tweets. The project processes text data, extracts features, trains a classification model, and provides an interactive interface for real-time sentiment prediction.

---

## 🎯 **Project Overview**

This project predicts whether a tweet expresses a **positive** or **negative** sentiment. It uses the **Sentiment140** dataset, applying Natural Language Processing (NLP) and Logistic Regression for sentiment classification.

---

## 🛠️ **Features**

- **Data Preprocessing**:
  - Text cleaning to remove non-alphabetic characters.
  - Removal of stopwords and stemming to simplify text data.
- **Feature Extraction**:
  - Text data is converted into numerical format using **TF-IDF Vectorization**.
- **Model Training**:
  - Trained a **Logistic Regression** model for sentiment classification.
- **Interactive Predictions**:
  - Real-time sentiment prediction using a simple GUI.
- **Model Saving**:
  - The trained model is saved using **Pickle** for future use.

---

## 📂 **Dataset**

The dataset used is [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140), which contains 1.6 million labeled tweets.

---

## 🛠️ **Libraries Used**

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **NLTK**: For text preprocessing (stopwords removal, stemming).
- **Scikit-learn**:
  - **TF-IDF Vectorizer**: For feature extraction.
  - **Logistic Regression**: For classification.
  - **Train-Test Split**: For splitting the data.
  - **Accuracy Metrics**: For model evaluation.
- **Pickle**: For saving and loading the trained model.
- **Ipywidgets**: For creating an interactive interface in Jupyter notebooks.

---

## ⚙️ **Setup Instructions**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-tweets.git
   cd sentiment-analysis-tweets
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset:
   - Place the `sentiment140.zip` file in the root folder.
4. Unzip the dataset:
   ```python
   from zipfile import ZipFile
   with ZipFile("sentiment140.zip", "r") as zip:
       zip.extractall()
   ```

---

## 🚀 **How to Run**

1. Open the notebook `SentimentAnalysis.ipynb` in Jupyter/Colab.
2. Follow the steps for data preprocessing, model training, and evaluation.
3. Use the interactive widgets to input custom tweets and analyze their sentiment.

---

## 📊 **Model Performance**

- **Training Accuracy**: ~78%
- **Testing Accuracy**: ~77.8%

---

## 📫 **Contributing**

Feel free to open issues or submit pull requests for any suggestions or improvements.

---

## 📜 **License**

This project is licensed under the MIT License.
```
