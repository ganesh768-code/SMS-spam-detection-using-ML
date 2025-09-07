# 📱 SMS Spam Detection using Machine Learning + Flask

## 📌 Project Overview
This project builds a machine learning model to classify SMS messages as **Spam** or **Not Spam**.  
It uses natural language processing (NLP) techniques to clean and transform text data, followed by machine learning algorithms for classification.  
Additionally, a **Flask web app** was created to allow users to input SMS text and get instant predictions.

---

## 📂 Dataset
- The dataset contains SMS text messages labeled as "ham" (not spam) or "spam".
- Public datasets such as the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) were used.

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn
- **web technology:** HTML, CSS
- **Web Framework:** Flask
- **Environment:** Jupyter Notebook / VS Code  

---

## 🚀 Steps Involved
1. **Data Preprocessing**
   - Remove punctuation, stopwords, and special characters
   - Tokenization and lemmatization using NLTK
   - Convert text into numeric features using Bag of Words and TF-IDF  

2. **Model Building**
   - Applied ML algorithms: Naive Bayes, Logistic Regression, SVM
   - Compared performance using accuracy, precision, recall, F1-score  

3. **Flask Web App**
   - Built a simple web interface where users can enter SMS text
   - Integrated trained ML model for real-time predictions
   - Designed minimal HTML templates for input/output  

---

## 📊 Results
- The **Naive Bayes model** gave the best performance (~95% accuracy).  
- Flask app successfully predicts spam vs. ham in real-time.  

---

## 📌 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/sms-spam-detection-ml.git
cd sms-spam-detection-ml
