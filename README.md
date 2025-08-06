# 🧠 Fake News Detector

A machine learning web application built with **Flask** that detects whether a news article is **real or fake**, using **Logistic Regression** and **TF-IDF Vectorization**.

![Fake News Detector Banner](https://img.shields.io/github/languages/top/aditya2824/fake-news-detector?style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/aditya2824/fake-news-detector?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/aditya2824/fake-news-detector?style=for-the-badge)
![GitHub license](https://img.shields.io/github/license/aditya2824/fake-news-detector?style=for-the-badge)

---

## 🚀 Features

- 🔍 Classifies news as **Real** or **Fake**
- 📊 98% model accuracy
- 🎙️ Supports **speech-to-text** input
- 🌙 **Dark mode toggle**
- 📈 Interactive analytics dashboard (optional)
- 💡 User-friendly design with Bootstrap 5

---

## 📂 Project Structure

fake-news-detector/
│
├── app.py # Flask web server
├── train_model.py # Model training script
├── model.pkl # Trained logistic regression model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Front-end HTML page
├── Fake.csv # Fake news dataset
├── True.csv # Real news dataset
└── README.md # You're reading it now!


---

## 🧠 How the Model Works

- Combines `title + text` as input
- Preprocesses the text using **TF-IDF Vectorization**
- Trained using **Logistic Regression**
- Achieved ~98% accuracy on the test set

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS (Bootstrap 5), JavaScript
- **Backend**: Python, Flask
- **ML**: Scikit-learn, Pandas, TfidfVectorizer
- **Visualization**: Chart.js (optional analytics)

---

## 🔧 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/aditya2824/fake-news-detector.git
   cd fake-news-detector
   
🧠 Skills Demonstrated
✅ Python
✅ Flask
✅ Machine Learning
✅ Scikit-learn
✅ Data Cleaning & Preprocessing
✅ Frontend Integration
✅ Git & GitHub


---

✅ Once pasted, commit it:
```bash
git add README.md
git commit -m "Added professional README"
git push

