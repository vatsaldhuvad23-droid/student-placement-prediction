# student-placement-prediction
Machine Learning project to predict student placement status using Logistic Regression.

🎓 Student Placement Prediction using Machine Learning

📌 Project Overview

This project predicts whether a student will be **Placed** or **Not Placed** based on academic performance, attendance, study habits, and other factors.

The objective of this project is to demonstrate a complete Machine Learning workflow, from data loading and preprocessing to model training, evaluation, and prediction.

This is my first end-to-end Machine Learning project built using Python, Pandas, NumPy, Scikit-Learn, and Matplotlib.

🚀 Project Workflow

### Step 1: Load and Understand Data
- Load dataset using Pandas
- View sample records
- Check dataset shape
- Analyze data types
- Generate statistical summary
- Check missing values

### Step 2: Data Preprocessing
- Handle missing values
- Encode categorical data
- Convert target variable into numerical format
- Verify data types

### Step 3: Feature Scaling
- Apply StandardScaler to numerical features
- Normalize data for better model performance

### Step 4: Train-Test Split
- Split dataset into training and testing sets
- 80% Training Data
- 20% Testing Data

### Step 5: Model Training
- Train Logistic Regression model

### Step 6: Model Evaluation
- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Step 7: Prediction System
- Accept user input
- Predict whether a student will be Placed or Not Placed

## 📊 Features Used

The model uses the following features:

- Study Hours
- Attendance
- Sleep Hours
- Internet Usage
- Assignments Completed
- Previous Score
- Exam Score

### Target Variable
- Placement Status

🤖 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is used because:
- Suitable for binary classification problems
- Fast and efficient
- Easy to understand and implement
- Provides strong baseline performance

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

📂 Project Structure

```text
student-placement-prediction/
│
├── student_placement_prediction.ipynb
├── student_dataset_10000_rows.csv
├── README.md
└── requirements.txt
```

-⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/student-placement-prediction.git
```

Move into the project directory:

```bash
cd student-placement-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

📦 Requirements

```txt
pandas
numpy
matplotlib
scikit-learn
jupyter
```

## 📈 Model Evaluation Metrics

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics help measure the performance and reliability of the classification model.

## 🎯 Learning Outcomes

Through this project, I learned:

- Data Loading and Exploration
- Data Preprocessing
- Label Encoding
- Feature Scaling
- Logistic Regression
- Train-Test Splitting
- Model Evaluation
- Confusion Matrix Visualization
- Building an End-to-End Machine Learning Workflow

## 🔮 Future Improvements

- Add multiple machine learning algorithms
- Perform hyperparameter tuning
- Apply cross-validation
- Build a Streamlit web application
- Deploy the model online
- Improve model accuracy

## 👨‍💻 Author

**Vatsal Dhuvad**

Aspiring Data Scientist and Machine Learning Enthusiast

GitHub: https://github.com/vatsaldhuvad23-droid

LinkedIn: https://www.linkedin.com/in/vatsal-dhuvad-7630482b2/

--------------------------------------------------------------

⭐ If you found this project useful, please consider giving it a star.
