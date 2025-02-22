# Music Genre Prediction Project

This project is a simple machine learning application that predicts the music genre preference of a user based on their age and gender. It uses a **Decision Tree Classifier** to make predictions and is designed to help beginners understand the basics of machine learning workflows, including data preprocessing, model training, evaluation, and deployment.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [About Decision Trees](#about-decision-trees)
3. [Steps in the Project](#steps-in-the-project)
4. [Evaluation Methods](#evaluation-methods)
5. [Bug Fixes and Improvements](#bug-fixes-and-improvements)
6. [What I Learned](#what-i-learned)
7. [How to Run the Project](#how-to-run-the-project)
8. [Future Improvements](#future-improvements)
9. [Conclusion](#conclusion)

---

## Project Overview
The goal of this project is to build a machine learning model that predicts the music genre a user might like based on their age and gender. The model is trained using a **Decision Tree Classifier**, and the dataset contains information about users' age, gender, and their preferred music genre.

---

## About Decision Trees
A **Decision Tree** is a supervised machine learning algorithm used for both classification and regression tasks. It works by splitting the data into subsets based on the value of input features. Each split represents a decision node, and the final predictions are made at the leaf nodes. Decision Trees are easy to interpret and visualize, making them a great choice for beginners.

---

## Steps in the Project

### 1. **Load the Dataset**
The dataset (`music.csv`) is loaded using `pandas`. It contains the following columns:
- **age**: The age of the user.
- **gender**: The gender of the user (0 for female, 1 for male).
- **genre**: The music genre preferred by the user.

### 2. **Prepare Input and Output**
- **Input (X)**: Features like `age` and `gender`.
- **Output (Y)**: Target variable `genre`.

### 3. **Split Data into Training and Testing Sets**
The dataset is split into training and testing sets using `train_test_split`. This ensures that the model is evaluated on unseen data.

### 4. **Train the Model**
A **Decision Tree Classifier** is trained on the training data. The model is configured with `max_depth=3` and `min_samples_split=5` to prevent overfitting.

### 5. **Save the Trained Model**
The trained model is saved using `joblib` so that it can be reused without retraining.

### 6. **Evaluate the Model**
The model's performance is evaluated on the test set using **accuracy**. Additionally, **cross-validation** is performed to ensure the model's robustness.

### 7. **Predict Music Preference for a New User**
The trained model is used to predict the music genre for a new user (e.g., a 21-year-old male).

---

## Evaluation Methods

### 1. **Accuracy on Test Set**
The model's accuracy is calculated by comparing its predictions on the test set with the actual labels. In this project, the model achieved an accuracy of **50.00%**.

### 2. **Cross-Validation**
Cross-validation is performed using 3 folds to evaluate the model's performance on different subsets of the data. The average cross-validation accuracy is **61.11%**.

### 3. **Bug Fixes and Improvements**
During the development of this project, several issues were identified and resolved:
- **Low Accuracy**: The model's accuracy was initially low due to limited features and a small dataset. Adding more features or collecting more data could improve performance.
- **Unbalanced Data**: Some music genres had very few samples, leading to biased predictions. Techniques like oversampling or undersampling could help balance the dataset.
- **Overfitting**: The model was prone to overfitting due to its simplicity. Adjusting hyperparameters like `max_depth` and `min_samples_split` helped mitigate this issue.

---

## What I Learned

### 1. **Data Preprocessing**
I learned how to load and preprocess data, including splitting it into training and testing sets.

### 2. **Model Training**
I gained hands-on experience in training a machine learning model using a Decision Tree Classifier.

### 3. **Model Evaluation**
I understood the importance of evaluating a model using both test set accuracy and cross-validation.

### 4. **Bug Fixing**
I learned how to identify and fix common issues in machine learning projects, such as low accuracy, overfitting, and unbalanced data.

### 5. **Model Deployment**
I explored how to save and load a trained model using `joblib`, making it reusable for future predictions.

---

## How to Run the Project

### 1. **Clone the Repository**
To get started, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/fatmhanafow/Music-genre-prediction-ML-project-.git
```

### 2. **Navigate to the Project Directory**
```bash
cd Music-genre-prediction-ML-project-
```
### 3. **Install Required Libraries**
Install the necessary Python libraries by running:
```bash
pip install -r requirements.txt
```
If you don't have a requirements.txt file, you can install the libraries manually:
```bash
pip install pandas scikit-learn joblib jupyter
```


### 4. **Run the Jupyter Notebook**
Start the Jupyter Notebook to explore and run the project:
```bash
jupyter notebook music-genre-prediction.ipynb
```

## Supplementary Materials
For a detailed walkthrough of this project, check out the following video:
https://youtu.be/7eh4d6sabA0?si=IogQ15JXR8ZyegFo

## GitHub Repository
You can find additional Python materials and resources in the following GitHub repository:
https://github.com/mosh-hamedani/python-supplementary-materials.git

## Future Improvements

### 1. **Collect More Data:**
Increase the size of the dataset to improve model performance.

### 2. **Add More Features:**
Include additional features like favorite_artist or mood to enhance predictions.

### 3. **Try Different Models:**
Experiment with other machine learning algorithms like Random Forest or SVM.

### 4. **Balance the Dataset:**
Use techniques like oversampling or undersampling to handle imbalanced data.

### 5. **Hyperparameter Tuning:**
Optimize the model's hyperparameters using grid search or random search.

## Conclusion
This project was a great introduction to machine learning. I learned how to preprocess data, train a model, evaluate its performance, and deploy it for predictions. Although the model's accuracy is currently low, this project provided valuable insights into the challenges of machine learning and the steps needed to improve a model's performance. I look forward to applying these lessons to more complex projects in the future!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

