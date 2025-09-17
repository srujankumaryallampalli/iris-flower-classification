#  Iris Flower Classification

A machine learning project to classify Iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on the dimensions of their petals and sepals. This project uses various classification algorithms and evaluates their performance to determine the most accurate model.

---

##  Dataset

- **Source**: [Kaggle - Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

- **Description**: The dataset contains 150 records under 5 attributes:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
  - Species (Target label)

---

##  Project Goals

- Develop classification models to identify Iris flower species.
- Perform exploratory data analysis (EDA) and visualize patterns.
- Identify key features that influence classification accuracy.
- Evaluate and compare multiple ML models using performance metrics.

---

##  Algorithms Used

This project compares the performance of five different classifiers:

| Model                   | Accuracy |
|------------------------|----------|
| Logistic Regression     | 95.56%   |
| K-Nearest Neighbors     | 95.56%   |
| Support Vector Machine  | 95.56%   |
| Decision Tree           | 93.33%   |
| Random Forest           | 95.56%   |

 **Logistic Regression** was selected as the final model due to its high accuracy and interpretability.

---

## 🗂 Project Structure

```
 iris-flower-classification/
├── iris_confusion_matrix_added.ipynb  # Main Jupyter notebook
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT license
└── .gitignore                         # Ignored files/folders
```

---

##  Results & Visualizations

The final model was evaluated using:
- Accuracy Score
- Confusion Matrix
- Feature Importance
- Model Comparison

Visualizations include:
- Pairplots of the dataset
- Confusion matrix heatmaps


---

##  Installation

###  Requirements

Install required Python packages:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

##  Run the Project

1. Clone this repository
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook iris_confusion_matrix_added.ipynb
   ```
3. Run all cells to train and evaluate the models

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Acknowledgements

- [Kaggle - Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- Scikit-learn and Matplotlib documentation

---

##  Future Improvements

- Hyperparameter tuning for better accuracy
- Integration with a simple web app (e.g., using Streamlit)
- Deploying the model for real-time predictions

---
