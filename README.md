# Machine Learning Models for Early Detection of Separation Anxiety Disorder (SAD)

This repository contains the implementation of machine learning models developed to detect **Separation Anxiety Disorder (SAD)** in preschool children using behavioral, psychophysiological, and demographic features. The models were evaluated on a real-world dataset and aim to support early diagnosis through explainable and accurate classification methods.

## 📌 Project Objectives

- To build ML classifiers for early detection of SAD in children aged 3–5.
- To compare performance of **Gradient Boosting**, **MLP**, **Extra Trees**, and **SVC** classifiers.
- To apply automated **feature selection** using Pearson correlation.
- To evaluate models using metrics including accuracy, precision, recall, F1-score, AUC, MCC, and Cohen’s Kappa.

## 📂 Files

- `gradientboostingmlpanksiyetesi-bozuklugu.ipynb`: Main Jupyter Notebook implementing the classification pipeline.
- (Optional) `data_description.txt`: Overview of dataset features and structure (not included here).
- (Optional) `figures/`: Plots of confusion matrices, ROC curves, correlation heatmaps.

## 📊 Models Used

- **Gradient Boosting Classifier**
- **Multi-Layer Perceptron (MLP)**
- **Extra Trees Classifier**
- **Support Vector Classifier (SVC)**

## 🧪 Dataset

- Dataset includes 917 instances with 79 features per subject.
- Features include:
  - Behavioral symptoms
  - Physiological indicators (e.g., heart rate)
  - Demographic attributes (e.g., age, gender)
- Original dataset was approved by [Duke IRB] and accessed via [Harvard Dataverse](https://doi.org/10.7910/DVN/N42LWG).

> ⚠️ **Note:** Raw data is not shared in this repository due to ethical and privacy reasons.

## 📈 Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy (ACC)
- Precision (PRE)
- Recall / Sensitivity
- Specificity
- F1-Score
- Area Under the ROC Curve (AUC)
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa Score

## 🛠️ Requirements

To run the notebook:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
python==3.10.10  
scikit-learn  
tensorflow==2.12  
matplotlib  
pandas  
numpy
```

## 📄 Citation

If you use this code for academic research, please cite the following (example):

```
Çelebi, B., & [Your Name]. (2025). Machine Learning for Early Detection of Separation Anxiety Disorder in Preschool Children. [Manuscript in preparation].
```

## 📬 Contact

For questions or collaborations, please contact [your-email@example.com].
