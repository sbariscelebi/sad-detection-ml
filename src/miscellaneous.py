#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# In[ ]:





# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:



sad_corr_df = pd.DataFrame({'Feature': sad_corr.index, 'Correlation_with_SAD': sad_corr.values})

sad_corr_df['Abs_Correlation'] = sad_corr_df['Correlation_with_SAD'].abs()
sad_corr_df = sad_corr_df.sort_values(by='Abs_Correlation', ascending=False)

threshold = sad_corr_df['Abs_Correlation'].quantile(0.75)

high_corr_features = sad_corr_df[sad_corr_df['Abs_Correlation'] > threshold]['Feature'].tolist()

if 'GAD' in high_corr_features:
    high_corr_features.remove('GAD')

df_selected = df[high_corr_features]


print(f"Threshold (75th percentile): {threshold}")


# In[ ]:


feature_mapping = {
    "Avoidance of being alone": "ADBA",
    "Anticipatory distress/resistance to separation": "ADRA",
    "Actual distress when attachment figure absent": "ADAA",
    "Fear about possible harm befalling major attachment figures": "FPHB",
    "Fear/anxiety about daycare/school attendance screen positive": "FAAS",
    "Onset Fear about calamitous separation": "OFCS",
    "Parent's plan disrupted due to child's distress at separation": "PPDS",
    "Frequency of reluctance to go to sleep": "FRGS",
    "Anticipatory fear of daycare/school": "AFDS",
    "Fear/Anxiety about leaving home for daycare/school": "FAAD",
    "Anxious affect that occurs in certain situations/environments": "AAOS",
    "Rising at night to check on family members": "RNCM",
    "SAD": "SAD"
}

available_features = [col for col in feature_mapping.keys() if col in df.columns]

df_selected = df[available_features].rename(columns={col: feature_mapping[col] for col in available_features})

# Korelasyon matrisini hesapla
corr_matrix = df_selected.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    linewidths=0.5, 
    annot_kws={"size": 10, "color": "black", "weight": "bold"}
)

plt.title('Correlation Matrix', fontsize=20, color='black')
plt.xticks(rotation=45, ha='right', fontsize=12, color='black')
plt.yticks(rotation=0, fontsize=12, color='black')

plt.tight_layout()
plt.show()


# In[ ]:


print(df_selected.columns)


# In[ ]:


import seaborn as sns

sns.heatmap(df_selected.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Matrisi", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


import seaborn as sns

plt.figure(figsize=(10, 8))
plt.xlabel('SAD', fontsize=14)
plt.ylabel('OFCS', fontsize=14)
plt.legend(title='SAD', loc='upper center')  # Legend konumunu 
plt.grid(True)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

sns.set_style('whitegrid')

plt.show()


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, BayesianRidge
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

X = df_selected.drop(['SAD'], axis=1)
y_SAD = df['SAD']
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Define transformations for numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
    ])

# Using balanced class weight
pipeline_SAD = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('model', 
                               )])



y_pred_SAD = pipeline_SAD.predict(X_test_SAD)

# Metrikleri hesaplama
precision = precision_score(y_test_SAD, y_pred_SAD, average='macro')
recall = recall_score(y_test_SAD, y_pred_SAD, average='macro')  # Recall = Sensitivity
f1 = f1_score(y_test_SAD, y_pred_SAD, average='macro')

# Confusion Matrix ile Specificity hesaplama
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity = TN / (TN + FP)


print(f"Precision for SAD: {precision * 100:.2f}%")
print(f"Sensitivity (Recall) for SAD: {recall * 100:.2f}%")
print(f"Specificity for SAD: {specificity * 100:.2f}%")
print(f"F1 Score for SAD: {f1 * 100:.2f}%")
print(f"AUC for SAD: {auc * 100:.2f}%")
mcc = matthews_corrcoef(y_test_SAD, y_pred_SAD)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
kappa = cohen_kappa_score(y_test_SAD, y_pred_SAD)
print(f"Cohen's Kappa Score: {kappa:.4f}")


# In[ ]:




# In[ ]:


import seaborn as sns
import numpy as np



group_counts = ["{0:0.0f}".format(value) for value in cf_mtx.flatten()]
box_labels = np.asarray(group_counts).reshape(cf_mtx.shape)

sns.set(style="whitegrid") 

sns.heatmap(cf_mtx, annot=box_labels, fmt="", cmap="Blues", linewidths=0.5, cbar=False,
            annot_kws={"size": 16, "weight": "bold"}, square=True, 
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)

plt.title('Extra Trees Classifier Accuracy', fontsize=16, weight='bold')
plt.xlabel('Predicted Class', fontsize=14, weight='bold')
plt.ylabel('True Class', fontsize=14, weight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')


plt.show()


# In[ ]:


from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np

y_score = pipeline_SAD.predict_proba(X_test_SAD)

y_test_bin = label_binarize(y_test_SAD, classes=np.unique(y_test_SAD))

if y_test_bin.shape[1] == 1:
    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])


plt.figure(figsize=(8, 6))
for i, (color, class_name) in enumerate(zip(colors, class_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18, color='black')
plt.ylabel('True Positive Rate', fontsize=18, color='black')
plt.title('ROC Curve for Extra Trees Classifier', fontsize=18, weight='bold', color='black')
plt.legend(loc="lower right", fontsize=12, fancybox=True, framealpha=1, facecolor='#ffcccb' )
plt.xticks(fontsize=14, color='black')
plt.yticks(fontsize=14, color='black')
plt.grid(True, linestyle='--', alpha=0.7, color='black')

plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')

plt.savefig("ROC.svg", format='svg', bbox_inches='tight', pad_inches=0.1)

plt.tight_layout()
plt.show()