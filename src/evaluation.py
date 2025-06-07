import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
scatter = sns.scatterplot(x=df_selected['SAD'], y=df_selected['OFCS'], hue=df_selected['SAD'], palette='coolwarm', s=100, edgecolor='k', alpha=0.7)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
accuracy = accuracy_score(y_test_SAD, y_pred_SAD)
cm = confusion_matrix(y_test_SAD, y_pred_SAD)
auc = roc_auc_score(y_test_SAD, y_prob_SAD)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
print(classification_report(y_test_SAD, y_pred_SAD))
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cf_mtx = confusion_matrix(y_test_SAD, y_pred_SAD)
plt.savefig("confusion_matrix.svg", format='svg', bbox_inches='tight', pad_inches=0.1)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)