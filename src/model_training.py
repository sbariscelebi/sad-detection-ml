from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
                                ExtraTreesClassifier(n_estimators=100, random_state=0)  
X_train_SAD, X_test_SAD, y_train_SAD, y_test_SAD = train_test_split(X, y_SAD, test_size=0.2, random_state=0)
pipeline_SAD.fit(X_train_SAD, y_train_SAD)
print(f"ExtraTreesClassifier accuracy: {accuracy * 100:.2f}%")