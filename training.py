#ASSIGNMENT 5

##Importing libraries

import numpy as np
import pandas as pd

##Importing the dataset

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

x_train = dataset_train.iloc[:, :-1].values
x_test = dataset_test.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1]
y_test = dataset_test.iloc[:, -1]

print(np.unique(y_test))

##Checking for missing values

missing_train = dataset_train.isnull().sum()
missing_test = dataset_test.isnull().sum()
print(missing_train[missing_train > 0])
print(missing_test[missing_test > 0])

##Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

##Logistic Regression

from sklearn.linear_model import LogisticRegression
logi_reg = LogisticRegression(max_iter=200, random_state = 0)
logi_reg.fit(x_train_sc, y_train)

##K-NN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train_sc, y_train)

##SVM

from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 0)
svm.fit(x_train_sc, y_train)

##Kernel SVM

k_svm = SVC(kernel = 'rbf', random_state = 0)
k_svm.fit(x_train_sc, y_train)

##Naive Bayes

from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(x_train_sc, y_train)

##Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(x_train_sc, y_train)

##Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(x_train_sc, y_train)

##Prediction

y_pred_logi = logi_reg.predict(x_test_sc)
y_pred_knn = knn.predict(x_test_sc)
y_pred_svm = svm.predict(x_test_sc)
y_pred_k_svm = k_svm.predict(x_test_sc)
y_pred_naive = naive.predict(x_test_sc)
y_pred_tree = tree.predict(x_test_sc)
y_pred_forest = forest.predict(x_test_sc)

##Performance

from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_classifier(y_true, y_pred, model_name):
  cm = confusion_matrix(y_true, y_pred)
  acc = accuracy_score(y_true, y_pred)

  print(f"{model_name} Performance")
  print("Confusion Matrix:")
  print(cm)
  print(f"Accuracy: {acc:.2f}\n")

evaluate_classifier(y_test, y_pred_logi, "Logistic Regression")
evaluate_classifier(y_test, y_pred_knn, "K-NN")
evaluate_classifier(y_test, y_pred_svm, "SVM")
evaluate_classifier(y_test, y_pred_k_svm, "Kernel SVM")
evaluate_classifier(y_test, y_pred_naive, "Naive Bayes")
evaluate_classifier(y_test, y_pred_tree, "Decision Tree")
evaluate_classifier(y_test, y_pred_forest, "Random Forest")
