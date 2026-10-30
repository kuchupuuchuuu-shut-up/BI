
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print(df.head())
print(df["species"].value_counts())


# ---------------- CELL 2: Prepare features and split the data ----------------
X = df.drop(columns="species")
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set:", X_train.shape, " Testing set:", X_test.shape)


# ---------------- CELL 3: Decision Tree classifier ----------------
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, pred_tree)
print("Decision Tree accuracy:", round(acc_tree, 3))

plt.figure(figsize=(11, 6))
plot_tree(
    tree, feature_names=X.columns,
    class_names=list(tree.classes_), filled=True, fontsize=8
)
plt.show()


# ---------------- CELL 4: K-Nearest Neighbours classifier ----------------
scaler = StandardScaler().fit(X_train)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(scaler.transform(X_train), y_train)
pred_knn = knn.predict(scaler.transform(X_test))
acc_knn = accuracy_score(y_test, pred_knn)
print("KNN accuracy:", round(acc_knn, 3))


# ---------------- CELL 5: Naive Bayes classifier ----------------
nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, pred_nb)
print("Naive Bayes accuracy:", round(acc_nb, 3))


# ---------------- CELL 6: Compare the models ----------------
results = pd.DataFrame({
    "Model": ["Decision Tree", "KNN", "Naive Bayes"],
    "Accuracy": [acc_tree, acc_knn, acc_nb]
}).round(3)
print(results)

results.plot(
    x="Model", y="Accuracy", kind="bar",
    ylim=(0.8, 1.02), legend=False,
    figsize=(6, 3.5), rot=0
)
plt.title("Accuracy of the classifiers")
plt.show()


# ---------------- CELL 7: Confusion matrix and classification report (KNN) ----------------
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
ConfusionMatrixDisplay.from_predictions(y_test, pred_knn, cmap="Blues")
plt.title("Confusion matrix - KNN")
plt.show()


# ---------------- CELL 8: Predict a new record ----------------
new = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=X.columns)
print("Predicted species:", knn.predict(scaler.transform(new))[0])
