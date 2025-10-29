# Load dataset
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- SVM Model ----
svm_model = SVC(kernel='rbf', gamma=0.05)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# ---- k-NN Model ----
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# ---- Evaluation ----
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("k-NN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Visualization
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.heatmap(cm_svm, annot=True, cmap='Blues')
plt.title("SVM Confusion Matrix")

plt.subplot(1,2,2)
sns.heatmap(cm_knn, annot=True, cmap='Greens')
plt.title("k-NN Confusion Matrix")
plt.show()
