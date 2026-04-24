import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')


# Загрузка данных
train = pd.read_csv('disease_train.csv')
test = pd.read_csv('disease_public_test.csv')
sub = pd.read_csv('disease_sample_submission.csv')   # содержит Y для тестовой выборки

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission shape:", sub.shape)
print("\nПервые 5 строк train:")
print(train.head())
print("\nСтатистика целевой переменной в train:")
print(train['Y'].value_counts())

# Проверка на пропуски
print("\nПропуски в train:\n", train.isnull().sum())
print("Пропуски в test:\n", test.isnull().sum())


# Подготовка признаков и целевой переменной
X_train = train.drop('Y', axis=1)
y_train = train['Y']
X_test = test.copy()
y_test = sub['Y']    # правильные метки для теста


# Обучение DecisionTreeClassifier на сырых данных

dt_raw = DecisionTreeClassifier(random_state=42)
dt_raw.fit(X_train, y_train)
y_pred_raw = dt_raw.predict(X_test)
y_proba_raw = dt_raw.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("ОЦЕНКА НА СЫРЫХ ДАННЫХ")
print("="*60)
print(classification_report(y_test, y_pred_raw))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_raw))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_raw))


# Предобработка (масштабирование)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_scaled = DecisionTreeClassifier(random_state=42)
dt_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = dt_scaled.predict(X_test_scaled)
y_proba_scaled = dt_scaled.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*60)
print("ОЦЕНКА ПОСЛЕ МАСШТАБИРОВАНИЯ")
print("="*60)
print(classification_report(y_test, y_pred_scaled))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_scaled))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_scaled))


# Сравнение результатов (таблица)

results = pd.DataFrame({
    'Модель': ['DecisionTree (raw)', 'DecisionTree (scaled)'],
    'Accuracy': [accuracy_score(y_test, y_pred_raw), accuracy_score(y_test, y_pred_scaled)],
    'ROC-AUC': [roc_auc_score(y_test, y_proba_raw), roc_auc_score(y_test, y_proba_scaled)]
})
print("\n" + "="*60)
print("СРАВНЕНИЕ МЕТРИК")
print("="*60)
print(results)


# Кластеризация (KMeans)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
test_clusters = kmeans.predict(X_test)

ari = adjusted_rand_score(y_test, test_clusters)
ami = adjusted_mutual_info_score(y_test, test_clusters)
sil = silhouette_score(X_test, test_clusters)

print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ (KMeans)")
print("="*60)
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Adjusted Mutual Info: {ami:.3f}")
print(f"Silhouette Score (test): {sil:.3f}")


# Визуализация
# ROC-кривая
plt.figure(figsize=(8,6))
fpr, tpr, _ = roc_curve(y_test, y_proba_raw)
plt.plot(fpr, tpr, label=f'DecisionTree raw (AUC = {roc_auc_score(y_test, y_proba_raw):.2f})')
fpr2, tpr2, _ = roc_curve(y_test, y_proba_scaled)
plt.plot(fpr2, tpr2, label=f'DecisionTree scaled (AUC = {roc_auc_score(y_test, y_proba_scaled):.2f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые')
plt.legend()
plt.grid()
plt.show()

# Матрицы ошибок (тепловые карты)
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_raw), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Матрица ошибок (raw)')
sns.heatmap(confusion_matrix(y_test, y_pred_scaled), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Матрица ошибок (scaled)')
plt.tight_layout()
plt.show()
