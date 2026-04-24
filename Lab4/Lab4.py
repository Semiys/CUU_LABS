import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')


# 1. Загрузка данных
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


# 2. Подготовка признаков и целевой переменной
X_train = train.drop('Y', axis=1)
y_train = train['Y']
X_test = test.copy()
y_test = sub['Y']    # правильные метки для теста


# ------------------------------------------------------------
# 3. Классификатор DecisionTreeClassifier на исходных данных
# ------------------------------------------------------------
dt_raw = DecisionTreeClassifier(random_state=42)
dt_raw.fit(X_train, y_train)
y_pred_raw = dt_raw.predict(X_test)
y_proba_raw = dt_raw.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("ОЦЕНКА НА СЫРЫХ ДАННЫХ (исходное разделение)")
print("="*60)
print(classification_report(y_test, y_pred_raw))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_raw))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_raw))


# ------------------------------------------------------------
# 4. Предобработка (масштабирование) + оценка
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_scaled = DecisionTreeClassifier(random_state=42)
dt_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = dt_scaled.predict(X_test_scaled)
y_proba_scaled = dt_scaled.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*60)
print("ОЦЕНКА ПОСЛЕ МАСШТАБИРОВАНИЯ (исходное разделение)")
print("="*60)
print(classification_report(y_test, y_pred_scaled))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_scaled))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_scaled))

# Сравнение метрик
results = pd.DataFrame({
    'Модель': ['DecisionTree (raw)', 'DecisionTree (scaled)'],
    'Accuracy': [accuracy_score(y_test, y_pred_raw), accuracy_score(y_test, y_pred_scaled)],
    'ROC-AUC': [roc_auc_score(y_test, y_proba_raw), roc_auc_score(y_test, y_proba_scaled)]
})
print("\n" + "="*60)
print("СРАВНЕНИЕ МЕТРИК")
print("="*60)
print(results)


# ------------------------------------------------------------
# 5. Разные методы формирования train/test (п. 8 ТЗ)
# ------------------------------------------------------------
print("\n" + "="*60)
print("ОБУЧЕНИЕ НА ОЧИЩЕННЫХ ДАННЫХ С РАЗНЫМИ РАЗБИЕНИЯМИ")
print("="*60)

# Объединяем все данные с метками для возможности различных разбиений
all_data = pd.concat([train, pd.DataFrame(X_test, columns=X_train.columns).assign(Y=y_test)], ignore_index=True)
X_all = all_data.drop('Y', axis=1)
y_all = all_data['Y']

# Способ 1: train_test_split с random_state=42
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
scaler1 = StandardScaler()
X_tr1_scaled = scaler1.fit_transform(X_tr1)
X_te1_scaled = scaler1.transform(X_te1)
dt1 = DecisionTreeClassifier(random_state=42).fit(X_tr1_scaled, y_tr1)
acc1 = accuracy_score(y_te1, dt1.predict(X_te1_scaled))
print(f"\n1) train_test_split (test_size=0.3, random_state=42): accuracy = {acc1:.4f}")

# Способ 2: train_test_split с random_state=123
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_all, y_all, test_size=0.3, random_state=123)
scaler2 = StandardScaler()
X_tr2_scaled = scaler2.fit_transform(X_tr2)
X_te2_scaled = scaler2.transform(X_te2)
dt2 = DecisionTreeClassifier(random_state=42).fit(X_tr2_scaled, y_tr2)
acc2 = accuracy_score(y_te2, dt2.predict(X_te2_scaled))
print(f"2) train_test_split (test_size=0.3, random_state=123): accuracy = {acc2:.4f}")

# Способ 3: кросс-валидация (5-fold) на исходном train (масштабированном)
cv_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"3) 5-fold кросс-валидация на исходном train (масштаб.): средняя accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# ------------------------------------------------------------
# 6. Кластеризация KMeans и визуализация (п. 14 ТЗ)
# ------------------------------------------------------------
print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ (KMeans)")
print("="*60)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)  # обучаем на масштабированных признаках
test_clusters = kmeans.predict(X_test_scaled)

ari = adjusted_rand_score(y_test, test_clusters)
ami = adjusted_mutual_info_score(y_test, test_clusters)
sil = silhouette_score(X_test_scaled, test_clusters)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Adjusted Mutual Info: {ami:.3f}")
print(f"Silhouette Score (test): {sil:.3f}")

# Визуализация кластеров – используем два первых признака для наглядности
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.scatter(X_test_scaled[:,0], X_test_scaled[:,1], c=test_clusters, cmap='viridis', alpha=0.6)
plt.xlabel('X1 (scaled)')
plt.ylabel('X2 (scaled)')
plt.title('Кластеры, предсказанные KMeans')
plt.colorbar(label='Кластер')

plt.subplot(1,2,2)
plt.scatter(X_test_scaled[:,0], X_test_scaled[:,1], c=y_test, cmap='coolwarm', alpha=0.6)
plt.xlabel('X1 (scaled)')
plt.ylabel('X2 (scaled)')
plt.title('Истинные метки')
plt.colorbar(label='Y (0/1)')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7. ROC-кривая и матрицы ошибок (п. 10 ТЗ)
# ------------------------------------------------------------
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

fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_raw), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Матрица ошибок (raw)')
sns.heatmap(confusion_matrix(y_test, y_pred_scaled), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Матрица ошибок (scaled)')
plt.tight_layout()
plt.show()
