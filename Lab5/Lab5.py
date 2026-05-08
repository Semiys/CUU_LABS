import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Загрузка данных
# ===============================
# train.tsv содержит 100 признаков и целевую переменную (последний столбец)
# test.tsv содержит 100 признаков (без целевой)
train = pd.read_csv('train.tsv', sep='\t', header=None)
test = pd.read_csv('test.tsv', sep='\t', header=None)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Целевая переменная – последний столбец в train
X_train_full = train.iloc[:, :-1].values
y_train_full = train.iloc[:, -1].values
X_test = test.values

print("Признаков:", X_train_full.shape[1])
print("Целевая переменная: min =", y_train_full.min(), "max =", y_train_full.max())

# Проверка на пропуски
print("Пропуски в train:", np.isnan(X_train_full).sum())
print("Пропуски в test:", np.isnan(X_test).sum())

# ===============================
# 2. Разделение train на обучение и валидацию (для оценки)
# ===============================
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

print("\nРазмеры: train =", X_train.shape, "val =", X_val.shape)

# ===============================
# 3. Обучение DecisionTreeRegressor на сырых данных
# ===============================
dt_raw = DecisionTreeRegressor(random_state=42)
dt_raw.fit(X_train, y_train)
y_pred_raw = dt_raw.predict(X_val)

print("\n" + "="*60)
print("ОЦЕНКА НА СЫРЫХ ДАННЫХ")
print("="*60)
print("MSE:", mean_squared_error(y_val, y_pred_raw))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_raw)))
print("MAE:", mean_absolute_error(y_val, y_pred_raw))
print("R2:", r2_score(y_val, y_pred_raw))

# ===============================
# 4. Предобработка (масштабирование) + снова обучение
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

dt_scaled = DecisionTreeRegressor(random_state=42)
dt_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = dt_scaled.predict(X_val_scaled)

print("\n" + "="*60)
print("ОЦЕНКА ПОСЛЕ МАСШТАБИРОВАНИЯ")
print("="*60)
print("MSE:", mean_squared_error(y_val, y_pred_scaled))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_scaled)))
print("MAE:", mean_absolute_error(y_val, y_pred_scaled))
print("R2:", r2_score(y_val, y_pred_scaled))

# Сравнение метрик
results = pd.DataFrame({
    'Модель': ['DecisionTree raw', 'DecisionTree scaled'],
    'RMSE': [np.sqrt(mean_squared_error(y_val, y_pred_raw)),
             np.sqrt(mean_squared_error(y_val, y_pred_scaled))],
    'R2': [r2_score(y_val, y_pred_raw), r2_score(y_val, y_pred_scaled)]
})
print("\nСРАВНЕНИЕ МЕТРИК")
print(results)

# ===============================
# 5. Кластеризация KMeans на признаках (без целевой)
# ===============================
print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ (KMeans)")
print("="*60)
# Используем масштабированные признаки (лучше для KMeans)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
clusters = kmeans.predict(X_val_scaled)

# Оценка качества кластеризации
sil = silhouette_score(X_val_scaled, clusters)
print(f"Silhouette Score (на валидации): {sil:.3f}")

# Визуализация кластеров через PCA (первые две главные компоненты)
pca = PCA(n_components=2)
X_val_pca = pca.fit_transform(X_val_scaled)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X_val_pca[:,0], X_val_pca[:,1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Кластеры KMeans (2 кластера)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.subplot(1,2,2)
# Сравним с целевой переменной (разобьём на 2 группы по медиане для визуализации)
y_val_binned = (y_val > np.median(y_val)).astype(int)
plt.scatter(X_val_pca[:,0], X_val_pca[:,1], c=y_val_binned, cmap='coolwarm', alpha=0.6)
plt.title('Истинные метки (бинарные по медиане)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# Согласованность кластеров с целевой переменной (Adjusted Rand Index)
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
ari = adjusted_rand_score(y_val_binned, clusters)
ami = adjusted_mutual_info_score(y_val_binned, clusters)
print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Adjusted Mutual Info (AMI): {ami:.3f}")

# ===============================
# 6. Визуализация предсказаний (для отчёта)
# ===============================
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred_scaled, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Предсказания DecisionTreeRegressor (scaled)')
plt.show()

# Гистограмма остатков
residuals = y_val - y_pred_scaled
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Остатки')
plt.ylabel('Частота')
plt.title('Распределение остатков')
plt.show()

print("\n✅ Работа завершена.")