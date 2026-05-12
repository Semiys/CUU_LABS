import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных
train = pd.read_csv('train.tsv', sep='\t', header=None)
test = pd.read_csv('test.tsv', sep='\t', header=None)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Целевая переменная – последний столбец в train
X_train_full = train.iloc[:, :-1].values
y_train_full = train.iloc[:, -1].values
X_test = test.values

print("Количество признаков:", X_train_full.shape[1])
print("Целевая: min =", y_train_full.min(), "max =", y_train_full.max())

# Пропуски
print("Пропуски в train:", np.isnan(X_train_full).sum())
print("Пропуски в test:", np.isnan(X_test).sum())
if np.isnan(X_train_full).sum() > 0 or np.isnan(X_test).sum() > 0:
    print("ВНИМАНИЕ: обнаружены пропуски! Нужно заполнить (например, медианой).")

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median')
    X_train_full = imp.fit_transform(X_train_full)
    X_test = imp.transform(X_test)


# 2. Разделение на обучение и валидацию
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
print("\nРазмеры: train =", X_train.shape, ", val =", X_val.shape)


# 3. Модель 1: DecisionTreeRegressor (для сравнения)
dt_raw = DecisionTreeRegressor(random_state=42)
dt_raw.fit(X_train, y_train)
y_pred_dt = dt_raw.predict(X_val)

print("\n" + "="*60)
print("ОЦЕНКА DecisionTreeRegressor (сырые данные)")
print("="*60)
print(f"MSE: {mean_squared_error(y_val, y_pred_dt):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_dt)):.4f}")
print(f"MAE: {mean_absolute_error(y_val, y_pred_dt):.4f}")
print(f"R2: {r2_score(y_val, y_pred_dt):.4f}")


# 4. Предобработка (масштабирование)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# DecisionTree после масштабирования
dt_scaled = DecisionTreeRegressor(random_state=42)
dt_scaled.fit(X_train_scaled, y_train)
y_pred_dt_scaled = dt_scaled.predict(X_val_scaled)
print("\nDecisionTreeRegressor (масштабированные данные)")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_dt_scaled)):.4f}, R2: {r2_score(y_val, y_pred_dt_scaled):.4f}")

# 5. Основной метод: BayesianRidge
br = BayesianRidge()
br.fit(X_train_scaled, y_train)
y_pred_br = br.predict(X_val_scaled)

print("\n" + "="*60)
print("ОЦЕНКА BayesianRidge (масштабированные данные)")
print("="*60)
print(f"MSE: {mean_squared_error(y_val, y_pred_br):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_br)):.4f}")
print(f"MAE: {mean_absolute_error(y_val, y_pred_br):.4f}")
print(f"R2: {r2_score(y_val, y_pred_br):.4f}")

# 6. Разные методы разбиения train/test (демонстрация)
print("\n" + "="*60)
print("ВЛИЯНИЕ РАЗНЫХ РАЗБИЕНИЙ (BayesianRidge)")
print("="*60)
for rs in [42, 123, 7]:
    X_tr, X_te, y_tr, y_te = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=rs)
    scaler_tmp = StandardScaler()
    X_tr_scaled = scaler_tmp.fit_transform(X_tr)
    X_te_scaled = scaler_tmp.transform(X_te)
    br_tmp = BayesianRidge()
    br_tmp.fit(X_tr_scaled, y_tr)
    y_pred_tmp = br_tmp.predict(X_te_scaled)
    r2_tmp = r2_score(y_te, y_pred_tmp)
    print(f"random_state={rs}: R2 = {r2_tmp:.4f}")

# 7. Кросс-валидация (BayesianRidge)
# Используем масштабирование внутри pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('br', BayesianRidge())])
cv_scores = cross_val_score(pipeline, X_train_full, y_train_full, cv=5, scoring='r2')
print(f"\n5‑fold кросс-валидация R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 8. Кластеризация KMeans (на признаках)
print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ (KMeans) на масштабированных train")
print("="*60)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
clusters_val = kmeans.predict(X_val_scaled)
sil = silhouette_score(X_val_scaled, clusters_val)
print(f"Silhouette Score (на валидации): {sil:.3f}")

# Бинаризуем целевую переменную по медиане для сравнения
median_y = np.median(y_val)
y_val_bin = (y_val > median_y).astype(int)
ari = adjusted_rand_score(y_val_bin, clusters_val)
ami = adjusted_mutual_info_score(y_val_bin, clusters_val)
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Adjusted Mutual Info: {ami:.3f}")

# Визуализация кластеров через PCA
pca = PCA(n_components=2)
X_val_pca = pca.fit_transform(X_val_scaled)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X_val_pca[:,0], X_val_pca[:,1], c=clusters_val, cmap='viridis', alpha=0.6)
plt.title('Кластеры KMeans')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.subplot(1,2,2)
plt.scatter(X_val_pca[:,0], X_val_pca[:,1], c=y_val_bin, cmap='coolwarm', alpha=0.6)
plt.title('Истинные метки (бинарные по медиане)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# 9. Визуализация предсказаний BayesianRidge
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred_br, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('BayesianRidge: предсказания vs истина')
plt.show()

# Гистограмма остатков
residuals = y_val - y_pred_br
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Остатки')
plt.ylabel('Частота')
plt.title('Распределение остатков (BayesianRidge)')
plt.show()

# 10. Итоговая таблица метрик
print("\n" + "="*60)
print("ИТОГОВОЕ СРАВНЕНИЕ МЕТРИК")
print("="*60)
summary = pd.DataFrame({
    'Модель': ['DecisionTree (raw)', 'DecisionTree (scaled)', 'BayesianRidge (scaled)'],
    'RMSE': [np.sqrt(mean_squared_error(y_val, y_pred_dt)),
             np.sqrt(mean_squared_error(y_val, y_pred_dt_scaled)),
             np.sqrt(mean_squared_error(y_val, y_pred_br))],
    'R2': [r2_score(y_val, y_pred_dt),
           r2_score(y_val, y_pred_dt_scaled),
           r2_score(y_val, y_pred_br)]
})
print(summary)


