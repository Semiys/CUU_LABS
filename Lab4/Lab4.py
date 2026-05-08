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


# ручная реализация дерева решений (CustomDecisionTreeClassifier)
class Node:
    """Узел дерева решений"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 value=None, is_leaf=False):
        self.feature_idx = feature_idx      # индекс признака для разбиения
        self.threshold = threshold          # порог разбиения
        self.left = left                    # левое поддерево (True)
        self.right = right                  # правое поддерево (False)
        self.value = value                  # класс (если лист)
        self.is_leaf = is_leaf

class CustomDecisionTreeClassifier:
    """
    Ручная реализация дерева решений для классификации.
    Используется критерий Джини (gini impurity).
    Поддерживается бинарная классификация (можно расширить на многоклассовую).
    """
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        self.n_classes = None
        self.classes_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _gini(self, y):
        """Расчёт примеси Джини для набора меток"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _entropy(self, y):
        """Расчёт энтропии для набора меток"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return -np.sum(proportions * np.log2(proportions + 1e-9))

    def _criterion_func(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("criterion must be 'gini' or 'entropy'")

    def _best_split(self, X, y):
        """Находит лучшее разбиение (признак и порог)"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        parent_impurity = self._criterion_func(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thr in thresholds:
                left_mask = X[:, feature] <= thr
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                y_left = y[left_mask]
                y_right = y[right_mask]
                # Взвешенная примесь
                left_impurity = self._criterion_func(y_left)
                right_impurity = self._criterion_func(y_right)
                n_left = len(y_left)
                n_right = len(y_right)
                n_total = n_left + n_right
                weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = thr
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Рекурсивное построение дерева"""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Условия остановки
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # Лист: наиболее частый класс
            values = np.bincount(y)
            node = Node(is_leaf=True, value=np.argmax(values))
            return node

        # Поиск лучшего разбиения
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            # Не удалось найти разбиение – лист
            values = np.bincount(y)
            node = Node(is_leaf=True, value=np.argmax(values))
            return node

        # Разделение данных
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # Рекурсивное построение поддеревьев
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature_idx=feature_idx, threshold=threshold,
                    left=left_child, right=right_child, is_leaf=False)

    def fit(self, X, y):
        """Обучение дерева"""
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        # Преобразуем метки в целые числа для бинарной классификации
        if self.n_classes == 2:
            # Считаем, что классы уже 0 и 1
            pass
        else:
            # Для простоты приведём к последовательным числам
            self.label_map = {cls: i for i, cls in enumerate(self.classes_)}
            y = np.array([self.label_map[val] for val in y])
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        """Рекурсивное предсказание для одного объекта"""
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        """Предсказание для набора объектов"""
        X = np.asarray(X)
        preds = [self._predict_one(x, self.root) for x in X]
        preds = np.array(preds)
        # Преобразуем обратно к исходным классам (если есть label_map)
        if hasattr(self, 'label_map'):
            inv_map = {v: k for k, v in self.label_map.items()}
            preds = np.array([inv_map[p] for p in preds])
        return preds

    def predict_proba(self, X):
        """Оценка вероятностей (приближённая, не реализована глубоко – для простоты возвращает 0/1)"""
        # Более точную реализацию можно сделать, но для демонстрации хватит
        preds = self.predict(X)
        proba = np.zeros((len(preds), self.n_classes))
        for i, p in enumerate(preds):
            if self.n_classes == 2:
                proba[i] = [1-p, p] if p == 1 else [1, 0]  # упрощённо
            else:
                proba[i][p] = 1.0
        return proba



# Загрузка данных
train = pd.read_csv('disease_train.csv')
test = pd.read_csv('disease_public_test.csv')
sub = pd.read_csv('disease_sample_submission.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission shape:", sub.shape)
print("\nПервые 5 строк train:")
print(train.head())
print("\nСтатистика целевой переменной в train:")
print(train['Y'].value_counts())
print("\nПропуски в train:\n", train.isnull().sum())
print("Пропуски в test:\n", test.isnull().sum())

X_train = train.drop('Y', axis=1)
y_train = train['Y']
X_test = test.copy()
y_test = sub['Y']


# Классификатор DecisionTreeClassifier (sklearn) на сырых данных
dt_raw = DecisionTreeClassifier(random_state=42)
dt_raw.fit(X_train, y_train)
y_pred_raw = dt_raw.predict(X_test)
y_proba_raw = dt_raw.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("ОЦЕНКА НА СЫРЫХ ДАННЫХ (sklearn DecisionTree)")
print("="*60)
print(classification_report(y_test, y_pred_raw))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_raw))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_raw))


# учная реализация дерева решений (на сырых данных)
custom_tree = CustomDecisionTreeClassifier(max_depth=10, min_samples_split=2, criterion='gini', random_state=42)
custom_tree.fit(X_train, y_train)
y_pred_custom = custom_tree.predict(X_test)
# Для вероятностей (упрощённо)
y_proba_custom = custom_tree.predict_proba(X_test)[:, 1] if custom_tree.n_classes == 2 else None

print("\n" + "="*60)
print("ОЦЕНКА НА СЫРЫХ ДАННЫХ (ручная реализация)")
print("="*60)
print(classification_report(y_test, y_pred_custom))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_custom))
if y_proba_custom is not None:
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_custom))

# Масштабирование и обучение (sklearn + ручной)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_scaled_sk = DecisionTreeClassifier(random_state=42)
dt_scaled_sk.fit(X_train_scaled, y_train)
y_pred_scaled_sk = dt_scaled_sk.predict(X_test_scaled)
y_proba_scaled_sk = dt_scaled_sk.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*60)
print("ОЦЕНКА ПОСЛЕ МАСШТАБИРОВАНИЯ (sklearn DecisionTree)")
print("="*60)
print(classification_report(y_test, y_pred_scaled_sk))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_scaled_sk))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_scaled_sk))

custom_tree_scaled = CustomDecisionTreeClassifier(max_depth=10, min_samples_split=2, criterion='gini', random_state=42)
custom_tree_scaled.fit(X_train_scaled, y_train)
y_pred_custom_scaled = custom_tree_scaled.predict(X_test_scaled)
y_proba_custom_scaled = custom_tree_scaled.predict_proba(X_test_scaled)[:, 1] if custom_tree_scaled.n_classes == 2 else None

print("\n" + "="*60)
print("ОЦЕНКА ПОСЛЕ МАСШТАБИРОВАНИЯ (ручная реализация)")
print("="*60)
print(classification_report(y_test, y_pred_custom_scaled))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_custom_scaled))
if y_proba_custom_scaled is not None:
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_custom_scaled))

# Сравнение метрик (sklearn vs ручная)
results = pd.DataFrame({
    'Модель': ['sklearn raw', 'ручная raw', 'sklearn scaled', 'ручная scaled'],
    'Accuracy': [accuracy_score(y_test, y_pred_raw),
                 accuracy_score(y_test, y_pred_custom),
                 accuracy_score(y_test, y_pred_scaled_sk),
                 accuracy_score(y_test, y_pred_custom_scaled)],
    'ROC-AUC': [roc_auc_score(y_test, y_proba_raw),
                roc_auc_score(y_test, y_proba_custom) if y_proba_custom is not None else np.nan,
                roc_auc_score(y_test, y_proba_scaled_sk),
                roc_auc_score(y_test, y_proba_custom_scaled) if y_proba_custom_scaled is not None else np.nan]
})
print("\n" + "="*60)
print("СРАВНЕНИЕ МЕТРИК (sklearn vs ручная реализация)")
print("="*60)
print(results)


# Разные методы формирования train/test
print("\n" + "="*60)
print("ОБУЧЕНИЕ НА ОЧИЩЕННЫХ ДАННЫХ С РАЗНЫМИ РАЗБИЕНИЯМИ")
print("="*60)

all_data = pd.concat([train, pd.DataFrame(X_test, columns=X_train.columns).assign(Y=y_test)], ignore_index=True)
X_all = all_data.drop('Y', axis=1)
y_all = all_data['Y']

# Способ 1
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
scaler1 = StandardScaler()
X_tr1_scaled = scaler1.fit_transform(X_tr1)
X_te1_scaled = scaler1.transform(X_te1)
dt1 = DecisionTreeClassifier(random_state=42).fit(X_tr1_scaled, y_tr1)
acc1 = accuracy_score(y_te1, dt1.predict(X_te1_scaled))
print(f"\n1) train_test_split (test_size=0.3, random_state=42): accuracy = {acc1:.4f}")

# Способ 2
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_all, y_all, test_size=0.3, random_state=123)
scaler2 = StandardScaler()
X_tr2_scaled = scaler2.fit_transform(X_tr2)
X_te2_scaled = scaler2.transform(X_te2)
dt2 = DecisionTreeClassifier(random_state=42).fit(X_tr2_scaled, y_tr2)
acc2 = accuracy_score(y_te2, dt2.predict(X_te2_scaled))
print(f"2) train_test_split (test_size=0.3, random_state=123): accuracy = {acc2:.4f}")

# Способ 3
cv_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"3) 5-fold кросс-валидация на исходном train (масштаб.): средняя accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# Кластеризация KMeans и визуализация
print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ (KMeans)")
print("="*60)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

ari = adjusted_rand_score(y_test, test_clusters)
ami = adjusted_mutual_info_score(y_test, test_clusters)
sil = silhouette_score(X_test_scaled, test_clusters)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Adjusted Mutual Info: {ami:.3f}")
print(f"Silhouette Score (test): {sil:.3f}")

# Визуализация кластеров
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.scatter(X_test_scaled[:,0], X_test_scaled[:,1], c=test_clusters, cmap='viridis', alpha=0.6)
plt.xlabel('X1 (scaled)'); plt.ylabel('X2 (scaled)')
plt.title('Кластеры KMeans'); plt.colorbar(label='Кластер')
plt.subplot(1,2,2)
plt.scatter(X_test_scaled[:,0], X_test_scaled[:,1], c=y_test, cmap='coolwarm', alpha=0.6)
plt.xlabel('X1 (scaled)'); plt.ylabel('X2 (scaled)')
plt.title('Истинные метки'); plt.colorbar(label='Y (0/1)')
plt.tight_layout()
plt.show()

# ROC-кривые и матрицы ошибок
plt.figure(figsize=(8,6))
fpr, tpr, _ = roc_curve(y_test, y_proba_raw)
plt.plot(fpr, tpr, label=f'DecisionTree sklearn raw (AUC={roc_auc_score(y_test, y_proba_raw):.2f})')
fpr2, tpr2, _ = roc_curve(y_test, y_proba_scaled_sk)
plt.plot(fpr2, tpr2, label=f'DecisionTree sklearn scaled (AUC={roc_auc_score(y_test, y_proba_scaled_sk):.2f})')
if y_proba_custom is not None:
    fpr3, tpr3, _ = roc_curve(y_test, y_proba_custom)
    plt.plot(fpr3, tpr3, '--', label=f'ручная raw (AUC={roc_auc_score(y_test, y_proba_custom):.2f})')
if y_proba_custom_scaled is not None:
    fpr4, tpr4, _ = roc_curve(y_test, y_proba_custom_scaled)
    plt.plot(fpr4, tpr4, '--', label=f'ручная scaled (AUC={roc_auc_score(y_test, y_proba_custom_scaled):.2f})')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC-кривые'); plt.legend(); plt.grid()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_raw), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Матрица ошибок (sklearn raw)')
sns.heatmap(confusion_matrix(y_test, y_pred_custom), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Матрица ошибок (ручная raw)')
plt.tight_layout()
plt.show()
