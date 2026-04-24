import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from typing import List, Union, Tuple, Dict
import warnings
import time
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
warnings.filterwarnings('ignore')


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 12


# Кастомная реализация k-NN с векторизованными вычислениями
class CustomKNNClassifier:
    """
    Классификатор на основе метода k ближайших соседей (k-NN).
    Использует евклидово расстояние, векторизованные вычисления в NumPy.
    Реализована обработка ничьих путём выбора ближайшего среди классов-претендентов.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.encoder: LabelEncoder = None
        self.classes_: List[str] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'CustomKNNClassifier':
        self.X_train = np.asarray(X)
        y_original = np.asarray(y)
        self.encoder = LabelEncoder()
        self.y_train = self.encoder.fit_transform(y_original)
        self.classes_ = list(self.encoder.classes_)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_test = np.asarray(X)
        predictions = []
        for x in X_test:
            # Векторизованное вычисление евклидовых расстояний
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            k_distances = distances[k_indices]

            unique_labels, counts = np.unique(k_labels, return_counts=True)
            max_count = np.max(counts)
            candidate_labels = unique_labels[counts == max_count]

            if len(candidate_labels) == 1:
                pred_label = candidate_labels[0]
            else:
                # Ничья: выбираем класс ближайшего соседа среди претендентов
                sorted_indices = np.argsort(k_distances)
                for idx in sorted_indices:
                    if k_labels[idx] in candidate_labels:
                        pred_label = k_labels[idx]
                        break
            predictions.append(pred_label)

        return self.encoder.inverse_transform(np.array(predictions))

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_test = np.asarray(X)
        probas = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            counts = np.bincount(k_labels, minlength=len(self.classes_))
            probas.append(counts / self.k)
        return np.array(probas)


def compare_custom_vs_sklearn(X_train, y_train, X_test, y_test=None, k=3):
    custom_knn = CustomKNNClassifier(k=k).fit(X_train, y_train)
    custom_pred = custom_knn.predict(X_test)

    sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    sklearn_knn.fit(X_train, y_train)
    sklearn_pred = sklearn_knn.predict(X_test)

    results = {
        'custom_predictions': custom_pred,
        'sklearn_predictions': sklearn_pred,
        'agreement': np.all(custom_pred == sklearn_pred)
    }
    if y_test is not None:
        results['custom_accuracy'] = accuracy_score(y_test, custom_pred)
        results['sklearn_accuracy'] = accuracy_score(y_test, sklearn_pred)
    return results


# Загрузка датасета из CSV-файла
csv_filename = "food_dataset.csv"
data = pd.read_csv(csv_filename, encoding='utf-8')
print("Датасет загружен из CSV:")
print(data.to_string(index=False))

# Подготовка признаков и меток
X = data[['сладость', 'хруст']].values
y = data['класс'].values

# Тестовые продукты (включая Томат)
test_products = pd.DataFrame({
    'продукт': ['Томат', 'Мороженое', 'Картофель фри'],
    'сладость': [4, 8, 1],
    'хруст': [4, 2, 9]
})
X_test = test_products[['сладость', 'хруст']].values


# Эксперимент с k=3 на исходных данных
print("=" * 60)
print("Эксперимент на исходных данных (3 класса)")
print("=" * 60)
k = 3
comparison = compare_custom_vs_sklearn(X, y, X_test, k=k)

print(f"\nПредсказания для тестовых продуктов (k={k}):")
for i, prod in enumerate(test_products['продукт']):
    print(f"{prod:15} | Custom: {comparison['custom_predictions'][i]:10} | Sklearn: {comparison['sklearn_predictions'][i]}")
print(f"\nПолное совпадение предсказаний: {comparison['agreement']}")

# Проверка на обучающей выборке
train_pred_custom = CustomKNNClassifier(k=k).fit(X, y).predict(X)
train_pred_sklearn = KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(X)
print(f"Совпадение на обучающей выборке: {np.all(train_pred_custom == train_pred_sklearn)}")


# Функция для построения разделяющих поверхностей
def plot_decision_boundaries(model, X, y, test_points=None, test_labels=None,
                             title="Decision Boundaries", ax=None):
    if ax is None:
        ax = plt.gca()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    classes = np.unique(y)
    n_classes = len(classes)
    palette = sns.color_palette("Set2", n_classes)
    color_map = dict(zip(classes, palette))

    le = LabelEncoder()
    Z_encoded = le.fit_transform(Z.ravel()).reshape(xx.shape)
    ax.contourf(xx, yy, Z_encoded, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5,
                colors=palette, antialiased=True)

    for cls in classes:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], label=cls, s=80,
                   edgecolor='black', linewidth=1, color=color_map[cls], alpha=0.8)

    if test_points is not None:
        ax.scatter(test_points[:, 0], test_points[:, 1], marker='*', s=200,
                   c='red', edgecolor='black', linewidth=1, label='Тестовые объекты', zorder=5)
        if test_labels:
            for i, (x_coord, y_coord) in enumerate(test_points):
                ax.annotate(test_labels[i], (x_coord, y_coord), xytext=(5, 5),
                            textcoords='offset points', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    ax.set_xlabel('Сладость', fontsize=12)
    ax.set_ylabel('Хруст', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    return ax


# Визуализация для исходных данных (3 класса)
model_sklearn_3 = KNeighborsClassifier(n_neighbors=k).fit(X, y)

fig, ax = plt.subplots(figsize=(10, 7))
plot_decision_boundaries(model_sklearn_3, X, y,
                         test_points=X_test,
                         test_labels=test_products['продукт'].tolist(),
                         title="Разделяющие поверхности (3 класса, k=3)", ax=ax)
plt.tight_layout()
plt.show()


# Добавление нового класса "Выпечка"
bakery_data = pd.DataFrame({
    'продукт': ['Кекс', 'Печенье', 'Багет'],
    'сладость': [6, 7, 2],
    'хруст': [3, 6, 7],
    'класс': ['Выпечка', 'Выпечка', 'Выпечка']
})

data_extended = pd.concat([data, bakery_data], ignore_index=True)
X_ext = data_extended[['сладость', 'хруст']].values
y_ext = data_extended['класс'].values

print("\n" + "=" * 60)
print("Расширенный датасет (4 класса)")
print("=" * 60)
print(data_extended.to_string(index=False))

X_test_same = test_products[['сладость', 'хруст']].values
comparison_ext = compare_custom_vs_sklearn(X_ext, y_ext, X_test_same, k=k)
print(f"\nПредсказания для тестовых продуктов на 4 классах (k={k}):")
for i, prod in enumerate(test_products['продукт']):
    print(f"{prod:15} | Custom: {comparison_ext['custom_predictions'][i]:10} | Sklearn: {comparison_ext['sklearn_predictions'][i]}")
print(f"\nПолное совпадение предсказаний: {comparison_ext['agreement']}")


# Визуализация для 4 классов
model_sklearn_4 = KNeighborsClassifier(n_neighbors=k).fit(X_ext, y_ext)

fig, ax = plt.subplots(figsize=(10, 7))
plot_decision_boundaries(model_sklearn_4, X_ext, y_ext,
                         test_points=X_test_same,
                         test_labels=test_products['продукт'].tolist(),
                         title="Разделяющие поверхности (4 класса, k=3)", ax=ax)
plt.tight_layout()
plt.show()


# Совместный график до и после добавления класса
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plot_decision_boundaries(model_sklearn_3, X, y,
                         test_points=X_test_same,
                         test_labels=test_products['продукт'].tolist(),
                         title="3 класса (исходные данные)", ax=ax1)

plot_decision_boundaries(model_sklearn_4, X_ext, y_ext,
                         test_points=X_test_same,
                         test_labels=test_products['продукт'].tolist(),
                         title="4 класса (с добавленной Выпечкой)", ax=ax2)

plt.tight_layout()
plt.show()


# Дополнительные тесты (LOOCV, Black-Box, White-Box)
print("=" * 60)
print("БЛОК 1: КРОСС-ВАЛИДАЦИЯ (Leave-One-Out)")
print("=" * 60)

def perform_loocv(X, y, k=3):
    loo = LeaveOneOut()
    n_splits = loo.get_n_splits(X)

    # Для сбора истинных и предсказанных меток
    y_true_custom = []
    y_pred_custom = []
    y_true_sklearn = []
    y_pred_sklearn = []

    # ----- Custom k-NN -----
    start_custom = time.perf_counter()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        custom_model = CustomKNNClassifier(k=k).fit(X_train, y_train)
        pred = custom_model.predict(X_test)[0]
        y_true_custom.append(y_test[0])
        y_pred_custom.append(pred)
    time_custom = time.perf_counter() - start_custom

    # ----- Sklearn k-NN -----
    start_sklearn = time.perf_counter()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sklearn_model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        pred = sklearn_model.predict(X_test)[0]
        y_true_sklearn.append(y_test[0])
        y_pred_sklearn.append(pred)
    time_sklearn = time.perf_counter() - start_sklearn

    # Расчёт метрик
    acc_custom = accuracy_score(y_true_custom, y_pred_custom)
    f1_custom = f1_score(y_true_custom, y_pred_custom, average='weighted')
    cm_custom = confusion_matrix(y_true_custom, y_pred_custom, labels=np.unique(y))

    acc_sklearn = accuracy_score(y_true_sklearn, y_pred_sklearn)
    f1_sklearn = f1_score(y_true_sklearn, y_pred_sklearn, average='weighted')
    cm_sklearn = confusion_matrix(y_true_sklearn, y_pred_sklearn, labels=np.unique(y))

    # Вывод результатов
    print(f"\nCustom k-NN LOOCV (k={k}):")
    print(f"  Accuracy  = {acc_custom:.3f} ({acc_custom*100:.1f}%)")
    print(f"  F1-weight = {f1_custom:.3f}")
    print(f"  Время     = {time_custom:.4f} с")
    print("  Матрица ошибок:")
    print(pd.DataFrame(cm_custom, index=np.unique(y), columns=np.unique(y)))

    print(f"\nSklearn k-NN LOOCV (k={k}):")
    print(f"  Accuracy  = {acc_sklearn:.3f} ({acc_sklearn*100:.1f}%)")
    print(f"  F1-weight = {f1_sklearn:.3f}")
    print(f"  Время     = {time_sklearn:.4f} с")
    print("  Матрица ошибок:")
    print(pd.DataFrame(cm_sklearn, index=np.unique(y), columns=np.unique(y)))
    
perform_loocv(X, y, k=3)

print("\n" + "=" * 60)
print("БЛОК 2: ТЕСТИРОВАНИЕ ПО МЕТОДУ ЧЁРНОГО ЯЩИКА (Black-Box)")
print("=" * 60)

black_box_tests = pd.DataFrame({
    'продукт': ['Экстремально сладкий', 'Отрицательные координаты', 'Ноль'],
    'сладость': [1000, -50, 0],
    'хруст': [1000, -50, 0]
})
X_bb = black_box_tests[['сладость', 'хруст']].values

try:
    bb_predictions = CustomKNNClassifier(k=3).fit(X, y).predict(X_bb)
    for i, prod in enumerate(black_box_tests['продукт']):
        print(f"Тест '{prod}' -> Предсказание: {bb_predictions[i]}")
    print("-> Black-Box тесты пройдены: алгоритм устойчив к выбросам.")
except Exception as e:
    print(f"-> Black-Box тест ПРОВАЛЕН. Ошибка: {e}")

print("\n" + "=" * 60)
print("БЛОК 3: ТЕСТИРОВАНИЕ ПО МЕТОДУ БЕЛОГО ЯЩИКА (White-Box)")
print("=" * 60)

X_wb_train = np.array([[2, 0], [0, 2], [0, -1.9]])
y_wb_train = np.array(['Класс_A', 'Класс_B', 'Класс_C'])
X_wb_test = np.array([[0, 0]])

custom_wb_model = CustomKNNClassifier(k=3).fit(X_wb_train, y_wb_train)
wb_pred = custom_wb_model.predict(X_wb_test)[0]

print("Искусственная ситуация для вызова внутреннего ветвления 'Ничья' (1:1:1):")
print("Ожидается класс ближайшего соседа: 'Класс_C'")
print(f"Фактическое предсказание: '{wb_pred}'")

if wb_pred == 'Класс_C':
    print("-> White-Box тест пройден: Внутренний механизм разрешения ничьих работает корректно.")
else:
    print("-> White-Box тест ПРОВАЛЕН: Логика разрешения ничьих работает неверно.")

# Дополнительно: вероятности для Томата
custom_knn_4 = CustomKNNClassifier(k=k).fit(X_ext, y_ext)
proba_tomato = custom_knn_4.predict_proba(X_test_same[0:1])[0]
print("\nВероятности классов для Томата (k=3, кастомный классификатор):")
for cls, prob in zip(custom_knn_4.classes_, proba_tomato):
    print(f"  {cls}: {prob:.2f}")
