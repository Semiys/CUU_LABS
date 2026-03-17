import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# Вспомогательные функции
def generate_random_cities(n, x_range=(0, 100), y_range=(0, 100)):
    """Генерирует n случайных городов в заданных пределах."""
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]
def distance(p1, p2):
    """Евклидово расстояние между двумя точками."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_wcss(cities, labels, centroids):
    """Сумма квадратов расстояний от точек до центроидов их кластеров."""
    wcss = 0.0
    cities = np.array(cities)
    for i, city in enumerate(cities):
        centroid = centroids[labels[i]]
        wcss += distance(city, centroid) ** 2
    return wcss



# Алгоритмы кластеризации

def algorithmic_clustering(cities, K):
    """
    Детерминированный алгоритм, не похожий на K-средних:
    1. Первый центроид – первый город в списке.
    2. Последующие центроиды выбираются как точки, максимально удалённые
       от уже выбранных центроидов (принцип максимального минимума).
    3. После выбора всех K центроидов – однократное распределение.
    Возвращает метки кластеров и массив центроидов.
    """
    if K <= 0 or not cities:
        return np.array([]), np.array([])

    cities_list = list(cities)
    # Шаг 1: первый центроид
    centroids = [cities_list[0]]

    # Шаг 2: выбираем оставшиеся K-1 центроидов
    for _ in range(1, K):
        max_min_dist = -1
        best_point = None
        for point in cities_list:
            # Пропускаем уже выбранные центроиды
            if any(np.array_equal(point, c) for c in centroids):
                continue
            # Расстояние до ближайшего центроида
            min_dist = min(distance(point, c) for c in centroids)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_point = point
        if best_point is None:
            # Не осталось точек для выбора (все стали центроидами)
            break
        centroids.append(best_point)

    centroids = np.array(centroids)

    # Шаг 3: однократное распределение
    labels = []
    for city in cities_list:
        dists = [distance(city, c) for c in centroids]
        labels.append(np.argmin(dists))

    return np.array(labels), centroids


def kmeans_clustering(cities, K, max_iters=100, tol=1e-4):
    """
    Классический K-средних со случайной инициализацией (ручная реализация).
    """
    centroids = np.array(random.sample(cities, K))
    cities = np.array(cities)
    labels = np.zeros(len(cities), dtype=int)

    iterations = 0
    for _ in range(max_iters):
        iterations += 1
        # Шаг назначения
        for i, city in enumerate(cities):
            dists = [distance(city, c) for c in centroids]
            labels[i] = np.argmin(dists)

        # Шаг обновления центроидов
        new_centroids = []
        for k in range(K):
            cluster_points = cities[labels == k]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        if np.allclose(new_centroids, centroids, atol=tol):
            break
        centroids = new_centroids

    return labels, centroids, iterations


def sklearn_kmeans_clustering(cities, K):
    """
    Использует реализацию KMeans из библиотеки scikit-learn.
    Инициализация по умолчанию – k-means++.
    Возвращает метки, центроиды и количество итераций.
    """
    cities_array = np.array(cities)
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(cities_array)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    iterations = kmeans.n_iter_
    return labels, centroids, iterations



# Запуск одного теста (три метода)

def run_test(cities, K, test_name):
    print(f"\n{'=' * 60}")
    print(f"{test_name}")
    print(f"Количество городов: {len(cities)}, K = {K}")
    print('=' * 60)

    # 1. Алгоритмическое решение (один проход)
    start = time.perf_counter()
    labels_algo, centroids_algo = algorithmic_clustering(cities, K)
    time_algo = time.perf_counter() - start
    wcss_algo = compute_wcss(cities, labels_algo, centroids_algo)

    # 2. Ручная реализация K-средних (случайная инициализация)
    start = time.perf_counter()
    labels_kmeans, centroids_kmeans, iter_kmeans = kmeans_clustering(cities, K)
    time_kmeans = time.perf_counter() - start
    wcss_kmeans = compute_wcss(cities, labels_kmeans, centroids_kmeans)

    # 3. Sklearn KMeans (k-means++ инициализация)
    start = time.perf_counter()
    labels_sk, centroids_sk, iter_sk = sklearn_kmeans_clustering(cities, K)
    time_sk = time.perf_counter() - start
    wcss_sk = compute_wcss(cities, labels_sk, centroids_sk)

    # Вывод результатов
    print("\n{:<35} {:>12} {:>12} {:>15}".format("Метод", "Время (с)", "WCSS", "Итерации"))
    print("-" * 75)
    print("{:<35} {:>12.6f} {:>12.2f} {:>15}".format("Алгоритмический (1 проход)", time_algo, wcss_algo, "—"))
    print("{:<35} {:>12.6f} {:>12.2f} {:>15}".format("Ручной K-средних (случ. иниц.)", time_kmeans, wcss_kmeans, iter_kmeans))
    print("{:<35} {:>12.6f} {:>12.2f} {:>15}".format("Sklearn KMeans (k-means++)", time_sk, wcss_sk, iter_sk))

    # Визуализация (три подграфика)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(test_name, fontsize=16)

    # Цвета для кластеров
    cmap = plt.cm.tab10
    colors = [cmap(i / K) for i in range(K)]

    # Данные для каждого метода
    methods_data = [
        (labels_algo, centroids_algo, "Алгоритмическое (один проход)"),
        (labels_kmeans, centroids_kmeans, f"Ручной K-средних\n{iter_kmeans} итер."),
        (labels_sk, centroids_sk, f"Sklearn KMeans\n{iter_sk} итер.")
    ]

    for ax, (labels, centroids, title) in zip(axes, methods_data):
        ax.set_title(title)
        for i in range(K):
            pts = [cities[j] for j in range(len(cities)) if labels[j] == i]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, color=colors[i], label=f'Кластер {i+1}', alpha=0.6)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Центроиды')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    plt.tight_layout()
    plt.show()



# Основная программа – автоматический прогон трёх тестов

def main():
    # Задаём тестовые наборы
    tests = [
        {
            'name': 'Тест 1: Случайные города',
            'cities': generate_random_cities(int(input("Количество городов N = "))),
            'K': int(input("Количество кластеров K = "))
        },
        {
            'name': 'Тест 2: Три разнесённых кластера',
            'cities': [
                (0, 0), (1, 0), (2, 0),
                (5, 5), (6, 5), (5, 6),
                (10, 10), (11, 10), (10, 11)
            ],
            'K': 3
        },
        {
            'name': 'Тест 3: Один кластер (все точки рядом)',
            'cities': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            'K': 1
        },
        {
            'name': 'Тест 4 (белый ящик): Все точки одинаковы',
            'cities': [(5, 5)] * 10,  # 10 точек в (5,5)
            'K': 2
        },
        {
            'name': 'Тест 5 (белый ящик): Пустой кластер после инициализации',
            'cities': [(0, 0), (1, 0), (0, 1), (1, 1), (10, 10)],
            'K': 3  # третий кластер может оказаться пустым
        },
        {
            'name': 'Тест 6 (чёрный ящик): Пересекающиеся кластеры',
            'cities': [
                # кластер 1
                (2, 2), (3, 2), (2, 3), (3, 3),
                # кластер 2 (перекрывается)
                (2.5, 2.5), (3.5, 2.5), (2.5, 3.5), (3.5, 3.5)
            ],
            'K': 2
        },
        {
            'name': 'Тест 7 (чёрный ящик): Кластеры разной плотности',
            'cities': [
                # плотный кластер (10 точек в малой области)
                (0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1), (0.2, 0),
                (0, 0.2), (0.2, 0.2), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3),
                # разреженный кластер (4 точки на большом расстоянии)
                (8, 8), (9, 9), (8, 9), (9, 8)
            ],
            'K': 2
        },
        {
            'name': 'Тест 8 (чёрный ящик): Кластеры + шум',
            'cities': [
                # два компактных кластера
                (0, 0), (1, 0), (0, 1), (1, 1),
                (5, 5), (6, 5), (5, 6), (6, 6),
                # шумовые точки, равномерно разбросанные
                (2, 2), (3, 4), (4, 2), (3, 3), (7, 2), (8, 3)
            ],
            'K': 2
        },
        {
            'name': 'Тест 9 (чёрный ящик): Один выброс',
            'cities': [(0, 0)] * 8 + [(100, 100)],
            'K': 2
        },
        {
            'name': 'Тест 10 (для метода локтя): Три сгенерированных кластера',
            'cities': [tuple(pt) for pt in make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=42)[0]],
            'K': 3
        }

    ]

    # Последовательно выполняем все тесты
    for test in tests:
        run_test(test['cities'], test['K'], test['name'])
    print("\nДемонстрация метода локтя для набора 'Тест 10'")
    elbow_method(tests[-1]['cities'], max_K=10)


def elbow_method(cities, max_K=10):
    """
    Строит график зависимости WCSS от K для ручного K-средних (можно и для sklearn).
    """
    wcss = []
    K_range = range(1, max_K + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(np.array(cities))
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, 'bo-')
    plt.xlabel('Количество кластеров K')
    plt.ylabel('WCSS')
    plt.title('Метод локтя для выбора K')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Фиксируем seed для воспроизводимости
    random.seed(42)
    np.random.seed(42)
    main()