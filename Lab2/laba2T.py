import numpy as np
import matplotlib.pyplot as plt
import time
import random



# Генерация данных и вспомогательные функции
def generate_cities(n_cities=100, x_range=(0, 100), y_range=(0, 100)):
    """Генерирует список городов со случайными координатами."""
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n_cities)]


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



# Алгоритмическое решение (один проход)
def algorithmic_clustering(cities, K):
    """
    Выбирает K случайных городов как центроиды и однократно распределяет
    все города по ближайшим центроидам.
    Возвращает метки кластеров и массив центроидов.
    """
    # Шаг 1: случайный выбор центроидов среди городов
    centroids = random.sample(cities, K)

    # Шаг 2: для каждого города найти ближайший центроид
    labels = []
    for city in cities:
        dists = [distance(city, c) for c in centroids]
        labels.append(np.argmin(dists))

    return np.array(labels), np.array(centroids)



# Метод K-средних (итеративный)

def kmeans_clustering(cities, K, max_iters=100, tol=1e-4):
    centroids = np.array(random.sample(cities, K))
    cities = np.array(cities)
    labels = np.zeros(len(cities), dtype=int)

    iterations = 0  # <--- счётчик
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



# Основная программа

def main():
    # Параметры эксперимента
    n_cities = 200

    while True:
        try:
            K = int(input("Введите количество кластеров K: "))
            if K <= 0:
                print("K должно быть положительным числом. Повторите ввод.")
                continue
            if K > n_cities:
                print(f"K не может превышать количество городов ({n_cities}). Повторите ввод.")
                continue
            break
        except ValueError:
            print("Ошибка: введите целое число.")
    random.seed(42)  # для воспроизводимости
    np.random.seed(42)

    # Генерация городов
    cities = generate_cities(n_cities)

    # Алгоритмическое решение
    start = time.perf_counter()
    labels_algo, centroids_algo = algorithmic_clustering(cities, K)
    time_algo = time.perf_counter() - start
    wcss_algo = compute_wcss(cities, labels_algo, centroids_algo)

    # Метод K-средних
    start = time.perf_counter()
    labels_kmeans, centroids_kmeans, iterations = kmeans_clustering(cities, K)
    time_kmeans = time.perf_counter() - start
    wcss_kmeans = compute_wcss(cities, labels_kmeans, centroids_kmeans)

    # Вывод результатов
    print("=" * 50)
    print(f"Количество городов: {n_cities}, число кластеров K = {K}")
    print("-" * 50)
    print("Алгоритмическое решение (один проход):")
    print(f"  Время выполнения: {time_algo:.6f} с")
    print(f"  WCSS: {wcss_algo:.2f}")
    print("-" * 50)
    print("Метод K-средних (итеративный):")
    print(f"  Время выполнения: {time_kmeans:.6f} с")
    print(f"  WCSS: {wcss_kmeans:.2f}")
    print(f"  Число итераций: {iterations}")
    print("=" * 50)

    # Визуализация
    plt.figure(figsize=(14, 6))

    # Алгоритмическое решение
    plt.subplot(1, 2, 1)
    plt.title("Алгоритмическое решение (один проход)")

    # Генерируем K цветов из палитры tab10
    cmap = plt.cm.tab10  # можно использовать plt.colormaps['tab10']
    colors = [cmap(i / K) for i in range(K)]  # равномерно распределяем по диапазону

    for i in range(K):
        pts = [cities[j] for j in range(n_cities) if labels_algo[j] == i]
        if pts:
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, color=colors[i], label=f'Кластер {i + 1}', alpha=0.6)
    plt.scatter(centroids_algo[:, 0], centroids_algo[:, 1], c='black', marker='X', s=200, label='Центроиды')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # K-средних
    plt.subplot(1, 2, 2)
    plt.title("Метод K-средних")
    for i in range(K):
        pts = [cities[j] for j in range(n_cities) if labels_kmeans[j] == i]
        if pts:
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, color=colors[i], label=f'Кластер {i + 1}', alpha=0.6)
    plt.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], c='black', marker='X', s=200, label='Центроиды')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()