from random import choice
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


def generate_random_dots(amount: int) -> np.ndarray:
    '''
    Функция генерации случайных групп точек, используется метод из sklearn.datasets.make_blobs

    amount - количество точек
    '''
    research_field, _ = make_blobs(n_samples = amount)
    return research_field


class KMeans():
    '''
    Класс реализации метода k - средних где:

    k - число кластеров
    dots - точки в формате numpy.ndarray
    distance_func - название функции определения расстояния (доступны функции Евклидова расстояния 'euqlid' и Манхэттена 'manhattan')
    max_iter - число максимальных итераций
    '''
    
    k: int
    dots: np.ndarray
    max_iter: int

    final_clusters: np.ndarray

    def __init__(self, k: int, dots: np.ndarray, distance_func: str, max_iter: int = 100) -> None:
        self.k        = k
        self.dots     = dots
        self.max_iter = max_iter

        if distance_func   == "euqlid":
            self.dist_func =  KMeans.euqlid
        elif distance_func == "manhattan":
            self.dist_func =  KMeans.manhattan
        else:
            raise NameError("wrong distance function!")


    def run(self):
        '''
        Запуск алгоритма K-Means
        '''
        centroids = self.getSeeds()

        for _ in range(self.max_iter):
            clusters = self.clusterize(centroids)
            prev_centroids = centroids
            centroids = self.compute_means(clusters)
            diff = prev_centroids - centroids
            if not diff.any():
                self.final_clusters = clusters
                return
        
        self.final_clusters = clusters


    def visualize(self):
        '''
        Визуализация исходного расположения точек и результата кластеризации
        '''
        fig = plt.figure(figsize=(8, 5))

        axes_orig = fig.add_subplot(1, 2, 1)
        axes_orig.scatter(self.dots[:, 0], self.dots[:, 1], marker="o")
        axes_orig.set_title("Dots placed")

        axes_orig = fig.add_subplot(1, 2, 2)
        axes_orig.scatter(self.dots[:, 0], self.dots[:, 1], c=self.final_clusters)
        axes_orig.set_title("K-Means result")

        plt.show()


    def getSeeds(self) -> np.ndarray:
        '''
        Функция получения начальных центроидов (сидов)
        '''
        m, n = np.shape(self.dots)

        centroids = np.empty((self.k, n))
        for i in range(self.k):
            centroids[i] =  self.dots[np.random.choice(range(m))] 
        return centroids
        

    def compute_means(self, cluster_idx):
        '''
        Функция нахождения центроидов
        '''
        _, n = np.shape(self.dots)
        centroids = np.empty((self.k, n))

        for i in range(self.k):
            points = self.dots[cluster_idx == i]
            centroids[i] = np.mean(points, axis=0)
        return centroids


    def clusterize(self, centroids):
        '''
        Функция кластеризации
        '''
        m, _ = np.shape(self.dots)
        cluster_idx = np.empty(m)
        for i in range(m):
            cluster_idx[i] = self.closest_centroid(self.dots[i], centroids)
        return cluster_idx


    def closest_centroid(self, dots, centroids):
        '''
        Функция определения ближайшего центроида для каждой из точек
        '''
        distances = np.empty(self.k)
        for i in range(self.k):
            distances[i] = self.dist_func(centroids[i], dots)
        return np.argmin(distances)


    @staticmethod
    def manhattan(dot_1, dot_2) -> float:
        '''
        Функция определения манхеттенского расстояния между точками
        '''
        return np.sum(np.abs(dot_1 - dot_2))


    @staticmethod
    def euqlid(dot_1, dot_2) -> float:
        '''
        Функция определения евклидового расстояния между точками
        '''
        return np.sqrt(np.sum(np.square(dot_1 - dot_2)))


Ks = KMeans(k = 2, dots = generate_random_dots(130), distance_func="manhattan")
Ks.run()
Ks.visualize()
