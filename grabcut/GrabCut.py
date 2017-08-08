# Author: Li Shang
import numpy as np
import networkx as nx
import time
from PIL import Image
from sklearn.cluster import KMeans
from GaussMixture import GaussMixture


def timeit(func):
    def wrapper(*args, **kw):
        time1 = time.time()
        result = func(*args, **kw)
        time2 = time.time()
        print(func.__name__, time2-time1)
        return result
    return wrapper


class GrabCut(object):

    def __init__(self, image_path, rect):
        self._GC_BGD = 0  # {'color' : BLACK, 'val' : 0}
        self._GC_FGD = 1  # {'color' : WHITE, 'val' : 1}
        self._GC_PR_BGD = 2  # {'color' : GREEN, 'val' : 2}
        self._GC_PR_FGD = 3  # {'color' : RED, 'val' : 3}

        self.K = 5

        self.image = Image.open(image_path)
        self.image_array = np.array(self.image, dtype=np.float64)
        self.h, self.w, channel = self.image_array.shape

        self.src = self.w * self.h
        self.sink = self.w * self.h + 1

        self.mask_array = np.zeros([self.h, self.w], dtype=np.uint8)
        self.graph = nx.Graph()

        self.fg_gmm = None
        self.bg_gmm = None

        self.gamma = 20.0
        r_diff = self.image_array[:, 1:] - self.image_array[:, :-1]
        d_diff = self.image_array[1:, :] - self.image_array[:-1, :]
        self.beta = np.sum(np.square(r_diff)) + np.sum(np.square(d_diff))
        self.beta /= (2 * self.w * self.h - self.w - self.h)
        self.beta = 0.5 / self.beta

        # init the mask
        self.mask_array[:, :] = self._GC_BGD
        self.mask_array[rect[1]: rect[3], rect[0]: rect[2]] = self._GC_PR_FGD
        self.show_mask()

        # init GMM with KMeans
        bg_set = np.where(np.logical_or(self.mask_array == self._GC_BGD, self.mask_array == self._GC_PR_BGD))
        fg_set = np.where(np.logical_or(self.mask_array == self._GC_FGD, self.mask_array == self._GC_PR_FGD))
        bg_pixels = self.image_array[bg_set]
        fg_pixels = self.image_array[fg_set]

        fg_k_means = KMeans(n_clusters=5, max_iter=10)
        fg_k_means.fit(fg_pixels)
        bg_k_means = KMeans(n_clusters=5, max_iter=10)
        bg_k_means.fit(bg_pixels)

        self.fg_gmm = GaussMixture(fg_pixels, fg_k_means.cluster_centers_, n_component=self.K)
        self.bg_gmm = GaussMixture(bg_pixels, bg_k_means.cluster_centers_, n_component=self.K)

        del fg_set, fg_pixels, fg_k_means
        del bg_set, bg_pixels, bg_k_means

    def build_graph(self):
        self.graph.clear()

        self.graph.add_nodes_from(range(self.w * self.h + 2))

        # add edges to source
        for i in range(self.h):
            edges = np.array([(self.src, i * self.w + j) for j in range(self.w)])
            weights = np.reshape(-np.log(self.bg_gmm.list_prob(self.image_array[i])), (self.w, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

        # add edges to sink
        for i in range(self.h):
            edges = np.array([(i * self.w + j, self.sink) for j in range(self.w)])
            weights = np.reshape(-np.log(self.fg_gmm.list_prob(self.image_array[i])), (self.w, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

        # add to right edges
        for i in range(self.h):
            edges = np.array([(i * self.w + j, i * self.w + j + 1) for j in range(self.w - 1)])
            diff = np.array(self.image_array[i, 0: -1] - self.image_array[i, 1:], dtype=np.float)
            diff = np.sum(np.square(diff), axis=1)
            weights = np.multiply(self.gamma, np.exp(-self.beta * diff))
            weights = np.reshape(weights, (-1, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

        # add to lower edges
        for j in range(self.w):
            edges = [(i * self.w + j, i * self.w + j + self.w) for i in range(self.h - 1)]
            diff = np.array(self.image_array[0: -1, j] - self.image_array[1:, j], dtype=np.float)
            diff = np.sum(np.square(diff), axis=1)
            weights = np.multiply(self.gamma, np.exp(-self.beta * diff))
            weights = np.reshape(weights, (-1, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

    def grab_cut(self):

        # iter
        for cc in range(10):
            self.build_graph()
            print('build graph done!')

            energy, partition = nx.minimum_cut(self.graph, self.src, self.sink, capacity='weight')
            fore, back = partition
            print(energy)
            fore.remove(self.src)
            back.remove(self.sink)
            fore = [(int(k // self.w), int(k % self.w)) for k in fore]
            back = [(int(k // self.w), int(k % self.w)) for k in back]
            for i, j in fore:
                if self.mask_array[i, j] == self._GC_PR_BGD:
                    self.mask_array[i, j] = self._GC_PR_FGD
            for i, j in back:
                if self.mask_array[i, j] == self._GC_PR_FGD:
                    self.mask_array[i, j] = self._GC_PR_BGD
            print(cc, 'mask changed.')
            self.show_mask()

            fg_set = np.where(np.logical_or(self.mask_array == self._GC_FGD, self.mask_array == self._GC_PR_FGD))
            bg_set = np.where(np.logical_or(self.mask_array == self._GC_BGD, self.mask_array == self._GC_PR_BGD))
            self.fg_gmm.em_learn_params(self.image_array[fg_set])
            self.bg_gmm.em_learn_params(self.image_array[bg_set])

    def show_mask(self):
        mask = self.image_array.copy()
        mask[
            np.where(np.logical_or(self.mask_array == self._GC_BGD, self.mask_array == self._GC_PR_BGD))
        ] = np.array([255, 255, 255])
        Image.fromarray(np.uint8(mask)).show()


if __name__ == '__main__':
    # gc = GrabCut('./cat2.jpg', (17, 5, 89, 50))
    gc = GrabCut('./cat.jpg', (68, 20, 366, 200))
    # gc = GrabCut('./bull.jpg', (82, 74, 454, 209))

    gc.grab_cut()
