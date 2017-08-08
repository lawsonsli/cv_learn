# Author: Li Shang
import numpy as np
import networkx as nx
from PIL import Image
import sklearn.mixture.gaussian_mixture as gmm


class GrabCut(object):

    def __init__(self, image_path, rect, _gamma=2.0):
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

        self.fg_gmm = gmm.GaussianMixture(n_components=self.K)
        self.bg_gmm = gmm.GaussianMixture(n_components=self.K)

        self.gamma = _gamma
        r_diff = self.image_array[:, 1:] - self.image_array[:, :-1]
        d_diff = self.image_array[1:, :] - self.image_array[:-1, :]
        self.beta = np.sum(np.square(r_diff)) + np.sum(np.square(d_diff))
        self.beta /= (2 * self.w * self.h - self.w - self.h)
        self.beta = 0.5 / self.beta

        # init the mask
        self.mask_array[:, :] = self._GC_BGD
        self.mask_array[rect[1]: rect[3], rect[0]: rect[2]] = self._GC_PR_FGD
        self.show_mask()

    def build_graph(self):
        self.graph.clear()

        self.graph.add_nodes_from(range(self.w * self.h + 2))

        # add edges to source
        for i in range(self.h):
            edges = np.array([(self.src, i * self.w + j) for j in range(self.w)])
            weights = -self.bg_gmm.score_samples(self.image_array[i])
            edges = np.hstack((edges, np.reshape(weights, (-1, 1))))
            self.graph.add_weighted_edges_from(edges)

        # add edges to sink
        for i in range(self.h):
            edges = np.array([(i * self.w + j, self.sink) for j in range(self.w)])
            weights = -self.fg_gmm.score_samples(self.image_array[i])
            edges = np.hstack((edges, np.reshape(weights, (-1, 1))))
            self.graph.add_weighted_edges_from(edges)

        # add to right edges
        for i in range(self.h):
            edges = np.array([(i * self.w + j, i * self.w + j + 1) for j in range(self.w - 1)])
            diff = np.array(self.image_array[i, 0: -1] - self.image_array[i, 1:], dtype=np.float64)
            diff = np.sum(np.square(diff), axis=1)
            weights = np.multiply(self.gamma, np.exp(-self.beta * diff))
            weights = np.reshape(weights, (-1, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

        # add to lower edges
        for j in range(self.w):
            edges = [(i * self.w + j, i * self.w + j + self.w) for i in range(self.h - 1)]
            diff = np.array(self.image_array[0: -1, j] - self.image_array[1:, j], dtype=np.float64)
            diff = np.sum(np.square(diff), axis=1)
            weights = np.multiply(self.gamma, np.exp(-self.beta * diff))
            weights = np.reshape(weights, (-1, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

    def grab_cut(self):

        # iter
        for cc in range(10):
            fg_set = np.where(np.logical_or(self.mask_array == self._GC_FGD, self.mask_array == self._GC_PR_FGD))
            bg_set = np.where(np.logical_or(self.mask_array == self._GC_BGD, self.mask_array == self._GC_PR_BGD))
            self.fg_gmm = gmm.GaussianMixture(n_components=self.K)
            self.bg_gmm = gmm.GaussianMixture(n_components=self.K)
            self.fg_gmm.fit(self.image_array[fg_set])
            self.bg_gmm.fit(self.image_array[bg_set])

            self.build_graph()
            print(cc, 'build graph done!')

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
            # if cc % 5 == 0:
            # self.show_mask()
            self.save_foreground(str(cc) + '.jpg')

    def show_mask(self):
        mask = self.image_array.copy()
        mask[
            np.where(np.logical_or(self.mask_array == self._GC_BGD, self.mask_array == self._GC_PR_BGD))
        ] = np.array([255, 255, 255])
        Image.fromarray(np.uint8(mask)).show()

    def save_foreground(self, fp):
        mask = self.image_array.copy()
        mask[
            np.where(np.logical_or(self.mask_array == self._GC_BGD, self.mask_array == self._GC_PR_BGD))
        ] = np.array([255, 255, 255])
        Image.fromarray(np.uint8(mask)).save(fp)

if __name__ == '__main__':
    # gc = GrabCut('./cat2.jpg', (17, 5, 89, 50), 2.0)
    # gc = GrabCut('./cat.jpg', (68, 20, 366, 200), 18.0)
    # gc = GrabCut('./bull.jpg', (82, 74, 454, 209), 18.0)
    gc = GrabCut('./test2.jpg', (10, 10, 186, 125), 2.5)

    gc.grab_cut()
    gc.save_foreground('./result.jpg')
