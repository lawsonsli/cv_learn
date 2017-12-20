# Author: LiShang
import numpy as np
import networkx as nx
from PIL import Image
import sklearn.mixture as mixture


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

        self.src = int(self.w * self.h)
        self.sink = int(self.w * self.h + 1)

        self.mask_array = np.zeros([self.h, self.w], dtype=np.uint8)
        self.graph = nx.Graph()

        self.fg_gmm = mixture.GaussianMixture(n_components=self.K, init_params='random', warm_start=True, verbose=2)
        self.bg_gmm = mixture.GaussianMixture(n_components=self.K, init_params='random', warm_start=True, verbose=2)

        self.gamma = _gamma
        left_diffs = self.image_array[:, 1:] - self.image_array[:, :-1]
        upleft_diffs = self.image_array[1:, 1:] - self.image_array[:-1, :-1]
        up_diffs = self.image_array[1:, :] - self.image_array[:-1, :]
        upright_diffs = self.image_array[1:, :-1] - self.image_array[:-1, 1:]
        sum_squared = (left_diffs * left_diffs).sum() + (upleft_diffs * upleft_diffs).sum() + \
                      (up_diffs * up_diffs).sum() + (upright_diffs * upright_diffs).sum()
        beta = sum_squared / (
            4 * self.image_array.shape[0] * self.image_array.shape[1] -
            3 * (self.image_array.shape[0] + self.image_array.shape[1]) + 2)
        self.beta = 1 / (2 * beta)

        # r_diff = self.image_array[:, 1:] - self.image_array[:, :-1]
        # d_diff = self.image_array[1:, :] - self.image_array[:-1, :]
        # self.beta = np.sum(np.square(r_diff)) + np.sum(np.square(d_diff))
        # self.beta /= (2 * self.w * self.h - self.w - self.h)
        # self.beta = 0.5 / self.beta

        # init the mask
        self.mask_array[:, :] = self._GC_BGD
        self.mask_array[rect[1]: rect[3], rect[0]: rect[2]] = self._GC_PR_FGD
        # self.show_mask()

    def build_graph(self):
        self.graph.clear()

        self.graph.add_nodes_from(np.array(range(self.w * self.h + 2), dtype=np.float64))

        # add edges to source
        for i in range(self.h):
            edges = np.array([(self.src, i * self.w + j) for j in range(self.w)], dtype=np.float64)
            # weights = -self.bg_gmm.score_samples(self.image_array[i])

            weights = -np.log(np.dot(self.fg_gmm.predict_proba(self.image_array[i]), self.fg_gmm.weights_.T))
            weights = np.reshape(weights, (-1, 1))
            for j in range(self.w):
                if self.mask_array[i, j] == self._GC_BGD:
                    weights[j] = float('inf')

            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

        # add edges to sink
        for i in range(self.h):
            edges = np.array([(i * self.w + j, self.sink) for j in range(self.w)], dtype=np.float64)
            # weights = -self.fg_gmm.score_samples(self.image_array[i])
            weights = -np.log(np.dot(self.bg_gmm.predict_proba(self.image_array[i]), self.bg_gmm.weights_.T))
            weights = np.reshape(weights, (-1, 1))
            for j in range(self.w):
                if self.mask_array[i, j] == self._GC_FGD:
                    weights[j] = float('inf')
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

        # add to right edges
        for i in range(self.h):
            edges = np.array([(i * self.w + j, i * self.w + j + 1) for j in range(self.w - 1)], dtype=np.float64)
            diff = np.array(self.image_array[i, 0: -1] - self.image_array[i, 1:], dtype=np.float64)
            diff = np.sum(np.square(diff), axis=1)
            weights = np.multiply(self.gamma, np.exp(-self.beta * diff))
            weights = np.reshape(weights, (-1, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

        # add to lower edges
        for j in range(self.w):
            edges = np.array([(i * self.w + j, i * self.w + j + self.w) for i in range(self.h - 1)], dtype=np.float64)
            diff = np.array(self.image_array[0: -1, j] - self.image_array[1:, j], dtype=np.float64)
            diff = np.sum(np.square(diff), axis=1)
            weights = np.multiply(self.gamma, np.exp(-self.beta * diff))
            weights = np.reshape(weights, (-1, 1))
            edges = np.hstack((edges, weights))
            self.graph.add_weighted_edges_from(edges)

    def grab_cut(self):

        # iter
        for cc in range(5):
            fg_set = np.where(np.logical_or(self.mask_array == self._GC_FGD, self.mask_array == self._GC_PR_FGD))
            bg_set = np.where(np.logical_or(self.mask_array == self._GC_BGD, self.mask_array == self._GC_PR_BGD))
            # self.fg_gmm = gmm.GaussianMixture(n_components=self.K)
            # self.bg_gmm = gmm.GaussianMixture(n_components=self.K)
            # print(len(self.image_array[fg_set]), len(self.image_array[bg_set]))

            self.fg_gmm = self.fg_gmm.fit(self.image_array[fg_set])
            self.bg_gmm = self.bg_gmm.fit(self.image_array[bg_set])
            # print(self.fg_gmm.means_, self.fg_gmm.weights_)
            # print(self.bg_gmm.means_, self.bg_gmm.weights_)

            prob_array1 = np.zeros([self.h, self.w])
            prob_array2 = np.zeros([self.h, self.w])
            for i in range(self.h):
                for j in range(self.w):
                    prob = np.dot(self.fg_gmm.predict_proba(self.image_array[i, j].reshape(1, -1)), self.fg_gmm.weights_.T)
                    # prob = np.exp(self.fg_gmm.score_samples(self.image_array[i, j].reshape(1, -1)))
                    prob_array1[i, j] = int(255 * prob)
                    prob = np.dot(self.bg_gmm.predict_proba(self.image_array[i, j].reshape(1, -1)), self.bg_gmm.weights_.T)
                    # prob = np.exp(self.fg_gmm.score_samples(self.image_array[i, j].reshape(1, -1)))
                    prob_array2[i, j] = int(255 * prob)

            # Image.fromarray(np.uint8(prob_array1), mode='L').show()
            # Image.fromarray(np.uint8(prob_array2), mode='L').show()

            self.build_graph()
            print(cc, 'build graph done!')

            energy, partition = nx.minimum_cut(self.graph, self.src, self.sink, capacity='weight')
            back, fore = partition
            print(energy)
            print(self.sink in fore, self.src in back)
            fore.remove(self.sink)
            back.remove(self.src)

            for k in fore:
                i, j = int(k // self.w), int(k % self.w)
                if self.mask_array[i, j] == self._GC_PR_BGD:
                    self.mask_array[i, j] = self._GC_PR_FGD
            for k in back:
                i, j = int(k // self.w), int(k % self.w)
                if self.mask_array[i, j] == self._GC_PR_FGD:
                    self.mask_array[i, j] = self._GC_PR_BGD

            print(cc, 'mask changed.')

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

    # gc = GrabCut('./dog2.jpg', (80, 22, 221, 142), 8.0)
    # gc = GrabCut('./cat2.jpg', (42, 34, 255, 239), 8.0)
    gc = GrabCut('./test3.jpg', (19, 7, 242, 200), 8.0)

    gc.grab_cut()
    gc.save_foreground('./result.jpg')