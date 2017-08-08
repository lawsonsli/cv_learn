import numpy as np


class GaussMixture(object):

    def __init__(self, x_list, centroids, n_component=5, max_iter=10):
        # x_list N * 3
        self.dim = 3
        self.K = n_component
        self.iter = max_iter
        self.N = len(x_list)

        self.means = np.array(centroids, dtype=np.float64) # K * 3
        self.weights = np.zeros([self.K, 1], dtype=np.float64) # K * 1
        self.covariances = np.zeros((self.K, self.dim, self.dim), dtype=np.float64) # K * 3 * 3

        self.counts = np.zeros([self.K], dtype=np.float64)
        self.sums = np.zeros([self.K, self.dim], dtype=np.float64)
        self.prods = np.zeros([self.K, self.dim, self.dim], dtype=np.float64)

        # dist: distance matrix N*K
        x = np.array(x_list)
        x_shift = np.tile(x, (self.K, 1, 1)).transpose((1, 0, 2)) - np.tile(self.means, (self.N, 1, 1))
        dist = np.sum(np.power(x_shift, 2), axis=2)
        label = np.argmin(dist, axis=1)

        self.counts = np.bincount(label)
        for j in range(self.N):
            xj = np.mat(x[j])
            lb = label[j]
            self.sums[lb] += x[j]
            self.prods[lb] += np.dot(xj.T, xj)

        for k in range(self.K):
            cx = x[np.where(label == k)]
            self.weights[k, 0] = float(cx.shape[0]) / self.N
            self.covariances[k, :, :] = np.cov(cx.T)

    def component_list_prob(self, x_list):
        self.N = x_list.shape[0]
        # Gaussian posterior probability
        prob = np.zeros((self.K, self.N))

        x = np.array(x_list) # x matrix N * 3
        for k in range(self.K):
            cov = np.mat(self.covariances[k].copy())
            det_cov = abs(np.linalg.det(cov))

            x_shift = np.mat(x - np.tile(self.means[k], (self.N, 1)))
            inv = np.linalg.pinv(cov)

            e = np.array([(x_shift[i, :] * inv * x_shift[i, :].T)[0, 0] for i in range(self.N)])
            # e = np.diag(x_shift * inv * x_shift.T)
            c = 1.0 / (np.sqrt(det_cov) * np.power(2 * np.pi, self.dim / 2))

            prob[k] = c * np.exp(-0.5 * e)
        # prob is a K*N matrix
        return prob

    def list_prob(self, x_list):
        component_prob = self.component_list_prob(x_list)
        list_prob = np.dot(self.weights.T, component_prob)[0]
        return list_prob

    def _likely_component(self, pixel):
        # component_prob = self.weights * self.component_list_prob(np.array([pixel]))
        component_prob = self.component_list_prob(np.array([pixel]))
        component_prob = component_prob[:, 0].T
        return np.argmax(component_prob)

    def _reset(self):
        self.counts = np.zeros([self.K], dtype=np.float64)
        self.sums = np.zeros([self.K, self.dim], dtype=np.float64)
        self.prods = np.zeros([self.K, self.dim, self.dim], dtype=np.float64)

    def _add_pixel(self, pixel, k):
        # pixel [p1, p2, p3] np array
        p = np.mat(pixel.copy())
        self.sums[k] += pixel
        self.prods[k] += np.dot(p.T, p)
        self.counts[k] += 1

    # def _em_learn(self):
    #     t_counts = np.sum(self.counts)
    #     for k in range(self.K):
    #         self.weights[k] = self.counts[k] / t_counts
    #         if self.counts[k] > 0:
    #             self.means[k] = self.sums[k] / self.counts[k]
    #             miu = np.mat(self.means[k].copy())
    #             self.covariances[k] = self.prods[k] / self.counts[k] - np.dot(miu.T, miu)
    #
    #         # cov_det = np.linalg.det(self.covariances[k])
    #         # while cov_det <= 0:
    #         #     self.covariances[k] += np.diag([0.01, 0.01, 0.01])
    #         #     cov_det = np.linalg.det(self.covariances[k])

    def em_learn_params(self, x_list):
        for i in range(self.iter):
            self._reset()

            for pixel in x_list:
                k = self._likely_component(pixel)
                self._add_pixel(pixel, k)

            t_counts = np.sum(self.counts)
            for k in range(self.K):
                self.weights[k] = self.counts[k] / t_counts
                if self.counts[k] > 0:
                    self.means[k] = self.sums[k] / self.counts[k]
                    miu = np.mat(self.means[k].copy())
                    self.covariances[k] = self.prods[k] / self.counts[k] - np.dot(miu.T, miu)
                    # while np.linalg.det(self.covariances[k]) <= 0:
                    #     self.covariances[k] += np.diag([0.01, 0.01, 0.01])


if __name__ == '__main__':
    xx_list = [[1.0, 3.0, 4],
                [2.0, 3.0, 4],
               [2.0, 3.5, 4],
              [3, 4, 5],
               [3.5, 4, 5],
              [4, 5, 6],
               [4, 5.5, 6],
              [3, 6, 7]]
    center = np.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])

    gm = GaussMixture(xx_list, center)

    print(gm.means)
    print(gm.list_prob(np.array([[3.0, 4.5, 4.5],
                                 [5.5, 5.5, 5.5]])))

