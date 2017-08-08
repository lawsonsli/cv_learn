from PIL import Image
import math
import numpy as np
import networkx
import cv2


class GraphCut(object):
    
    def __init__(self, image_path, foreground, background):
        # undirected graph
        self.graph = networkx.Graph()
        # directed flow
        self.flow = networkx.DiGraph()
        self.unlabeled_nodes = set()
        self.min_cut = set()

        img = Image.open(image_path).convert('L')
        fore_img = np.array(img.crop(foreground))
        back_img = np.array(img.crop(background))

        # self.img.show()
        self.img_array = np.array(img)
        # w, h = self.img_array.shape
        # for i in range(w):
        #     self.img_array[i] /= 256
        # print(self.img_array)
        self.fore_mean = np.mean(fore_img)
        self.back_mean = np.mean(back_img)

        # self.fore_hist = cv2.calcHist([fore_img], [0], None, [256], [0, 256])
        # self.back_hist = cv2.calcHist([back_img], [0], None, [256], [0, 256])
        # self.fore_hist /= sum(self.fore_hist)
        # self.back_hist /= sum(self.back_hist)

    def _diff(self, a, b):
        sigma = 20000
        kappa = 2.0
        y1 = float(self.img_array[a[0]][a[1]])
        y2 = float(self.img_array[b[0]][b[1]])
        return kappa * math.exp(-1.0 * (y1 - y2)**2 / sigma)

    def _conf(self, a, st):
        pixel = int(self.img_array[a[0]][a[1]])
        f = -math.log(abs(pixel - self.fore_mean) / (abs(pixel - self.fore_mean) + abs(pixel - self.back_mean)))
        b = -math.log(abs(pixel - self.back_mean) / (abs(pixel - self.fore_mean) + abs(pixel - self.back_mean)))
        if st == 'source':
            if b > f:
                return (b - f) / (f + b)
            else:
                return 0.0
        if st == 'sink':
            if b < f:
                return (f - b) / (f + b)
            else:
                return 0

    def build_graph(self):
        
        w, h = self.img_array.shape
        print(w, h)

        # add source and sink
        self.graph.add_node('source')
        self.graph.add_node('sink')

        for i in range(w):
            for j in range(h):
                self.graph.add_node((i, j))
                self.flow.add_node((i, j))
        for i in range(w):
            for j in range(h):
                self.graph.add_edge('source', (i, j), weight=self._conf((i, j), 'source'))
                self.graph.add_edge('sink', (i, j), weight=self._conf((i, j), 'sink'))
                self.flow.add_edge('source', (i, j), weight=0.0)
                self.flow.add_edge((i, j), 'source', weight=0.0)
                self.flow.add_edge('sink', (i, j), weight=0.0)
                self.flow.add_edge((i, j), 'sink', weight=0.0)
        for i in range(w):
            for j in range(h):
                if j < h - 1:
                    self.graph.add_edge((i, j), (i, j + 1), weight=self._diff((i, j), (i, j + 1)))
                    self.flow.add_edge((i, j), (i, j + 1), weight=0.0)
                    self.flow.add_edge((i, j + 1), (i, j), weight=0.0)
                if i < w - 1:
                    self.graph.add_edge((i, j), (i + 1, j), weight=self._diff((i, j), (i + 1, j)))
                    self.flow.add_edge((i, j), (i + 1, j), weight=0.0)
                    self.flow.add_edge((i + 1, j), (i, j), weight=0.0)

    def max_flow(self):
        count = 1
        # Ford-Fulkerson algorithm for undirected graph
        while True:
            print(count, end=' ')
            count += 1
            # bfs to find a augmenting chain
            augmenting_chain = False

            marked_nodes = list()
            marked_nodes.append('source')
            unmarked_nodes = set(self.graph.nodes())
            unmarked_nodes.remove('source')

            last_node = dict()
            last_node_forward = dict()
            delta = dict()
            last_node['source'] = '#'
            delta['source'] = float('inf')

            while len(marked_nodes) > 0:
                i = marked_nodes[0]
                marked_nodes.pop(0)
                linked_edges = self.graph.edge[i]
                for j in linked_edges:
                    if j not in unmarked_nodes:
                        continue
                    # print(out_edges[j])
                    if self.flow.edge[i][j]['weight'] < linked_edges[j]['weight']:
                        last_node[j] = i
                        last_node_forward[j] = True
                        delta[j] = min(delta[i], linked_edges[j]['weight'] - self.flow.edge[i][j]['weight'])
                        marked_nodes.append(j)
                        if j in unmarked_nodes:
                            unmarked_nodes.remove(j)

                        if j == 'sink':  # find augment chain
                            augmenting_chain = True
                            break

                    if self.flow.edge[j][i]['weight'] > 0:
                        last_node[j] = i
                        last_node_forward[j] = False
                        delta[j] = min(delta[i], self.flow.edge[j][i]['weight'])
                        marked_nodes.append(j)
                        if j in unmarked_nodes:
                            unmarked_nodes.remove(j)
                        if j == 'sink':  # find augment chain
                            augmenting_chain = True
                            break

                if augmenting_chain:  # find augment chain
                    break

            if not augmenting_chain:
                self.unlabeled_nodes = unmarked_nodes
                self.unlabeled_nodes.remove('sink')
                break

            # w, h = self.img_array.shape
            # marked = set(zip(range(w), range(h))) - unmarked_nodes
            # for i in marked:
            #     self.img_array[i[0]][i[1]] = 255
            # Image.fromarray(self.img_array).show()

            delta_flow = delta['sink']
            j = 'sink'
            print(j, end=' ')
            while last_node[j] != '#':
                i = last_node[j]
                print(i, end=' ')
                if last_node_forward[j]:
                    self.flow.edge[i][j]['weight'] += delta_flow
                else:
                    self.flow.edge[j][i]['weight'] -= delta_flow
                j = i

            print('')
        return self.flow.edges()


if __name__ == '__main__':
    img_seg = GraphCut('./input2.jpg', (35, 30, 70, 60), (0, 0, 10, 10))

    img_seg.build_graph()
    img_seg.max_flow()

    print('\n')
    w, h = img_seg.img_array.shape
    marked = set(zip(range(w), range(h))) - img_seg.unlabeled_nodes
    print(len(marked))

    # for i in marked:
    #     img_seg.img_array[i[0]][i[1]] = 255

    # labels = set()
    # for u, v, d in img_seg.flow.edges(data=True):
    #     if d['weight'] > 0:
    #         if u == 'source':
    #             labels.add(v)
    #         if v == 'source':
    #             labels.add(u)

    augmenting_chain = False
    visit_queue = list()
    visit_queue.append('source')
    marked_nodes = set()

    last_node = dict()
    last_node_forward = dict()
    delta = dict()
    last_node['source'] = '#'
    delta['source'] = float('inf')

    while len(visit_queue) > 0:
        i = visit_queue[0]
        visit_queue.pop(0)
        linked_edges = img_seg.graph.edge[i]
        for j in linked_edges:
            if j in marked_nodes:
                continue
            # print(out_edges[j])
            if img_seg.flow.edge[i][j]['weight'] < linked_edges[j]['weight']:
                last_node[j] = i
                last_node_forward[j] = True
                delta[j] = min(delta[i], linked_edges[j]['weight'] - img_seg.flow.edge[i][j]['weight'])
                visit_queue.append(j)
                marked_nodes.add(j)

                if j == 'sink':  # find augment chain
                    augmenting_chain = True
                    break

            if img_seg.flow.edge[j][i]['weight'] > 0:
                last_node[j] = i
                last_node_forward[j] = False
                delta[j] = min(delta[i], img_seg.flow.edge[j][i]['weight'])
                visit_queue.append(j)
                marked_nodes.add(j)

                if j == 'sink':  # find augment chain
                    augmenting_chain = True
                    break

        if augmenting_chain:  # find augment chain
            break

    print(len(marked_nodes))
    if 'source' in marked_nodes:
        marked_nodes.remove('source')

    # w, h = img_seg.img_array.shape
    for i in marked_nodes:
        img_seg.img_array[i[0]][i[1]] = 255

    img = Image.fromarray(img_seg.img_array)
    img.show()

