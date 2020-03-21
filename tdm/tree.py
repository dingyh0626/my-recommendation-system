from sklearn.cluster import KMeans
import numpy as np
import random
import multiprocessing as mp
# import prefetch_generator



class TreeNode(object):
    def __init__(self):
        self.id = -1
        self.parent = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.all_items = None


class Tree(object):
    def __init__(self):
        self.root: TreeNode = None
        self.levels = []
        self.leaves = []
        self.n_nodes = 0
        self.leaf_map = {}
        self.c = None
        self.height = 0

    def initialize_tree(self, all_items):
        '''

        :param all_items: (n_examples,)
        :return:
        '''
        self.height = int(np.ceil(np.log2(len(all_items))) + 1)
        self.root: TreeNode = None
        self.leaf_map = {}
        self.levels = [[] for _ in range(self.height)]
        self.leaves = []
        self.n_nodes = 0
        self.root = self.__initialize_tree(all_items)
        self.set_id()

    def cluster_tree(self, all_items, item_embed):
        '''
        :param all_items: (n_examples,)
        :param item_embed: (n_examples, m_features)
        :return:
        '''
        self.height = int(np.ceil(np.log2(len(all_items))) + 1)
        self.root: TreeNode = None
        self.leaf_map = {}
        self.levels = [[] for _ in range(self.height)]
        self.leaves = []
        self.n_nodes = 0
        self.root = self.__build_tree(all_items, item_embed)
        self.set_id()

    def __initialize_tree(self, all_items, level=0):
        if len(all_items) == 0:
            return None
        root = TreeNode()
        self.levels[level].append(root)
        root.all_items = list(all_items)
        if len(all_items) == 1:
            if level == self.height - 1:
                root.is_leaf = True
                self.leaves.append(root)
                self.leaf_map[all_items[0]] = root
                return root
            else:
                left = self.__initialize_tree(all_items, level=level + 1)
                left.parent = root
                root.left = left
                return root
        else:
            mid = len(all_items) // 2
            left = self.__initialize_tree(all_items[:mid], level=level + 1)
            right = self.__initialize_tree(all_items[mid:], level=level + 1)
            left.parent = root
            right.parent = root
            root.left = left
            root.right = right
            return root

    # @staticmethod
    # def __cluster_items(all_items, item_embed):
    #     item_embed = item_embed[all_items]
    #     clf = KMeans(2, random_state=2020)
    #     dist = clf.fit_transform(item_embed)
    #     dist_diff = dist[:, 0] - dist[:, 1]
    #     rank = np.argsort(dist_diff)
    #     mid = len(item_embed) // 2
    #     left = all_items[rank[:mid]]
    #     right = all_items[rank[mid:]]
    #     return left, right

    @staticmethod
    def __cluster_items(item_embed):
        # index = np.arange(len(item_embed))
        # data = item_embed[index]
        kmeans = KMeans(n_clusters=2, random_state=2020).fit(item_embed)
        labels = kmeans.labels_
        left_index = np.where(labels == 0)[0]
        right_index = np.where(labels == 1)[0]
        # left_index = index[l_i]
        # right_index = index[r_i]
        if len(right_index) - len(left_index) > 1:
            distances = kmeans.transform(item_embed[right_index])[:, 1]
            rank = np.argsort(distances)[::-1]
            idx = np.concatenate((left_index, right_index[rank]))
            mid = len(idx) // 2
            left_index = idx[:mid]
            right_index = idx[mid:]

            # left_index, right_index = Tree.rebalance(
            #     left_index, right_index, distances[:, 1])
        elif len(left_index) - len(right_index) > 1:
            distances = kmeans.transform(item_embed[left_index])[:, 0]
            rank = np.argsort(distances)
            idx = np.concatenate((left_index[rank], right_index))
            mid = len(idx) // 2
            left_index = idx[:mid]
            right_index = idx[mid:]
            # left_index, right_index = Tree.rebalance(
            #     right_index, left_index, distances[:, 0])

        return left_index, right_index, kmeans.cluster_centers_[0], kmeans.cluster_centers_[1]

    def __build_tree(self, all_items, item_embed, level=0):
        if len(all_items) == 0:
            return None
        root = TreeNode()
        self.levels[level].append(root)
        root.all_items = list(all_items)
        if len(all_items) == 1:
            if level == self.height - 1:
                root.is_leaf = True
                self.leaves.append(root)
                self.leaf_map[all_items[0]] = root
                return root
            else:
                left = self.__build_tree(all_items, item_embed, level=level + 1)
                left.parent = root
                root.left = left
                return root
        else:
            left_index, right_index, left_c, right_c = self.__cluster_items(item_embed)
            left = self.__build_tree(all_items[left_index], item_embed[left_index], level=level + 1)
            right = self.__build_tree(all_items[right_index], item_embed[right_index], level=level + 1)
            left.parent = root
            right.parent = root
            root.left = left
            left.c = left_c
            root.right = right
            right.c = right_c
            return root

    def set_id(self):
        id = len(self.root.all_items)
        for level in self.levels:
            for node in level:
                if node.is_leaf:
                    node.id = node.all_items[0]
                else:
                    id += 1
                    node.id = id
        self.n_nodes = id

    @property
    def n_levels(self):
        return len(self.levels)

    def get_parent_nodes(self, item_id):
        node = self.leaf_map.get(item_id, None)
        if node is None:
            raise ValueError('Wrong item ID {}!'.format(item_id))
        parent_nodes = []
        while node:
            parent_nodes.append(node.id)
            node = node.parent
        return parent_nodes

    def generate_dataset(self, positive_item_id, excluded_items=None, neg_samples=600):
        positive_nodes = set()
        positive_nodes.update(self.get_parent_nodes(positive_item_id))
        negative_nodes = set()
        if excluded_items is None:
            excluded_items = set()
        else:
            excluded_items = set(excluded_items)
        while len(negative_nodes) < neg_samples:
            for i in range(len(self.levels) - 1, -1, -1):
                if i < 1:
                    continue
                nodes = self.levels[i]
                for node in random.sample(nodes, k=min(len(nodes), i)):
                    # if node.id not in positive_nodes:
                    if len(excluded_items.intersection(node.all_items)) == 0:
                       negative_nodes.add(node.id)
                    if len(negative_nodes) == neg_samples:
                        break
                if len(negative_nodes) == neg_samples:
                    break
        return list(positive_nodes), list(negative_nodes)





if __name__ == '__main__':
    all_items = np.arange(1, 1048, dtype=np.int)

    # all_items = list(np.load('../datasets/movieSorted.npy'))
    tree = Tree()
    item_embed = np.random.normal(0, 1, (all_items.shape[0], 20))
    tree.cluster_tree(all_items, item_embed)
    # tree.initialize_tree(all_items)
    # print(len(tree.levels))
    # print(np.log2(len(all_items)))
    for level in tree.levels:
        # print(len(level))
        print(len([n for n in level if n.is_leaf]))
    # print(tree.n_nodes)
    # tree.generate_dataset(5, [1,2,3])
    # n_nodes = tree.n_nodes

    # tree.get_parent_nodes(0)
    # print(tree.n_nodes)
    # print(tree.n_levels)
    # print(np.log(tree.n_nodes) / np.log(2))
