import os
import subprocess
import time
import numpy as np
from sklearn import metrics
from collections import defaultdict
from prefixspan import PrefixSpan
import pandas as pd
from datainpute import datainput
from splitmeasures import GiniIndex, relative_risk_index, odd, gain_ratio, relative_risk_p, odd_p


def isSubsequence(s: str, t: str) -> bool:
    if len(s) == 0:
        return True
    if len(s) == 0 and len(t) == 0:
        return True
    if len(s) != 0 and len(t) == 0:
        return False

    i = 0
    for j in t:
        if j == s[i]:
            i += 1
            if i == len(s):
                # print("True" + str(s) + " " + str(t))
                return True
    return False


class PatternGini:
    def __init__(self, pattern, gini, hit_sequence):
        self.pattern = pattern
        self.gini = gini
        self.hit_sequence = hit_sequence

    def __lt__(self, other):
        if np.all(self.gini == other.gini):
            return len(self.pattern) < len(other.pattern)
        return self.gini < other.gini


def TopkPattern(dataset, label, k):
    lebel_set = set(label)
    # for a given dataset, find the top-k pattern
    # using PrefixSpan to find the pattern
    data_set_per_label = defaultdict(list)
    for i in range(len(dataset)):
        data_set_per_label[label[i]].append(dataset[i])
    patterns = []
    for per_label in data_set_per_label:
        ps = PrefixSpan(data_set_per_label[per_label])
        patterns += ps.topk(512)
    # patterns = PrefixSpan(dataset).topk(1000)
    fre_patterns = [i[1] for i in patterns]
    # calculate the gini index and return the top-k pattern and hit_index of top-k pattern
    fre_patterns = list(set([tuple(t) for t in fre_patterns]))
    import heapq
    priority_queue = []
    label_size = {}
    for i in range(len(label)):
        if label[i] not in label_size.keys():
            label_size[label[i]] = 1
        else:
            label_size[label[i]] += 1
    for pattern in fre_patterns:
        hit_index = []
        hit_sequence = []
        for i in range(len(dataset)):
            if isSubsequence(pattern, dataset[i]):
                hit_index.append(i)
                hit_sequence.append(dataset[i])

        # 1. uing gini
        # hit_label = [label[i] for i in hit_index]
        # # self.label to list
        # label = list(label)
        # measure = GiniIndex(label, hit_label)

        # 2. using relative risk index
        # measure =  relative_risk_index(label, hit_index)

        # 3. using odd
        # measure = odd(label, hit_index)

        # 4. using relative_risk_p odd_p

        measure = relative_risk_p(label, hit_index, label_size, lebel_set)


        heapq.heappush(priority_queue, PatternGini(
            pattern, measure, hit_sequence))

    patternmeasure = heapq.nlargest(k, priority_queue)
    if len(patternmeasure) == 0:
        return None
    try:
        return patternmeasure[k-1]
    except:
        return patternmeasure[-1]
    # return the top-k pattern and hit_index of top-k pattern


class TreeNode:
    def __init__(self, pattern=None, gini=None, hit_index=None, left=None, right=None, sequential_data=None, cluster_num=None):
        self.pattern = pattern
        self.gini = gini
        self.hit_index = hit_index
        self.left = left
        self.right = right
        self.sequential_data = sequential_data
        self.cluster_num = cluster_num
        self.labels = None

    def getPattern(self):
        return self.pattern

    def getGini(self):
        return self.gini

    def getHitIndex(self):
        return self.hit_index

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def getLabel(self):
        return

    def get_x(self):
        return self.sequential_data

    def cluster(self, maxseqlength=5):
        # save the data hit to file
        self.save_rest2file(self.get_x())
        # run the clustering algorithm
        temp_labels, feature_X = self.kmeans_cluster(
            self.cluster_num, maxseqlength)
        # save the labels
        self.labels = temp_labels
        return temp_labels, feature_X

    def kmeans_cluster(self, k, maxseqlength):
        # call sequence to feature cpp
        zhilin = './RandomProjection ' + 'temprestboost ' + '1' + \
            ' ' + str(2048) + ' ' + str(maxseqlength) + ' ' + str("lcs")
        print(zhilin)
        subprocess.call(zhilin, shell=True)
        feature_X = pd.read_csv('tempvec/boost' +
                                str(0) + '.csv', header=None, sep=' ')
        feature_X = feature_X.values
        # remove the last column
        feature_X = feature_X[:, :-1]
        # get the initial cluster result with kmeans
        from sklearn.cluster import KMeans
        # # pca
        from sklearn.decomposition import PCA, KernelPCA
        # pca = PCA(n_components=3
        pca = KernelPCA(n_components=3, kernel='cosine')
        feature_X = pca.fit_transform(feature_X)
        initial_kmeans = KMeans(
            n_clusters=k, n_init=10)
        initial_labels = initial_kmeans.fit_predict(feature_X)

        return initial_labels, feature_X

    def save_rest2file(self, x):
        # write the dataset to the file
        file = open('dataset/temprestboost.txt', 'w')
        for i in range(len(x)):
            file.write(str("1")+'\t' +
                       ' '.join([str(j) for j in x[i]])+'\n')
        file.close()


class TreeCluster():
    def __init__(self, num_clusters, datasetname, maxseqlength, min_length=5):
        self.num_clusters = num_clusters
        self.datasetname = datasetname
        self.feature_X = None
        self.real_label = None
        self.fake_labels = None
        self.min_length = min_length
        self.one_kmeans_labels = None
        self.FIRSTKMEANS = True
        self.maxseqlength = maxseqlength

    def fit(self, x):
        root = TreeNode(hit_index=list(range(len(x))), pattern=None, gini=None,
                        left=None, right=None, sequential_data=x, cluster_num=self.num_clusters)
        # built the tree
        self.built_tree(root)
        sequence_label = self.get_cluster_labels(root)
        # sequence_label.append( (sequence, label) )
        # treat label, with the order of x
        final_resut = []
        for i in range(len(x)):
            temp_seq = x[i]
            for j in range(len(sequence_label)):
                if temp_seq == sequence_label[j][0]:
                    final_resut.append(sequence_label[j][1])
                    break
        print("final result: ", len(set(final_resut)))
        self.visualize_tree(root, filename=self.datasetname)
        return final_resut

    def built_tree(self, node):
        # 0. Node clustering
        # 1. find the top-k pattern
        # 2. split the node
        # 3. built the tree
        # 4. return the tree
        # 1. find the top-k pattern
        print("numb of cluster: ", self.num_clusters)
        if self.num_clusters == 1 or len(node.get_x()) <= 1:
            return
        if len(node.get_x()) < self.min_length:
            return
        fake_labels, feature_X = node.cluster(self.maxseqlength)
        if self.FIRSTKMEANS == True:
            self.one_kmeans_labels = fake_labels
            self.FIRSTKMEANS = False
        topk_pattern = TopkPattern(node.get_x(), fake_labels, 1)
        # 2. split the node
        # 2.1 get the hit_index of top-k pattern
        hit_sequence = topk_pattern.hit_sequence
        unhit_sequence = []
        for i in range(len(node.get_x())):
            if node.get_x()[i] not in hit_sequence:
                unhit_sequence.append(node.get_x()[i])
        node.pattern = topk_pattern.pattern
        node.gini = topk_pattern.gini
        self.num_clusters = self.num_clusters - 1
        # 3. built the tree
        left_node = TreeNode(pattern=None, gini=None, left=None, right=None,
                             sequential_data=unhit_sequence, cluster_num=self.num_clusters)
        node.left = left_node
        # 4. return the tree
        self.built_tree(left_node)
        right_node = TreeNode(pattern=None, gini=None, left=None, right=None,
                              sequential_data=hit_sequence, cluster_num=self.num_clusters)
        node.right = right_node
        self.built_tree(right_node)

    def get_cluster_labels(self, root):
        # 1. find all leaf node
        leaf_nodes = []
        self.find_leaf(root, leaf_nodes)
        # 2. every node has a label
        sequence_label = []
        for i in range(len(leaf_nodes)):
            for sequence in leaf_nodes[i].get_x():
                sequence_label.append((sequence, i))
        return sequence_label

    def find_leaf(self, node, leaf_nodes):
        if node.left is None and node.right is None:
            leaf_nodes.append(node)
        else:
            self.find_leaf(node.left, leaf_nodes)
            self.find_leaf(node.right, leaf_nodes)
            
    def visualize_tree(self, root, filename="tree", format="pdf"):
        if root is None:
            return
        print("Visualizing tree...")
        graph = Digraph(comment="Decision Tree")


        node_id_dict = {}
        cluster_total = 1

        def add_nodes(node, parent_id=None):
            nonlocal cluster_total
            if node.pattern is None:
                # change the label
                node_id = str(id(node))
                graph.node(str(id(node)), label="Cluster " + str(cluster_total) + '\nHit:' + str(len(node.sequential_data)))
                cluster_total = cluster_total + 1
                return

            id_node = str(id(node))
            if node.left.pattern is None and node.right.pattern is None:
                print("good")
                # create a node
                graph.node(str(id_node), label=str(node.pattern), shape="box")
                # create two ouput node: cluster_num
                graph.node(str(id(node.left)), label="Cluster " + str(cluster_total) + '\nHit:' + str(len(node.left.sequential_data)))
                cluster_total = cluster_total + 1
                graph.node(str(id(node.right)), label="Cluster " + str(cluster_total) + '\nHit:' + str(len(node.right.sequential_data)))
                cluster_total = cluster_total + 1
                # create two edges
                graph.edge(str(id_node), str(id(node.left)))
                graph.edge(str(id_node), str(id(node.right)))
                return
            else:
                # create a node
                graph.node(str(id_node), label=str(node.pattern), shape="box")
                # create two edges
                graph.edge(str(id_node), str(id(node.left)), shape="box")
                graph.edge(str(id_node), str(id(node.right)), shape="box")
                # add nodes recursively
                add_nodes(node.left)
                add_nodes(node.right)
        add_nodes(root)
        graph.render(filename=filename, format=format)

def measure_performance(data_label, y_pred):

    # measure the performance
    # 1. Purityf
    from sklearn.metrics.cluster import contingency_matrix
    import numpy as np

    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix_ = contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix_, axis=0)) / np.sum(contingency_matrix_)

    purity = purity_score(data_label, y_pred)
    # 2 . NMI
    # \operatorname{NMI}\left(\Omega, \Omega^{*}\right)=\frac{H(\Omega)+H\left(\Omega^{*}\right)-H\left(\Omega, \Omega^{*}\right)}{\left(H(\Omega)+H\left(\Omega^{*}\right)\right) / 2}
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(data_label, y_pred)

    def f_measure(labels_true, labels_pred, beta=1.):
        (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(
            labels_true, labels_pred)
        p, r = tp / (tp + fp), tp / (tp + fn)
        f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
        return f_beta

    f1 = f_measure(data_label, y_pred)
    acc = [purity, nmi, f1]
    print(acc)
    return acc


if __name__ == '__main__':
    dataset = ['activity', 'aslbu', 'auslan2', 'context', 'epitope', 'pioneer',
               'question', 'reuters', 'robot', 'skating', 'unix', 'webkb', 'news']
    # dataset = ['gene']
    # when using pca, it may have little difference when project the featrue into 2-d or 3-d.
    tree_results = []

    for data in dataset:
        print(data)
        tree_dict = {'dataset': data, 'purity': 0,
                     'nmi': 0, 'f1': 0, 'cluster': 0}

        n = 10
        for i in range(n):
            db, data_label, itemset, pattern_sequence_length = datainput(
                'dataset/{}.txt'.format(data))
            ok = TreeCluster(len(set(data_label)), data,
                             pattern_sequence_length)
            y_pred = ok.fit(db)
            tree_acc = measure_performance(data_label, y_pred)

            tree_dict['purity'] += tree_acc[0]
            tree_dict['nmi'] += tree_acc[1]
            tree_dict['f1'] += tree_acc[2]
            tree_dict['cluster'] += len(set(y_pred))
        tree_dict['purity'] /= n
        tree_dict['nmi'] /= n
        tree_dict['f1'] /= n
        tree_dict['cluster'] /= n
        tree_results.append(tree_dict)
        print(tree_results)
