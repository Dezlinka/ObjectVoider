import numpy as np
from random import random as rand
class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None
    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        features_unique = [np.unique(f) for f in np.array(features).T]
        features = np.array(features).tolist()
        # build the tree
        self.root_node = TreeNode(features, labels, features_unique)
        if self.root_node.splittable:
            self.root_node.split()
        return
    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred
# Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    entropies = []
    attribute_probabilities = []
    for attribute in range(len(branches)):
        total_points = sum(branches[attribute])
        attribute_probabilities.append(float(total_points) / (np.sum(np.array(branches))))
        entropy = sum([(-1) * (float(x) / total_points) * (np.log2(float(x) / total_points)) if x != 0 else 0 for x in
                       branches[attribute]])
        entropies.append(entropy)
    conditional_entropy = sum([x * y for x, y in zip(entropies, attribute_probabilities)])
    return S - conditional_entropy
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    y_pred = decisionTree.predict(X_test)
    correct_count = len([j for i, j in zip(y_test, y_pred) if i == j])
    accuracy = float(correct_count) / len(y_test)
    if not decisionTree.root_node.splittable:
        return
    all_nodes = []
    sub_node = []
    sub_node.append(decisionTree.root_node)
    counter = -1
    while len(sub_node) != 0:
        all_nodes.append(sub_node)
        counter += 1
        sub_node = [[treenode for treenode in n.children if treenode.splittable] for n in all_nodes[counter]]
        sub_node = np.hstack(sub_node).tolist()
    all_nodes = np.hstack(all_nodes).tolist()
    for node in reversed(range(0, len(all_nodes))):
        children = all_nodes[node].children
        all_nodes[node].children = []
        all_nodes[node].splittable = False
        y_prune_pred = decisionTree.predict(X_test)
        correct_prune_count = len([j for i, j in zip(y_test, y_prune_pred) if i == j])
        accuracy_prune = float(correct_prune_count) / len(y_test)
        if accuracy_prune >= accuracy:
            accuracy = accuracy_prune
        else:
            all_nodes[node].splittable = True
            all_nodes[node].children = children
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')
    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])
    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t',
                       deep=deep + 1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
class TreeNode(object):
    def __init__(self, features, labels, features_unique):
        # features: List[List[any]], labels: List[int], features_unique: List[List[any]]
        self.features = features
        self.labels = labels
        self.children = []
        self.features_unique = features_unique
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2 or len(np.array(self.features).T) < 1:
            self.splittable = False
        else:
            self.splittable = True
        self.dim_split = None  # the index of the feature to be split
        self.feature_uniq_split = None  # the possible unique values of the feature to be split
    # Split current node
    def split(self):
        feature_information_gains = []
        unique_labels, unique_label_count = np.unique(self.labels, return_counts=True)
        for f in range(len(np.array(self.features).T)):
            feature_class_count = [
                [len([i for i, j in zip(np.array(self.features)[:, f], np.array(self.labels)) if i == feature
                      and j == label]) for label in unique_labels] for feature in self.features_unique[f]]
            Entropy = sum([(-1) * (float(x) / sum(unique_label_count)) * np.log2(float(x) / sum(unique_label_count))
                           for x in unique_label_count])
            feature_information_gains.append(
                (Information_Gain(Entropy, feature_class_count), len(np.unique(np.array(self.features)[:, f]))))
        information_gains = np.array([i[0] for i in feature_information_gains])
        if all(information_gains == 0.0):
            self.splittable = False
            return
        self.dim_split = feature_information_gains.index(max (feature_information_gains, key=lambda x: (x[0], x[1])))
        feature_labels = np.column_stack((self.features, self.labels)).tolist()
        feature_labels.sort(key=lambda x: x[self.dim_split])
        feature_unique, unique_index = np.unique(np.array(feature_labels)[:, self.dim_split], return_index=True)
        feature_class_split = np.split(feature_labels, unique_index[1:])
        self.feature_uniq_split = self.features_unique[self.dim_split]
        self.feature_uniq_split = self.feature_uniq_split.tolist()
        self.feature_uniq_split.sort()
        feature_unique = feature_unique.tolist()
        for i in range(len(self.feature_uniq_split)):
            if not self.feature_uniq_split[i] in feature_unique:
                new_child = TreeNode([[]], self.labels, [[]])
                new_child.cls_max = self.cls_max
                self.children.append(new_child)
            else:
                index = feature_unique.index(self.feature_uniq_split[i])
                child_labels = feature_class_split[index][:, -1]
                child_features = np.delete(feature_class_split[index], -1, 1)
                child_features = np.delete(child_features, self.dim_split, 1)
                child_features_unique = np.delete(self.features_unique, self.dim_split, 0)
                new_child = TreeNode(child_features.tolist(), child_labels.astype(int).tolist(), child_features_unique)
                self.children.append(new_child)
                if new_child.splittable:
                    new_child.split()
    # Predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int≥
        if not self.splittable:
            return self.cls_max
        else:
           split_child = self.children[self.feature_uniq_split.index(feature[self.dim_split])]
            feature = np.delete(np.array(feature), self.dim_split)
            return split_child.predict(feature)
if __name__ == "__main__":
