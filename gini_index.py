# remove 1 hash tag '#' to make it run like Jupyter Notebook
##%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

import numpy as np

# X = np.array([
#     [0, 0.6, 1], # 0
#     [1.5, 1, 1], # 1
#     [1, 1, 0], # 2
#     [1, 2, 0], # 2
#     [2, 3, 2], # 1
#     [0.6, 0, 3], # 0
#     [2, 2, 0], # 2
#     [0, 0, 1.5], # 0
#     [2, 1, 0], # 2
# ])
# y = [0, 1, 2, 2, 1, 0, 2, 0, 2]

# classes = [0, 1, 2]
# feature_name = ["col1", "col2", "col3"]
# class_name = ["zero", "one", "two"]

iris = load_iris()
X = iris.data
y = iris.target
classes = [0, 1, 2]
feature_name = iris.feature_names
class_name = ['Setosa', 'Versicolor', 'Virginica']

clf = DecisionTreeClassifier()
clf.fit(X, y)


# calculate each class in left and right nodes
def calc_class_distribution(left_group, right_group):
    left_cls_dist = []
    right_cls_dist = []

    for cls in classes:
        if (cls in left_group):
            true_vector = [y == cls for y in left_group]
            left_cls_dist.append(sum(true_vector))
        else:
            left_cls_dist.append(0)

    for cls in classes:
        if (cls in right_group):
            true_vector = [y == cls for y in right_group]
            right_cls_dist.append(sum(true_vector))
        else:
            right_cls_dist.append(0)

    return [left_cls_dist, right_cls_dist]


# calculate probability of each class
def calc_probs(labels: list):
    list_size = labels.__len__()
    probs = []

    for cls in classes:
        if (cls in labels):
            true_vector = [y == cls for y in labels]
            probs.append(sum(true_vector) / list_size)
            print(f"Prob class {cls}: {sum(true_vector) / list_size}")

    return probs


# calculate gini value of the node
def calc_gini(labels: list):
    list_size = labels.__len__()
    probs = []

    for cls in classes:
        if (cls in labels):
            true_vector = [y == cls for y in labels]
            probs.append(sum(true_vector) / list_size)

    sum_of_p_squared = 0.0
    for p in probs:
        sum_of_p_squared += p**2

    gini = 1.0 - sum_of_p_squared
    return gini


print("\nInitial Gini:", calc_gini(y))
calc_probs(y)

# loop through each feature
for i in range(X.shape[1]):
    # take individual columns of feature
    ith_features = X[:, i]

    split_gini_value = []
    split_class_dist = []

    # all values of each feature is used to split the data
    for val in ith_features:
        # group into left and right according to seperation value
        left_group = []
        left_labels = []
        right_group = []
        right_labels = []
        for j in range(ith_features.size):
            if (ith_features[j] <= val):
                left_group.append(ith_features[j])
                left_labels.append(y[j])
            else:
                right_group.append(ith_features[j])
                right_labels.append(y[j])

        # append [feature number, split value, left node gini value + right node gini value]
        split_gini_value.append([i, val, calc_gini(left_labels) + calc_gini(right_labels)])

        # append class ditribution of left and right node
        # example [50, 0, 0] | [0, 50, 50]
        split_class_dist.append(calc_class_distribution(left_labels, right_labels))

    # get 1D array of all gini values
    # get index of lowest gini values
    split_gini_value_np_array = np.asarray(split_gini_value)
    ginis = np.squeeze(split_gini_value_np_array[:, 2])
    index = np.argmin(split_gini_value_np_array[:, 2])

    print(f"\nSpliting with feature no. {i+1}")
    print("Feature Name:", feature_name[i])
    print("Split Value:", split_gini_value[index][1])
    print("Resulting Gini:", split_gini_value[index][2])
    print("Split Class Dist:", split_class_dist[index])

    # # find minimum seperation value
    # minimum = min(ith_features)
    # second_smallest = max(ith_features)
    # for num in ith_features:
    #     if (num < second_smallest and num > minimum):
    #         second_smallest = num

    # half_interval = minimum + (second_smallest - minimum)/2.0

    # # group into left and right according to seperation value
    # left_group = []
    # left_labels = []
    # right_group = []
    # right_labels = []
    # for j in range(ith_features.size):
    #     if (ith_features[j] <= half_interval):
    #         left_group.append(ith_features[j])
    #         left_labels.append(y[j])
    #     else:
    #         right_group.append(ith_features[j])
    #         right_labels.append(y[j])

    # print(f"\nSpliting with feature no. {i+1}")
    # print("Split Value:", half_interval)
    # print("Resulting Left Gini:", calc_gini(left_labels))
    # print("Resulting Right Gini:", calc_gini(right_labels))
    # print(f"Gini Reduction: {calc_gini(y) - max([calc_gini(left_labels), calc_gini(right_labels)])}")
    # print(f"Left leaf: {left_group}")
    # print(f"Left labels: {left_labels}")
    # print(f"Right leaf: {right_group}")
    # print(f"Right labels: {right_labels}")

# show decision tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf,
                out_file=dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names=feature_name,
                class_names=class_name)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
#%%
