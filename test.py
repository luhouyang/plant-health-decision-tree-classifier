#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import numpy as np

X = np.array([
    [0, 0.6, 1], # 0
    [1.5, 1, 1], # 1
    [1, 1, 0], # 2
    [1, 2, 0], # 2
    [2, 3, 2], # 1
    [0.6, 0, 3], # 0
    [2, 2, 0], # 2
    [0, 0, 1.5], # 0
    [2, 1, 0], # 2
])
y = [0, 1, 2, 2, 1, 0, 2, 0, 2]

classes = [0, 1, 2]


clf = DecisionTreeClassifier()
clf.fit(X, y)
# print("Accuracy: ", accuracy_score(y_pred=clf.predict(X), y_true=y))


def calc_probs(labels: list):
    list_size = labels.__len__()
    probs = []

    for cls in classes:
        if (cls in labels):
            true_vector = [y==cls for y in labels]
            probs.append(sum(true_vector) / list_size)
            print(f"Prob class {cls}: {sum(true_vector) / list_size}")
    
    return probs

def calc_gini(labels: list):
    list_size = labels.__len__()
    probs = []

    for cls in classes:
        if (cls in labels):
            true_vector = [y==cls for y in labels]
            probs.append(sum(true_vector) / list_size)
    
    sum_of_p_squared = 0.0
    for p in probs:
        sum_of_p_squared += p**2

    gini = 1.0 - sum_of_p_squared
    return gini


print("Initial Gini:", calc_gini(y))
calc_probs(y)

# loop through each feature
for i in range(X.shape[1]):
    # take individual columns of feature
    ith_features = X[:, i]

    # find minimum seperation value
    minimum = min(ith_features)
    second_smallest = max(ith_features)
    for num in ith_features:
        if (num < second_smallest and num > minimum):
            second_smallest = num

    half_interval = (second_smallest + minimum)/2.0

    # group into left and right according to seperation value
    left_group = []
    left_labels = []
    right_group = []
    right_labels = []
    for j in range(ith_features.size):
        if (ith_features[j] <= half_interval):
            left_group.append(ith_features[j])
            left_labels.append(y[j])
        else:
            right_group.append(ith_features[j])
            right_labels.append(y[j])

    print(f"\nSpliting with feature no. {i+1}")
    print("Split Value:", half_interval)
    print("Resulting Left Gini:", calc_gini(left_labels))
    print("Resulting Right Gini:", calc_gini(right_labels))
    print(f"Gini Reduction: {calc_gini(y) - max([calc_gini(left_labels), calc_gini(right_labels)])}")
    # print(left_group, right_group)
    # print(left_labels, right_labels)
    


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
                feature_names=["col1", "col2", "col3"],
                class_names=["zero", "one", "two"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
#%%