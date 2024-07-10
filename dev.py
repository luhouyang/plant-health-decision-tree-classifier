#
# load_iris implementation: https://github.com/scikit-learn/scikit-learn/blob/70fdc843a/sklearn/datasets/_base.py#L616
# features, ..., target
#

#%%
# import dependancies
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from IPython.display import display
import numpy as np

# create decision tree classifier
clf = DecisionTreeClassifier(random_state=0)

# load dataset
# iris = load_iris()

# inspect data
import pandas as pd
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
df = pd.read_csv('plant_health/plant_health_train_csv.csv')

print(f'Size of dataframe: {df.shape}\n')
display(df.head())

data = df.iloc[:, :-1]
targets = df.iloc[:, -1]

display(data.head())
display(targets.head())

targets_mapping = {
    "Nitrogen_Deficiency": 0,
    "Nitrogen_Toxicity": 1,
    "Water": 2,
    "Phosphorus_Deficiency": 3,
    "Potassium_Deficiency": 4,
    "Magnesium_Deficiency": 5,
    "Calcium_Deficiency": 6,
    "Sulfur_Deficiency": 7,
    "Iron_Deficiency": 8,
    "Zinc_Deficiency": 9,
    "Manganese_Deficiency": 10,
    "Copper_Deficiency": 11,
    "Boron_Deficiency": 12,
    "Boron_Toxicity": 13,
    "Molybdenum_Deficiency": 14,
}

feature_names = [
    "youngLeavesChlorosis", "oldLeavesChlorosis", "entirePlantChlolosis",
    "leavesEdgeChlorosis", "spotsChlorosis", "leavesTipChlorosis",
    "completeInterveinalChlorosis", "partialInterveinalChlorosis",
    "leavesVeinChlorosis", "deepGreenYoungLeaves", "deepGreenOldLeaves",
    "paleGreenLeaves", "rollCurlingLeaves", "flaccidLeaves", "witchesBroom",
    "dieback", "leavesEdgeBrownAsh", "spotsBrownAsh", "leavesTipBrownAsh",
    "leavesEdgeRedPurple", "spotsRedPurple", "leavesTipRedPurple",
    "leavesVeinRedPurple"
]

targets = pd.DataFrame([targets_mapping[x] for x in targets])
display(targets.head(50))

# seperate data into train and test
from sklearn.model_selection import train_test_split
# X = iris.data
# y = iris.target
X = data
y = targets

train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42)

# check shape of train_data, test_data | (number of examples, number of features)
print(f'Train Data Shape: {train_data.shape}')
print(f'Test Data Shape: {test_data.shape}\n')

# check shape of train_labels, test_labels (number of examples, )
print(f'Train Labels Shape: {train_labels.shape}')
print(f'Test Labels Shape: {test_labels.shape}')

# fit classifier
clf.fit(train_data, train_labels)

# accuracy after fitting
print("Accuracy", metrics.accuracy_score(test_labels, clf.predict(test_data)))

# show decision tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

class_names = list(targets_mapping.keys())

dot_data = StringIO()
export_graphviz(clf,
                out_file=dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names=feature_names,
                class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('dev_plant.png')
Image(graph.create_png())

#%%
