#%%
# import dependancies
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
#%%

#%%
# create decision tree classifier
clf = DecisionTreeClassifier(random_state=0)
#%%

#%%
# load dataset
iris = load_iris()
#%%

#%%
# inspect data
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#%%

#%%
print(f'Size of dataframe: {df.shape}\n')
df.head()
#%%

#%%
# seperate data into train and test
from sklearn.model_selection import train_test_split
X = iris.data
y = iris.target

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=0)
#%%

#%%
# check shape of train_data, test_data | (number of examples, number of features)
print(f'Train Data Shape: {train_data.shape}')
print(f'Test Data Shape: {test_data.shape}\n')

# check shape of train_labels, test_labels (number of examples, )
print(f'Train Labels Shape: {train_labels.shape}')
print(f'Test Labels Shape: {test_labels.shape}')
#%%

#%%
# fit classifier
clf.fit(train_data, train_labels)
#%%

#%%
# show decision tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names=iris.feature_names,
                class_names=['0', '1', '2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('iris.png')
Image(graph.create_png())
#%%

#%%
# accuracy after fitting
print("Accuracy", metrics.accuracy_score(test_labels, clf.predict(test_data)))
#%%
