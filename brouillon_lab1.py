
### part1
from sklearn import tree
from matplotlib import pyplot as plt # for a good visualization of the trees 

import csv
import numpy as np
from utils import load_from_csv

# X is the training set 
# Each example in X has 4 binary features
X = [[0, 0, 1, 0], [0, 1, 0, 1] , [1, 1, 0, 0] , [1, 0, 1, 1] , [0, 0, 0, 1] , [1, 1, 1, 0]]

# Y is the CLASSES associated with the training set. 
# For instance the label of the first and second example is 1; of the third example is 0, etc
Y = [1, 1, 0, 0, 1, 1]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

#print(clf.predict([[1,1,1,1] , [0,1,0,0] , [1,1,0,1] ]))


### part2
# text_representation = tree.export_text(clf)
# print(text_representation)

# fig = plt.figure(figsize=(10,7))
# _ = tree.plot_tree(clf, 
#                    feature_names= ("f1","f2" , "f3", "f4"),
#                    class_names= ("false (0)", "true (1)" ), 
#                    filled=True)

# plt.show()

#new binary dataset 
Z = [[0, 0, 1, 0], [0, 1, 0, 1] , [1, 1, 0, 0] , [1, 0, 1, 1] , [0, 0, 0, 1] , [1, 1, 1, 0],[1,1,1,1],[0,0,0,0],[0,1,1,0] ]

#3 more arrays added
Z_match = [1, 1, 0, 0, 1, 1,1,0,1]



clf = tree.DecisionTreeClassifier()

clf = clf.fit(Z, Z_match)

print(clf.predict([[1,1,1,1] , [0,1,0,0] , [1,1,0,1] ]))

# fig2 = plt.figure(figsize=(10,7))
# _ = tree.plot_tree(clf, 
#                    feature_names= ("eye","hair" , "ear", "nose"),
#                    class_names= ("human", "monkey"), 
#                    filled=True)
# plt.show()


### part3
## compass dataset 
# what are the features --> criterias that will help in a prediction
#for ex : date of birth, ethnicity, sex_code_Text, 
#questions de comprehension de la dataset 

train_examples, train_labels, features, prediction = load_from_csv("./compass.csv") 
print(features)
