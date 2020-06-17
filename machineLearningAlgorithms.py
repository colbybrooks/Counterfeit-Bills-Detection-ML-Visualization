
import numpy as np                                     # needed for arrays and math
import pandas as pd                                    # needed to read the data
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # the algorithm Perceptron
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.linear_model import LogisticRegression    # the algorithm Logistic Regression
from sklearn.svm import SVC                             # the algorithm Support Vector Machine
from sklearn.tree import DecisionTreeClassifier         # the algorithm Decision Tree Learning
from sklearn.ensemble import RandomForestClassifier    # the algorithm Random Forest
from sklearn.neighbors import KNeighborsClassifier     # the algorithm K-Nearest Neighbor

################################################################################
# Data Reading / Normalizaton / Scaling / Splitting / Transforming

table = []  # Blank array for table of accuracys

# Reading the data file and making a datafram by adding columns
data = pd.read_csv('data_banknote_authentication.txt')
data.columns = ["Variance","Skewness","Curtosis","Entropy","Class"]

# Setting the variables
x = data.iloc[:, lambda df: [0,1,2,3]]
y = data['Class']

# Splitting the data into test and train
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# Scaling, fitting, and transforming the data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

##########################################################################
# PERCEPTRON
print('-' * 15,'Perceptron','-' * 15)
ppn = Perceptron(max_iter=6, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=True)
ppn.fit(X_train_std, y_train.values.ravel())              # do the training

print('Number in test ',len(y_test))
y_pred = ppn.predict(X_test_std)           # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred = ppn.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f \n' % accuracy_score(y_combined, y_combined_pred))

#  Adding the accuracy array to the table
table.append([float("{0:.4f}".format(accuracy_score(y_test, y_pred))),
              float("{0:.4f}".format(accuracy_score(y_combined, y_combined_pred)))])

################################################################################
# Logistic Regression
print('-' * 15,'Logistic Regression','-' * 15)
lr = LogisticRegression(C=8, solver='liblinear', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)                # apply the algorithm to training data

y_pred = lr.predict(X_test_std)                   # work on the test data

# show the results
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# and analyze the combined sets
print('Number in combined ',len(y_combined))
y_combined_pred = lr.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f \n' % accuracy_score(y_combined, y_combined_pred))

table.append([float("{0:.4f}".format(accuracy_score(y_test, y_pred))),
              float("{0:.4f}".format(accuracy_score(y_combined, y_combined_pred)))])

##################################################################################
# Support Vector Machine
# kernal - specify the kernal type to use - Linear
# C - the penalty parameter - it controls the desired margin size
print('-' * 15,'Support Vector Machine','-' * 15)
svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X_train_std, y_train)                      # do the training

y_pred = svm.predict(X_test_std)                   # work on the test data

# show the results
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# and analyze the combined sets
print('Number in combined ',len(y_combined))
y_combined_pred = svm.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f \n' % accuracy_score(y_combined, y_combined_pred))

table.append([float("{0:.4f}".format(accuracy_score(y_test, y_pred))),
              float("{0:.4f}".format(accuracy_score(y_combined, y_combined_pred)))])

##########################################################################
# Decision Tree Classifier
# create the classifier and train it
print('-' * 15,'Decision Tree','-' * 15)
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
tree.fit(X_train,y_train)

y_pred = tree.predict(X_test_std)                   # work on the test data

# show the results
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# and analyze the combined sets
print('Number in combined ',len(y_combined))
y_combined_pred = tree.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f \n' % accuracy_score(y_combined, y_combined_pred))

table.append([float("{0:.4f}".format(accuracy_score(y_test, y_pred))),
              float("{0:.4f}".format(accuracy_score(y_combined, y_combined_pred)))])

#############################################################################
# Random Forest Classifier
# create the classifier and train it
# n_estimators is the number of trees in the forest
# the entropy choice grades based on information gained
# n_jobs allows multiple processors to be used
print('-' * 15,'Random Forest','-' * 15)
forest = RandomForestClassifier(criterion='entropy', n_estimators=8,
                                random_state=1, n_jobs=2)
forest.fit(X_train,y_train)

y_pred = forest.predict(X_test)         # see how we do on the test data
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))  # check accuracy

# combine the train and test data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))

# see how we do on the combined data
y_combined_pred = forest.predict(X_combined)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f \n' % accuracy_score(y_combined, y_combined_pred))

table.append([float("{0:.4f}".format(accuracy_score(y_test, y_pred))),
              float("{0:.4f}".format(accuracy_score(y_combined, y_combined_pred)))])

####################################################################################################
# K - Nearest Neighbor
# create the classifier and fit it
# using 10 neighbors
# since only 2 features, minkowski is same as euclidean distance
# where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)
print('-' * 15,'K Nearest Neighbor','-' * 15)
knn = KNeighborsClassifier(n_neighbors=10,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)

# run on the test data and print results and check accuracy
y_pred = knn.predict(X_test_std)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))

# check results on combined data
y_combined_pred = knn.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f \n' % accuracy_score(y_combined, y_combined_pred))

table.append([float("{0:.4f}".format(accuracy_score(y_test, y_pred))),
              float("{0:.4f}".format(accuracy_score(y_combined, y_combined_pred)))])

print('-' * 60)

table = np.array(table) * 100   # Coverting to Percentages
# Maxing the display of the list
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

# Creating the dataframe for the accuracy table
algorithims = ['Perceptron','Logistic Regression','Support Vector Machine','Decision Tree Learning','Random Forest','K-Nearest Neighbor']
accuracy = pd.DataFrame(table, columns = ['Accuracy %','Combined Accuracy %'], index = algorithims)
print(accuracy)
