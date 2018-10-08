import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#algorithms from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



print("Welcome")


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

print("dataset.shape: (x instances, y attributes")
print(dataset.shape) #150 instances, 5 attributes

#head, eyeball data, 1st 20 rows of data
print("dataset.head: show first 20 rows of data")
print(dataset.head(20))
print("dataset.describe(): show count, mean, min,max values ")
print(dataset.describe()) #includes count, mean, min, max values

#class distribution
print("dataset.groupby(class): show class distribution")
print(dataset.groupby('class').size())

#data visualization
#univariate plots: to better understand each attribute
# multivariate plots: to better understand the relationship between attributes

#box and whisker plot

dataset.plot(kind="box",subplots = True, layout=(2,2), sharex = False, sharey = False)
plt.show()

#historgrams
dataset.hist()
plt.show()

#multivariate plots


#scatter plot matrix
scatter_matrix(dataset) #notice diagonal grouping of some attributes. This suggests a high correlation & predictable relationship
plt.show()

#create a validation dataset - a dataset we can use to check if the model we created is any good
#distribution - 80% of which we will use to train, 20% to test: validation dataset

array = dataset.values
X = array[:,0:4] 
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size= validation_size,random_state=seed)
#training data: X_train, Y_train
#validation data: X_validation, Y_validation

#Test harness
#test options & evaluation metric
seed = 7
scoring = 'accuracy'
#this will split dataset into 10 parts: train on 9, test on 1, repeat for all combinations of train-test splits
#we are using the metric of accuracy to evaluate models: ratio of (# of correctly predicted instances/ total # of instances in dataset) * 100
#gives a percentage, e.g. 95% accurate

#6 different algorithms to evaluate:

#Logistic Regression: uses logistic function to model a binary variable: pass/fail, win/lose, 0,1

#Linear Discriminant Analysis (LDA): method used to find a linear combination of features that characterizes 
#or separates two or more classes of objects or events. LDA explicitly attempts to model the difference between the classes of data.
#Discriminant function analysis is classification - the act of distributing things into groups, classes, etc

#K-Nearest Neighbors: non parametric method used for classification & regression. Output is a class membership
#classified by the majority vote of it's neighbors, with the object being assigned to the class most common 
#amongst its k nearest neighbors. Simplest of ML algorithms

#Classification & Regression trees (CART): trees used to go from observations of an item to conclusion about an
#item's target value
#tree models where target variable can take a discrete set of values -> classification trees

#Gaussian Naive bayes: Bayes theorem, extended real-valued attributes

#Support Vector Machines (SVM): supervised learning models with associated learning algorithms that analyze 
# data used for classification & regression analysis
# given set of training examples, each marked as belonging to one or the other of two categories it builds a 
# model that assigns new examples to one category or another


#spot check algorithms
models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

#evaluate each model in turn

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) " % (name, cv_results.mean(),cv_results.std())
    print(msg)
#from this, we see KNN has the largest estimated accuracy score
#
# Compare Algorithms
#  
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#KNN most accurate model tested
#now to get an idea of accuracy of model on validation set

#make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

