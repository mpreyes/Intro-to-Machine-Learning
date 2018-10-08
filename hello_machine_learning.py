import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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

#6 different algorithms to evaluate
#Logistic Regression: