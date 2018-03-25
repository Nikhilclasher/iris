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
from sklearn import cross_validation
from sklearn import svm
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
Y = iris.target
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size)
clf  = svm.SVC(kernel = 'poly')
#clf.kernel
clf.fit(X_train,Y_train)
accuracy = clf.score(X_validation,Y_validation)
print(accuracy*100)
# as u can see that knc has higher accuracy thats why iam use it or u can also try it with another versions
