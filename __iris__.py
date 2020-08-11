from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd

iris = datasets.load_iris()

X = iris.data
y = iris.target

# diabetes = pd.read_csv('./data/input/diabetes.csv')
#
# #X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
# X = diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
#
# y = diabetes[['Outcome']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# print(predictions == y_test)
# print(predictions)
# print(y_test)

print(accuracy_score(y_test, predictions))