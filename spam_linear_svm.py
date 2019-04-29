import time
from numpy import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# have to use default delimiter 'white space'
rawTrainingData = loadtxt('spam.txt')  # read '.txt' file
clf = LinearSVC()  # classifier

X = rawTrainingData[:, 0:57]  # attributes , [raw, column]  from:to
y = rawTrainingData[:, 57]  # class

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)  # spliting into test and training

# Timing The Code
start_time = time.time()

clf.fit(X_train, y_train)  # fitting the model
y_pred = clf.predict(X_test)  # predicted class

elapsed_time = time.time() - start_time
y_true = y_test  # actual class
accuracy = clf.score(X_test, y_test)

print (confusion_matrix(y_true, y_pred))

print("\n", "Elapsed Time = ", elapsed_time*1000, " ms.")
print("Accuracy: ", accuracy*100, "%")

# print ("\n", y_pred)
