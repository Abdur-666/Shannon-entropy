
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn import svm


# read data
x1 = np.random.multivariate_normal([51,54], [[3, 0.5], [0.5, 4]], 100)
y1= np.random.multivariate_normal([49.9,47], [[4,1],[1,3]], 100)

#sessioning the random data
with open("set1.csv","r+") as set1:
    writer = csv.writer(set1)
    writer.writerows(x1)
with open("set2.csv","r+") as set2:
    writer = csv.writer(set2)
    writer.writerows(y1)

X= np.concatenate((x1,y1),axis =0 )

y= np.array([0]*100 + [1]*100)

log_reg = LogisticRegression()
log_reg.fit(X, y)

# clf = svm.SVC(kernel="linear")
# clf = clf.fit(X,y)
parameters = log_reg.coef_[0]

parameter0 = log_reg.intercept_

# Plotting the decision boundary
fig = plt.figure()
x_values = [np.min(X[:, 1] -5 ), np.max(X[:, 1] +5 )]

y_values = np.dot((-1./parameters[1]), (np.dot(parameters[0],x_values) + parameter0))
colors=['blue' if l==0 else 'yellow' for l in y]
f = plt.axes()
f.set_facecolor("black")
plt.scatter(X[:, 0], X[:, 1], label='Logistics regression', color=colors)
plt.plot(x_values, y_values, label='Decision Boundary',c="white")
plt.title("Classification of two sets of 2D Normal distribution")

plt.show()
