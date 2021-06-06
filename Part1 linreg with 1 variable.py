import matplotlib.pyplot as plt
import numpy as np

def predicted (theeta, theeta_one,x):
    pred_y = np.arange (len(x))
    for i in range (len(x)-1):
        pred_y[i]=x[i]*theeta_one+theeta
    return pred_y

with open('ex1data1.txt', 'r') as f:
    lines = f.readlines()
    x = np.asarray([float(line.split(',')[0]) for line in lines])
    y = np.asarray([float(line.split(',')[1]) for line in lines])

plt.scatter(x, y, c="r", marker='x', label="training feature")
plt.legend()
plt.xlabel("Population of a city in 10000s")
plt.ylabel("Profit in $10,000s")
plt.show()

alpha = 0.001
init_cost=np.sum(abs(x))
while True:
    theeta=1
    theeta_one=1
    m=len(x)
    iter=1000
    for i in range (iter):
        y_pred=predicted (theeta,theeta_one,x)
        der_th = 2/m * sum(y_pred-y)
        der_thone = 2/m * sum ((y_pred - y) * x)
        theeta=theeta-alpha*der_th
        theeta_one=theeta_one-alpha*der_thone
    updated_cost=np.sum(abs(y_pred))
    if (updated_cost > init_cost):
        break
    print("Parameters(theeta0,theeta1): ", theeta, theeta_one, "Alpha", alpha)
    alpha = alpha + 0.0001
    init_cost=updated_cost

y_pred = theeta_one*x + theeta
plt.scatter(x, y, c="r", marker='x', label="Training feature")
plt.plot(x, y_pred, color='blue', label="Linear regression")  # regression line
plt.legend()
plt.show()