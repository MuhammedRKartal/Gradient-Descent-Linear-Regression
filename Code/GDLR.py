#Author: Muhammed Rahmetullah Kartal
#This code implements linear regression with gradient descent algorithm.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

#returns the y=w0x+w1 formula, for each x value
def predict(x,w0,w1):
    y_pred = np.dot(x,w0) + w1
    return y_pred

#returns the absolute difference between last 2 elements of MSE
def diffL2(values):
    diff = abs(values[-2] - values[-1])
    return diff


# Batch size means it takes n data from the dataset at each epoch
# threshold is if the difference between last two loss values less than this value code will stop
# epoch is the number of iterations
def gradientDescent(data, LR, batch_size, threshold, epoch):
    data.columns = ['x', 'y']

    # creating 2 lists for logging loss function and w0,w1 values at each iteration
    mse, values = [(0, 0.1)], list()

    # defining random w0 and w1 at the start
    w0, w1 = np.random.random_sample(), np.random.random_sample()

    for i in range(1, epoch + 1):
        # taking n samples from data as new data
        s_data = data.sample(batch_size)
        x, y = s_data.x, s_data.y
        N = x.shape[0]

        # mathematical formulas
        pred = y - (w0 * x + w1)
        w0 = w0 - LR * (-2 * (np.dot(x, pred).sum() / N))
        w1 = w1 - LR * (-2 * pred.sum() / N)

        values.append((w0, w1))  # storing w0 and w1 values
        mean_se = ((y - (w0 * x + w1)) ** 2).sum() / (2 * N)  # calculating the mean squared error

        mse.append((i, mean_se))  # storing the mean squared error

        # comparing the difference between last two mses and threshold
        if diffL2([x[1] for x in mse]) < threshold:
            break
    print(i)

    return w0, w1, mse, values


# analytical solution with mathematical formulas
def analytical_solution(data):
    data.columns = ['x', 'y']
    x, y = data.x, data.y

    x_mean, y_mean = x.mean(), y.mean()

    w0 = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
    w1 = y_mean - w0 * x_mean

    return w0, w1

#preparing the data
data = pd.read_csv("data2.csv",header=None)
data.columns = ['x', 'y']
x = data.x
y = data.y

#plotting the data
sns.scatterplot(data=data, x="x", y="y").set_title('Data1')
plt.show()

#parameters
LR = 0.1
epoch = 30000
threshold = 0.000001
batch_size = 15

#running the gradient descent algorithm
w0,w1,mse,values = gradientDescent(data=data,LR=LR,batch_size=batch_size,threshold= threshold,epoch=epoch)
#making predictions
preds = predict(data.x,w0,w1)

#running analytical solution and making predictions
w0_a, w1_a = analytical_solution(data)
preds_a = predict(data.x,w0_a,w1_a)

#Formulas
print("Gradient Descent")
print("W0:",w0, "W1:",w1)
print("y=",w0,"x +",w1)
print()
print("Analytical Solution")
print("W0:",w0_a, "W1:",w1_a)
print("y=",w0_a,"x +",w1_a)

# plotting the gradient descent at each 10 iteration
for i in range(0, len(values), 10):
    plt.plot(x, values[i][0] * x + values[i][1], c='orange', alpha=0.1)
# plotting original gradient descent output
plt.plot(x, preds, c='red', label='Original')
# plotting the data
sns.scatterplot(data=data, x="x", y="y").set_title("Gradient Descent at Each Epoch")
plt.show()

#plotting original gradient descent output and the data
plt.plot(x, preds, c='black', label='Gradient Descent')
sns.scatterplot(data=data, x="x", y="y").set_title('Data1 Estimated Gradient Descent')
plt.show()

#plotting original gradient descent, analytical solution and the data
plt.plot(x, preds, c='black', label='Gradient Descent')
plt.plot(x, preds_a, c='red', label='Analytical')
sns.scatterplot(data=data, x="x", y="y").set_title("Data1 Gradient vs Analytical Solution")
plt.show()

#creating a 3rd degree polynomial model by using numpy library
weights = np.polyfit(np.array(x),np.array(y),3)
model = np.poly1d(weights) #training the model
new_y = model(x) #making predictions with model

#plotting gradient descent, analytical and polynomial outputs
plt.plot(x, preds, c='black', label='Gradient Descent')
plt.plot(x, preds_a, c='red', label='Analytical')
plt.plot(x,new_y,c = 'green',label='Polynomial')
sns.scatterplot(data=data, x="x", y="y")

plt.show()

#loss vs iteration by matplotlib
plt.plot([x[0] for x in mse], [ x[1] for x in mse], '-ok')
plt.show()

#loss vs iteration by seaborn lineplot
mse = pd.DataFrame(mse,columns = ['Iteration','MSE'])
sns.lineplot(data = mse, x='Iteration', y='MSE').set_title("Loss vs Iteration")
plt.show()

#last elements of loss
print(mse.tail())