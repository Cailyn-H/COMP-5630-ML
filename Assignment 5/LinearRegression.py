import pandas as pd                                
import numpy as np       
import matplotlib.pyplot as plt


data = pd.read_csv("numpydataset.csv")      # load the csv file as a DataFrame
data.head()                                 # displays the first 5 rows in the dataset
print(data)

samples = len(data)              # calculating number of samples


def MSE(points, m, b):
    # write your code here...
    X = points['Features'].values
    Y = points['Targets'].values
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    n = len(X)
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (X[i] - x_mean) * (Y[i] - y_mean)
        denominator += (X[i] - x_mean) ** 2

    b1 = numerator / denominator
    b0 = y_mean - (b1 * x_mean)
    rmse = 0
    for i in range(n):
        y_pred = b0 + b1 * X[i]
        rmse += (Y[i] - y_pred) ** 2

    rmse = np.sqrt(rmse / n)
    return rmse
    #return mean_sqaured_error


def gradient_descent(m_current, b_current, points, step_size):
    learning_rate = step_size
    iterations = 1000
    current_weight = m_current
    current_bias = b_current
    x = points['Features']
    y = points['Targets']
    n = float(len(x))

    for i in range(iterations):

        # Making predictions
        y_predicted = (current_weight * x) + current_bias


        # Calculating the gradients
        weight_derivative = -(2 / n) * sum(x * (y - y_predicted))
        bias_derivative = -(2 / n) * sum(y - y_predicted)

        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

    #return m_new, b_new
    return current_weight, current_bias



m, b = 0, 0
L = 0.001       # initial learning rate, can be adjusted later
epochs = 100    # we iterate over the same dataset 100 times

for epoch in range(1, epochs+1):
    m, b = gradient_descent(m, b, data, L)
    loss = MSE(data, m, b)
    print(f"Epoch {epoch}, m: {m}, b:{b}, Loss: {loss}")
print(m, b, loss)



fig, ax = plt.subplots(1,1)

ax.scatter(data.Features, 
           data.Targets, 
           color="red", 
           linewidths=0.5, 
           label="Points")
ax.plot(data.Features, 
        [m * x + b for x in data.Features], 
        linewidth=3, 
        linestyle="dashed", 
        label="$ f(x) = mx+c $")

ax.legend(loc="lower right", bbox_to_anchor=(.96, 0.0))
ax.set_xlabel("Features")
ax.set_ylabel("Targets")

plt.savefig('LinearRegression001.png')

plt.close()


m, b = 0, 0
L = 0.01   # new learning rate
epochs = 100

for epoch in range(1, epochs+1):
    m, b = gradient_descent(m, b, data, L)
    loss = MSE(data, m, b)
    print(f"Epoch {epoch}, m: {m}, b:{b}, Loss: {loss}")
print(m, b, loss)



fig, ax = plt.subplots(1,1)

ax.scatter(data.Features, 
           data.Targets, 
           color="red", 
           linewidths=0.5, 
           label="Points")
ax.plot(data.Features, 
        [m * x + b for x in data.Features], 
        linewidth=3, 
        linestyle="dashed", 
        label="$ f(x) = mx+c $")

ax.legend(loc="lower right", bbox_to_anchor=(.96, 0.0))
ax.set_xlabel("Features")
ax.set_ylabel("Targets")

plt.savefig('LinearRegression01.png')
plt.close()

