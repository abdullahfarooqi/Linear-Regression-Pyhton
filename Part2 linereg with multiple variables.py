import numpy as np
import matplotlib.pyplot as plt

def plot_cost(save):
    plt.plot(np.linspace(0, iteration, iteration, endpoint=False), costs, color='lime')
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")
    plt.title("Learning Rate :" + str(alpha))
    plt.xlim([0, iteration])
    if save:
        plt.savefig('Plot-' + str(count_updalpha) + '.png')
        plt.clf()
    else:
        plt.show()

def load_data(n):
    data = np.loadtxt("ex1data2.txt", dtype=np.float64, delimiter=",")
    feature = data[::, 0:n]
    label = data[::, -1:]
    visualize_feature(feature, label)
    return feature, label

def visualize_feature(feature, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature[::, 0:1], feature[::, 1:2], label, c='r', marker='x')
    ax.set_xlabel('Size of the house')
    ax.set_ylabel('Number of bedrooms')
    ax.set_zlabel('Price of the house')
    plt.show()

def add_val(feature):
    m, n = feature.shape
    feature_val = np.ones((m, n + 1))
    feature_val[::, 1:] = feature
    return feature_val

def predict(feature, theetas):
    return feature.dot(theetas)

def normalizer(feature):
    m, n = feature.shape
    mean = [1] * n
    std_dev = [0] * n
    for i in range(1, n):
        mean[i] = np.mean(feature[:, i])
        std_dev[i] = np.std(feature[:, i])
        feature[:, i] = feature[:, i] - mean[i]
        feature[:, i] = feature[:, i] / std_dev[i]
    return mean, std_dev, feature

def cost_func(feature, theetas, label):
    temp = feature.dot(theetas) - label
    m,n = feature.shape
    j = (1.0 / (2.0 * m)) * (temp.transpose().dot(temp))
    return j

def derivative_cost(feature, theetas, label):
    m, n = feature.shape
    derivative_j = (-1.0 / m) * np.sum(feature.transpose().dot(predict(feature, theetas)-label))
    return derivative_j

def gradient_descent(feature, label, theetas, iterations, alpha):
    m, n = feature.shape
    cost_i = np.array([])
    for count in range(1,iterations+1):
        predicted = predict(feature, theetas)-label
        for i in range(n):
            theetas[i, :] = theetas[i,:] - alpha*(1.0/m)*((feature[:, i]).dot(predicted))
        cost_i = np.append(cost_i, cost_func(feature, theetas, label))
    return theetas, cost_i

def test_func(feature, mean, std_dev, theetas):
    feature = add_val(feature)
    m, n = feature.shape
    for i in range(1, n):
        feature[:, i] = feature[:, i] - mean[i]
        feature[:, i] = feature[:, i] / std_dev[i]
    return predict(feature, theetas)

no_of_features = 2  # no of features to read
X, Y = load_data(no_of_features)  # features and label
X = add_val(X)
mean, std_Dev, X = normalizer(X)  # mean, standard deviation and normalized features
r, c = X.shape
alpha = 0.01 # learning Rate
iteration = 50
count_updalpha = 1

while True:
    theetas = np.ones((c, 1))  # theetas
    upd_theetas, costs = gradient_descent(X, Y, theetas, iteration, alpha)
    plot_cost(True)

    if derivative_cost(X, upd_theetas, Y) < 0.01:
        alpha = alpha / 3
        theetas = np.ones((c, 1))  # reset theetas
        upd_theetas, costs = gradient_descent(X, Y, theetas, iteration, alpha)  # theetas and cost function for best fit learning rate
        print("Optimal Alpha:", alpha)
        print("Theetas:", theetas)
        plot_cost(False)
        break

    plt.clf()
    alpha = alpha * 3
    count_updalpha = count_updalpha + 1

test = np.array([[1650, 3]])
price = test_func(test, mean, std_Dev, upd_theetas)
print("Predicted price for (1650,3): ",price)