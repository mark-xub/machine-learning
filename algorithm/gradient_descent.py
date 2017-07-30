# coding=UTF-8

import numpy as np

LEARNING_RATE = 0.001


def get_cost_derivative(index, vect_x_i, vect_y_i, vect_theta):
    # print(np.shape(vect_x_i))
    # print(np.shape(vect_y_i))
    # print(np.shape(vect_theta))
    vect_h = hypothesis(vect_theta, vect_x_i)
    vect_cost = (vect_h - vect_y_i) * vect_x_i[index]
    return vect_cost


def hypothesis(vect_theta, vect_x_i):
    return np.sum(vect_theta.dot(vect_x_i))


def run_gradient_descent(vect_x, vect_y, vect_theta):
    absolute_error_limit = 0.000002
    relative_error_limit = 0

    k = 0
    while k < 5000:
        k += 1
        temp_vect_theta = np.array(vect_theta)
        for i in range(0, len(vect_x)):
            for j in range(0, len(vect_theta)):
                cost_derivative = get_cost_derivative(j, vect_x[i], vect_y[i], vect_theta)
                vect_theta[j] = vect_theta[j] - LEARNING_RATE * cost_derivative
        if np.allclose(vect_theta, temp_vect_theta,
                       atol=absolute_error_limit, rtol=relative_error_limit):
            break
    print("The Num of Iteration: %d" % k)
    print(vect_theta)


def get_data(fname):
    training_set = np.loadtxt(fname, delimiter=',')
    m, n = np.shape(training_set)
    one_set = np.ones(1)
    training_set = (np.insert(training_set, 0, values=one_set, axis=1))
    vect_x = training_set[:, :n]
    vect_y = training_set[:, n:]
    vect_theta = np.zeros(n)
    return vect_x, vect_y, vect_theta


def main(fname):
    vect_x, vect_y, vect_theta = get_data(fname)
    # print vect_x
    # print vect_y
    # print vect_theta
    run_gradient_descent(vect_x, vect_y, vect_theta)


if __name__ == '__main__':
    # main("data/unary.txt")
    main("data/multi.txt")
    # main("data/lsr_test.txt")
