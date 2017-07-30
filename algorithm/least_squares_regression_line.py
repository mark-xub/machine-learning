import numpy as np


def compute_theta1(vect_x, vect_y):
    n = len(vect_x)
    x_dot_y = np.transpose(vect_x).dot(vect_y)
    sum_x = np.sum(vect_x)
    sum_y = np.sum(vect_y)
    sum_x_square = np.transpose(vect_x).dot(vect_x)
    theta1 = (n * x_dot_y - sum_x * sum_y) / (n * sum_x_square - sum_x ** 2)
    return theta1


def compute_theta0(vect_x, vect_y):
    n = len(vect_y)
    sum_x = np.sum(vect_x)
    sum_y = np.sum(vect_y)
    theta1 = compute_theta1(vect_x, vect_y)
    theta0 = (sum_y - theta1 * sum_x) / n
    return theta0


def main(fname):
    data = np.loadtxt(fname, delimiter=',')
    vect_x = data[:, 0]
    vect_y = data[:, 1]
    print("theta_0: %.3f" % compute_theta0(vect_x, vect_y))
    print("theta_1: %.3f" % compute_theta1(vect_x, vect_y))


if __name__ == '__main__':
    ''' least squares regression line'''
    # main("data/lsr_test.txt")
    main("data/unary.txt")

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(X, vect_y, color='red', marker='o')
    # plt.show()
