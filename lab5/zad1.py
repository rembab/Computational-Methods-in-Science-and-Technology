import numpy as np
import matplotlib.pyplot as plt


def draw_3d(X, Y, Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(X, Y, Z)
    ax.set_aspect("equal", "box")
    plt.show()


def draw_3d_sphere():
    X, Y, Z = get_sphere()
    draw_3d(X, Y, Z)


def get_sphere(n=30):
    S = np.linspace(0, 2 * np.pi, n)
    T = np.linspace(0, np.pi, n)

    S, T = np.meshgrid(S, T)

    X = np.cos(S) * np.sin(T)
    Y = np.sin(S) * np.sin(T)
    Z = np.cos(T)

    return X, Y, Z


def get_random_transformation(a=10):
    rng = np.random.default_rng()
    return 2 * a * rng.random((3, 3)) - a


def draw_random_eclipsoid(n=30):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y, Z = get_sphere()
    X, Y, Z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
    A = get_random_transformation()
    U, S, Vh = np.linalg.svd(A)
    print(U)

    for i in range(3):
        ax.plot([0, U[i][0] * S[i]], [0, U[i][1] * S[i]], [0, U[i][2] * S[i]])

    X, Y, Z = A @ [X, Y, Z]
    ax.scatter(X, Y, Z, s=1)
    ax.set_aspect("equal", "box")
    plt.show()

    # draw_3d(X, Y, Z)


draw_random_eclipsoid(50)
