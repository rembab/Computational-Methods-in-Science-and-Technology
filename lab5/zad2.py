import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

I = plt.imread("./images/kitty.png")


if len(I.shape) == 3:
    I = np.dot(I[..., :3], [0.2989, 0.5870, 0.1140])


U, S, V = np.linalg.svd(I, full_matrices=False)


norm_data_x = []
norm_data_y = []
print(V[0, :].shape)
for k in range(2, 50, 2):

    I_a = S[0] * np.outer(U[:, 0], V[0, :])
    for i in range(1, k):
        I_a += S[i] * np.outer(U[:, i], V[i, :])

    # I_a = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(V[:k, :])

    norm_data_x.append(k)
    norm_data_y.append(np.linalg.norm(I - I_a))

    plt.imsave("./images/kitty_approx_" + str(k) + ".png", I_a, cmap="gray")
    print("Saved ./images/kitty_approx_" + str(k) + ".png")


plt.plot(norm_data_x, norm_data_y)
plt.show
