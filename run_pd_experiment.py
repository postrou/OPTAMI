import torch
import numpy as np
from scipy.spatial.distance import cdist
from mnist import MNIST

from OPTAMI import *

def load_data():
    mndata = MNIST('./data/')
    return mndata.load_training()


def mnist(eps, p, q, images, n):
    p, q = np.float64(images[p]), np.float64(images[q])
    # normalize
    p, q = p / sum(p), q / sum(q)

    p = (1 - eps / 8) * p + eps / (8 * n)
    q = (1 - eps / 8) * q + eps / (8 * n)

    return p, q


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)



def main():
    images, labels = load_data()
    l = len(images)
    n = len(images[0])
    m = int(np.sqrt(n))

    M = np.arange(m)
    M = cartesian_product(M, M)
    M = cdist(M, M)
    M /= np.max(M)
    M = torch.tensor(M)

    # experiments were done for
    p_list = [34860, 31226, 239, 37372, 17390]
    q_list = [45815, 35817, 43981, 54698, 49947]

    x_array = np.linspace(1 / 2e-2, 1 / 4e-4, 6)
    epslist = 1 / x_array

    eps = 1e-3
    k = 0

    epsp = eps
    p, q = mnist(epsp, p_list[k], q_list[k], images, n)
    b = torch.cat((torch.tensor(p), torch.tensor(q)))
    p_ref, q_ref = mnist(0, p_list[k], q_list[k], images, n)
    gamma = eps / 3 / np.log(n)

    lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=True)
    # lamb = torch.tensor([1e-6] * (n * 2), dtype=torch.double, requires_grad=True)
    half_lamb_len = int(len(lamb) / 2)
    psi = lamb[:half_lamb_len]
    eta = lamb[half_lamb_len:]
    psi_outer = torch.outer(psi, torch.ones(len(psi)))
    eta_outer = torch.outer(torch.ones(len(eta)), eta)

    M_over_gamma = M / gamma

    def calculate_x():
        return torch.softmax(under_exp_vector, dim=0)

    optimizer = PrimalDualAccelerated(
        [lamb],
        eps=0.01,
        calculate_x_function=calculate_x
    )

    def f():
        optimizer.zero_grad()
        lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
        under_exp_vector = (lamb_factor_over_gamma - M_over_gamma).view(-1)
        return torch.logsumexp(under_exp_vector, dim=0) - lamb @ b

    optimizer.step(f)


if __name__ == '__main__':
    main()
