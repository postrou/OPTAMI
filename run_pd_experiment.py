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


def calculate_x(lamb, half_lamb_len, gamma, M_over_gamma):
    psi = lamb[:half_lamb_len]
    eta = lamb[half_lamb_len:]
    psi_outer = torch.outer(psi, torch.ones(len(psi)))
    eta_outer = torch.outer(torch.ones(len(eta)), eta)
    lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
    under_exp_vector = (lamb_factor_over_gamma - M_over_gamma).view(-1)
    return torch.softmax(under_exp_vector, dim=0)


def phi(lamb, optimizer, half_lamb_len, gamma, M_over_gamma, b):
    optimizer.zero_grad()
    psi = lamb[:half_lamb_len]
    eta = lamb[half_lamb_len:]
    psi_outer = torch.outer(psi, torch.ones(len(psi)))
    eta_outer = torch.outer(torch.ones(len(eta)), eta)
    lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
    under_exp_vector = (lamb_factor_over_gamma - M_over_gamma).view(-1)
    return torch.logsumexp(under_exp_vector, dim=0) - lamb @ b


def f(x, M, gamma):
    x_copy = x.detach().clone()
    x_copy_under_log = x_copy.clone()
    x_copy_under_log[x_copy == 0.] = 1e-6
    return (M.view(-1) * x_copy).sum() + gamma * (x_copy * np.log(x_copy_under_log)).sum()


def optimize(optimizer, closure, eps, M, gamma):
    i = 1
    while True:
        optimizer.step(closure)

        with torch.no_grad():
            x_hat = optimizer.state['default']['x_hat'][0]
            lamb = optimizer.param_groups[0]['params'][0]
            phi_value = closure()
            f_value = f(x_hat, M, gamma)
            criterion = abs(phi_value + f_value)

            print('\n'.join(
                [
                    f'Step #{i}, criterion={criterion}, phi={phi_value}, f={f_value}',
                    f'lambda.sum()={lamb.detach().sum()}',
                    f'x_hat.sum()={x_hat.detach().sum()}'
                ]), end='\n\n')

            i += 1

            if criterion < eps:
                break


def run_experiment(M_p, gamma, eps, image_index=0):
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

    # x_array = np.linspace(1 / 2e-2, 1 / 4e-4, 6)
    # epslist = 1 / x_array

    epsp = eps
    p, q = mnist(epsp, p_list[image_index], q_list[image_index], images, n)
    b = torch.cat((torch.tensor(p), torch.tensor(q)))

    # lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=True)
    lamb = torch.tensor([1e-6] * (n * 2), dtype=torch.double, requires_grad=True)
    half_lamb_len = int(len(lamb) / 2)
    M_over_gamma = M / gamma

    optimizer = PrimalDualAccelerated(
        [lamb],
        M_p=M_p,
        eps=0.01,
        calculate_x_function=lambda lamb: calculate_x(lamb, half_lamb_len, gamma, M_over_gamma)
    )
    closure = lambda: phi(lamb, optimizer, half_lamb_len, gamma, M_over_gamma, b)
    optimize(optimizer, closure, eps, M, gamma)


if __name__ == '__main__':
    eps = 1e-3
    M_p = 3 * 1e+8
    gamma = eps / 3 / np.log(n)
    image_index = 0
    run_experiment(M_p, gamma, eps, image_index)
