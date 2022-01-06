import torch
import numpy as np
from scipy.spatial.distance import cdist
from mnist import MNIST
from tqdm import trange

from OPTAMI import *

def load_data(path='./data/'):
    mndata = MNIST(path)
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
        arr[..., i] = a
    return arr.reshape(-1, la)


def calculate_M_matrix(m):
    M_matrix = np.arange(m)
    M_matrix = cartesian_product(M_matrix, M_matrix)
    M_matrix = cdist(M_matrix, M_matrix)
    # M_matrix /= np.max(M_matrix)
    M_matrix /= np.median(M_matrix)
    return torch.tensor(M_matrix)


def calculate_x(lamb, half_lamb_len, gamma, M_matrix_over_gamma, device='cpu'):
    psi = lamb[:half_lamb_len]
    eta = lamb[half_lamb_len:]
    psi_outer = torch.outer(psi, torch.ones(half_lamb_len, device=device))
    eta_outer = torch.outer(torch.ones(half_lamb_len, device=device), eta)
    lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
    under_exp_vector_splitted = (lamb_factor_over_gamma - M_matrix_over_gamma).hsplit(M_matrix_over_gamma.shape[1])
    assert len(under_exp_vector_splitted) == M_matrix_over_gamma.shape[1]
    under_exp_vector = torch.vstack(under_exp_vector_splitted).view(-1)
    return torch.softmax(under_exp_vector, dim=0)


def phi(lamb, optimizer, half_lamb_len, gamma, M_matrix_over_gamma, b, device='cpu'):
    optimizer.zero_grad()
    psi = lamb[:half_lamb_len]
    eta = lamb[half_lamb_len:]
    psi_outer = torch.outer(psi, torch.ones(len(psi), device=device))
    eta_outer = torch.outer(torch.ones(len(eta), device=device), eta)
    lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
    under_exp_vector_splitted = (lamb_factor_over_gamma - M_matrix_over_gamma).hsplit(M_matrix_over_gamma.shape[1])
    assert len(under_exp_vector_splitted) == M_matrix_over_gamma.shape[1]
    under_exp_vector = torch.vstack(under_exp_vector_splitted).view(-1)
    return gamma * torch.logsumexp(under_exp_vector, dim=0) - lamb @ b


def f(x, M_matrix, gamma, device='cpu'):
    x_copy = x.detach().clone().to(device)
    x_copy_under_log = x_copy.clone()
    # TODO: check
    x_copy_under_log[x_copy == 0.] = 1e-6

    M_matrix_splitted = M_matrix.hsplit(M_matrix.shape[1])
    M_matrix_to_vector = torch.vstack(M_matrix_splitted).view(-1)
    return (M_matrix_to_vector * x_copy).sum() + gamma * (x_copy * torch.log(x_copy_under_log)).sum()


def optimize(optimizer, closure, eps, M_matrix, gamma, max_steps=100, device='cpu'):
    i = 1
    while True:
        optimizer.step(closure)

        with torch.no_grad():
            x_hat = optimizer.state['default']['x_hat'][0]
            lamb = optimizer.param_groups[0]['params'][0]
            phi_value = closure()
            f_value = f(x_hat, M_matrix, gamma, device)
            criterion = abs(phi_value + f_value)

            print('\n'.join(
                [
                    f'Step #{i}, criterion={criterion}, phi={phi_value}, f={f_value}',
                    f'lambda={lamb.detach()}',
                    f'x_hat={x_hat.detach()}'
                ]), end='\n\n')

            if criterion < eps or i == max_steps:
                break

            i += 1


def calculate_A_matrix(n):
    A = torch.hstack([torch.eye(n)] * n)
    vectors = torch.vstack(
        [torch.hstack([
            torch.zeros(1, n) if j != i else torch.ones(1, n) for j in range(n)
        ]) for i in trange(n, desc='Building matrix A')]
    )
    A = torch.vstack((A, vectors))
    return A


def calculate_lipschitz_constant(n, gamma, p_order=3, device='cpu'):
    A = calculate_A_matrix(n).to(device)
    A_A_T = A @ A.T
    _, s, _ = torch.svd(A_A_T, compute_uv=False)
    if p_order == 3:
        return s.max() ** 2 * 15 / gamma ** 3
    else:
        raise NotImplementedError(f'Lipschitz constant calculation for p={p_order} is not implemented!')


def run_experiment(M_p, gamma, eps, image_index=0, max_steps=100, device='cpu'):
    images, labels = load_data()
    n = len(images[0])
    m = int(np.sqrt(n))
    gamma = torch.tensor(gamma, device=device)

    M_matrix = calculate_M_matrix(m)
    M_matrix = M_matrix.to(device)
    M_matrix_over_gamma = M_matrix / gamma

    # experiments were done for
    p_list = [34860, 31226, 239, 37372, 17390]
    q_list = [45815, 35817, 43981, 54698, 49947]

    # x_array = np.linspace(1 / 2e-2, 1 / 4e-4, 6)
    # epslist = 1 / x_array

    epsp = eps
    p, q = mnist(epsp, p_list[image_index], q_list[image_index], images, n)
    b = torch.cat((torch.tensor(p), torch.tensor(q)))
    b = b.to(device)

    lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=True, device=device)
    # lamb = torch.tensor([1e-6] * (n * 2), dtype=torch.double, requires_grad=True)
    half_lamb_len = int(len(lamb) / 2)

    optimizer = PrimalDualAccelerated(
        [lamb],
        M_p=M_p,
        p_order=torch.tensor(3, device=device),
        eps=0.01,
        calculate_primal_var=lambda lamb: calculate_x(lamb, half_lamb_len, gamma, M_matrix_over_gamma, device)
    )
    closure = lambda: phi(lamb, optimizer, half_lamb_len, gamma, M_matrix_over_gamma, b, device)
    optimize(optimizer, closure, eps, M_matrix, gamma, max_steps, device)
    return optimizer


if __name__ == '__main__':
    eps = 1e-3
    n = 784
    image_index = 0
    gamma = 0.9

    print(f'Calculating Lipschitz constant...')
    M_p = calculate_lipschitz_constant(n, gamma, p_order=3)

    run_experiment(M_p, gamma, eps, image_index)
