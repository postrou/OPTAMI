import torch
import numpy as np
from scipy.spatial.distance import cdist
from mnist import MNIST
from tqdm import trange
from IPython.display import clear_output

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


# def calculate_x(lamb, half_lamb_len, gamma, M_matrix_over_gamma, device='cpu'):
#     psi = lamb[:half_lamb_len]
#     eta = lamb[half_lamb_len:]
#     psi_outer = torch.outer(psi, torch.ones(half_lamb_len, device=device))
#     eta_outer = torch.outer(torch.ones(half_lamb_len, device=device), eta)
#     lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
#     under_exp_vector = (lamb_factor_over_gamma - M_matrix_over_gamma).T.reshape(-1)
#     return torch.softmax(under_exp_vector, dim=0)


def calculate_x(lamb, n, M_matrix_over_gamma, ones):
    A = (-M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:]))
    return torch.softmax(A.view(-1), dim=0)


# def phi(lamb, half_lamb_len, gamma, M_matrix_over_gamma, b, optimizer=None, device='cpu'):
#     if optimizer is not None:
#         optimizer.zero_grad()
#     psi = lamb[:half_lamb_len]
#     eta = lamb[half_lamb_len:]
#     psi_outer = torch.outer(psi, torch.ones(len(psi), device=device))
#     eta_outer = torch.outer(torch.ones(len(eta), device=device), eta)
#     lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
#     under_exp_vector = (lamb_factor_over_gamma - M_matrix_over_gamma).T.reshape(-1)
#     return gamma * torch.logsumexp(under_exp_vector, dim=0) - lamb @ b

# def phi_nazar(gamma, lamb, C, n, one, p, q):
#     A = (-C / gamma + torch.outer(lamb[:n], one) + torch.outer(one, lamb[n:]))
#     a = A.max()
#     A -= a
#     s = a + torch.log(torch.exp(A).sum())
#     return gamma * (-lamb[:n].dot(p) - lamb[n:].dot(q) + s)
# 
# def phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=None, device='cpu'):
#     A = (-M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:]))
#     a = A.max()
#     A -= a
#     s = a + torch.log(torch.exp(A).sum())
#     return gamma * (-lamb[:n].dot(p) - lamb[n:].dot(q) + s)

def phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()
    A = (-M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:]))
    s = torch.logsumexp(A.view(-1), dim=0)
    return gamma * (-lamb[:n].dot(p) - lamb[n:].dot(q) + s)


# just in case
def grad_phi(lamb, M_matrix_over_gamma, A_matrix, b, ones):
    exp_matrix = np.exp(torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:]) - M_matrix_over_gamma)
    exp_matrix_vector = exp_matrix.T.reshape(-1)
    
    numerator = np.sum(exp_matrix_vector.reshape(-1) * A_matrix, axis=1)
    denominator = exp_matrix.sum()
    return numerator / denominator - b


def f(x, M_matrix, gamma, device='cpu'):
    x_copy = x.detach().clone().to(device)
    x_copy_under_log = x_copy.clone()
    # TODO: check
    x_copy_under_log[x_copy == 0.] = 1

    M_matrix_to_vector = M_matrix.view(-1)  # M is symmetric
    return (M_matrix_to_vector * x_copy).sum() + gamma * (x_copy * torch.log(x_copy_under_log)).sum()


def B_round(x, p_ref, q_ref, ones):
    r = p_ref / x.dot(ones)
    r[r>1] = 1.
    F = np.diag(r).dot(x)
    c = q_ref / (x.T).dot(ones)
    c[c>1] = 1.
    F = F.dot(np.diag(c))
    err_r = p_ref - F.dot(ones)
    err_c = q_ref - (F.T).dot(ones)
    return F + np.outer(err_r, err_c) / abs(err_r).sum()


def optimize(optimizer, closure, eps, M_matrix, A_matrix, gamma, b, max_steps=100, device='cpu'):
    i = 1
    while True:
        optimizer.step(closure)

        with torch.no_grad():
            x_hat = optimizer.state['default']['x_hat'][0]
            phi_value = closure()
            f_value = f(x_hat, M_matrix, gamma, device)

            cr_1 = abs(phi_value + f_value)
            cr_2 = torch.norm(A_matrix @ x_hat - b)
            if i == 0:
                init_cr_1 = cr_1
                init_cr_2 = cr_2
            clear_output(wait=True)
            print('\n'.join([
                f'Step #{k + 1}',
                f'cr_1: {init_cr_1} -> {cr_1}',
                f'cr_2: {init_cr_2} -> {cr_2}'
            ]))

            # print('\n'.join(
            #     [
            #         f'Step #{i}, criterion={criterion}, phi={phi_value}, f={f_value}',
            #         f'lambda={lamb.detach()}',
            #         f'x_hat={x_hat.detach()}'
            #     ]), end='\n\n')

            if cr_1 < eps and cr_2 < eps or i == max_steps:
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
    return A.double()


def calculate_lipschitz_constant(n, gamma, p_order=3, A_A_T=None, device='cpu'):
    if A_A_T is None:
        A_matrix = calculate_A_matrix(n).to(device)
        A_A_T = A_matrix @ A_matrix.T
    else:
        A_A_T = A_A_T.to(device)
    _, s, _ = torch.svd(A_A_T, compute_uv=False)
    if p_order == 3:
        return s.max() ** 2 * 15 / gamma ** 3
    elif p_order == 1:
        return s.max() / gamma
    else:
        raise NotImplementedError(f'Lipschitz constant calculation for p={p_order} is not implemented!')


def run_experiment(M_p, gamma, eps, A_matrix=None, image_index=0, max_steps=100, device='cpu'):
    images, labels = load_data()
    n = len(images[0])
    m = int(np.sqrt(n))
    gamma = torch.tensor(gamma, device=device)

    M_matrix = calculate_M_matrix(m)
    M_matrix = M_matrix.to(device)
    M_matrix_over_gamma = M_matrix / gamma

    if A_matrix is None:
        A_matrix = calculate_A_matrix(n).to(device)

    # experiments were done for
    p_list = [34860, 31226, 239, 37372, 17390]
    q_list = [45815, 35817, 43981, 54698, 49947]

    # x_array = np.linspace(1 / 2e-2, 1 / 4e-4, 6)
    # epslist = 1 / x_array

    epsp = eps
    p, q = mnist(epsp, p_list[image_index], q_list[image_index], images, n)
    p = torch.tensor(p, device=device)
    q = torch.tensor(q, device=device)
    b = torch.cat((p, q))

    lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=True, device=device)
    # lamb = torch.tensor([1e-6] * (n * 2), dtype=torch.double, requires_grad=True)
    half_lamb_len = int(len(lamb) / 2)

    ones = torch.ones(n, device=device)
    optimizer = PrimalDualAccelerated(
        [lamb],
        M_p=M_p,
        p_order=torch.tensor(3, device=device),
        eps=0.01,
        calculate_primal_var=lambda lamb: calculate_x(lamb, n, M_matrix_over_gamma, ones)
    )
    closure = lambda: phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer)
    optimize(optimizer, closure, eps, M_matrix, A_matrix, gamma, b, max_steps, device)
    return optimizer


if __name__ == '__main__':
    eps = 1e-3
    n = 784
    image_index = 0
    gamma = 0.9

    print(f'Calculating Lipschitz constant...')
    M_p = calculate_lipschitz_constant(n, gamma, p_order=3)

    run_experiment(M_p, gamma, eps, image_index)
