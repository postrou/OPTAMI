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
    return torch.tensor(M_matrix, dtype=torch.double)


# def calculate_x(lamb, half_lamb_len, gamma, M_matrix_over_gamma, device='cpu'):
#     psi = lamb[:half_lamb_len]
#     eta = lamb[half_lamb_len:]
#     psi_outer = torch.outer(psi, torch.ones(half_lamb_len, device=device))
#     eta_outer = torch.outer(torch.ones(half_lamb_len, device=device), eta)
#     lamb_factor_over_gamma = (psi_outer + eta_outer) / gamma
#     under_exp_vector = (lamb_factor_over_gamma - M_matrix_over_gamma).T.reshape(-1)
#     return torch.softmax(under_exp_vector, dim=0)


# def calculate_x(lamb, n, M_matrix_over_gamma, ones):
#     A = (-M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:]))
#     return torch.softmax(A.view(-1), dim=0)


def calculate_x(lamb, n, M_matrix_over_gamma, ones):
    log_X = -M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])
    max_log_X = log_X.max()
    log_X_stable = log_X - max_log_X
    X_stable = torch.exp(log_X_stable)
    X_stable_sum = X_stable.sum()
    return X_stable / X_stable_sum, X_stable_sum, max_log_X


def phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, X_stable_sum=None, max_log_X=None, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()
    if X_stable_sum is None or max_log_X is None:
        A = (-M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:]))
        s = torch.logsumexp(A.view(-1), dim=0)
    else:
        s = torch.log(X_stable_sum) + max_log_X
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

#     M_matrix_to_vector = M_matrix.view(-1)  # M is symmetric
#     return (M_matrix_to_vector * x_copy).sum() + gamma * (x_copy * torch.log(x_copy_under_log)).sum()
    return (M_matrix * x_copy).sum() + gamma * (x_copy * torch.log(x_copy_under_log)).sum()


def B_round(x, p_ref, q_ref, ones):
    r = p_ref / (x @ ones)
    r[r > 1] = 1.
    F = torch.diag(r) @ x
    c = q_ref / (x.T @ ones)
    c[c > 1] = 1.
    F = F @ torch.diag(c)
    err_r = p_ref - (F @ ones)
    err_c = q_ref - (F.T @ ones)
    return F + torch.outer(err_r, err_c) / torch.abs(err_r).sum()


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


def optimize(optimizer, closure, round_function, eps, M_matrix, gamma, max_steps=None, device='cpu'):
    i = 0
    while True:
        optimizer.step(closure)

        with torch.no_grad():
            X_hat_matrix_next = optimizer.state['default']['x_hat'][0]
            phi_value = optimizer.state['default']['phi_next']
            f_value = f(X_hat_matrix_next, M_matrix, gamma, device)

            cr_1 = abs(phi_value + f_value)
            cr_2 = (M_matrix * (round_function(X_hat_matrix_next) - X_hat_matrix_next)).sum()
            # cr_2 = torch.norm(A_matrix @ x_hat - b)
            if i == 0:
                init_cr_1 = cr_1
                init_cr_2 = cr_2
            clear_output(wait=True)
            print('\n'.join([
                f'Step #{i}',
                f'cr_1: {init_cr_1} -> {cr_1}',
                f'cr_2: {init_cr_2} -> {cr_2}'
            ]))

            # print('\n'.join(
            #     [
            #         f'Step #{i}, criterion={criterion}, phi={phi_value}, f={f_value}',
            #         f'lambda={lamb.detach()}',
            #         f'x_hat={x_hat.detach()}'
            #     ]), end='\n\n')

            if cr_1 < eps and cr_2 < eps:
                break

            if max_steps is not None:
                if i == max_steps - 1:
                    break

            i += 1
    return i, cr_1, cr_2


def run_experiment(M_p, gamma, eps, image_index=0, max_steps=100, device='cpu'):
    images, labels = load_data()
    n = len(images[0])
    m = int(np.sqrt(n))
    gamma = torch.tensor(gamma, device=device)

    M_matrix = calculate_M_matrix(m)
    M_matrix = M_matrix.to(device)
    M_matrix_over_gamma = M_matrix / gamma

    # if A_matrix is None:
    #     A_matrix = calculate_A_matrix(n).to(device)

    # experiments were done for
    p_list = [34860, 31226, 239, 37372, 17390]
    q_list = [45815, 35817, 43981, 54698, 49947]

    # x_array = np.linspace(1 / 2e-2, 1 / 4e-4, 6)
    # epslist = 1 / x_array

    epsp = eps
    p, q = mnist(epsp, p_list[image_index], q_list[image_index], images, n)
    p = torch.tensor(p, device=device, dtype=torch.double)
    q = torch.tensor(q, device=device, dtype=torch.double)
    p_ref, q_ref = mnist(0, p_list[image_index], q_list[image_index], images, n)
    p_ref = torch.tensor(p_ref, device=device, dtype=torch.double)
    q_ref = torch.tensor(q_ref, device=device, dtype=torch.double)

    lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=True, device=device)

    ones = torch.ones(n, device=device, dtype=torch.double)
    optimizer = PrimalDualAccelerated(
        [lamb],
        M_p=M_p,
        p_order=torch.tensor(3, device=device),
        eps=0.01,
        calculate_primal_var=lambda lamb: calculate_x(lamb, n, M_matrix_over_gamma, ones)
    )
    closure = lambda: phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=optimizer)
    round_function = lambda X_matrix: B_round(X_matrix, p_ref, q_ref, ones)
    i, cr_1, cr_2 = optimize(optimizer, closure, round_function, eps, M_matrix, gamma, max_steps, device)
    return optimizer, i, cr_1, cr_2


if __name__ == '__main__':
    eps = 1e-3
    n = 784
    image_index = 0
    gamma = 0.9

    print(f'Calculating Lipschitz constant...')
    M_p = calculate_lipschitz_constant(n, gamma, p_order=3)

    run_experiment(M_p, gamma, eps, image_index)
