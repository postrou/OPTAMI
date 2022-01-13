import os

import torch
import numpy as np
from scipy.spatial.distance import cdist
from mnist import MNIST
from tqdm import trange
import matplotlib.pylab as plt
from IPython.display import clear_output
import cv2

from OPTAMI import *


def load_data(path='./data/'):
    mndata = MNIST(path)
    return mndata.load_training()


def mnist(eps, p, q, images, m):
    p, q = np.float64(images[p]), np.float64(images[q])
    old_m = int(p.size ** 0.5)
    if old_m != m:
        p = cv2.resize(p.reshape(old_m, old_m), (m, m))
        q = cv2.resize(q.reshape(old_m, old_m), (m, m))
        p = p.reshape(-1)
        q = q.reshape(-1)
    n = m ** 2
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


def calculate_x(lamb, n, gamma, M_matrix_over_gamma, ones):
    log_X = -M_matrix_over_gamma + (torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])) / gamma
    max_log_X = log_X.max()
    log_X_stable = log_X - max_log_X
    X_stable = torch.exp(log_X_stable)
    X_stable_sum = X_stable.sum()
    return X_stable / X_stable_sum, X_stable_sum, max_log_X


def phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, X_stable_sum=None, max_log_X=None, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()
    if X_stable_sum is None or max_log_X is None:
        A = -M_matrix_over_gamma + (torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])) / gamma
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
    return A


def calculate_lipschitz_constant(n, gamma, p_order=3, A_A_T=None, device='cpu'):
#     if A_A_T is None:
#         A_matrix = calculate_A_matrix(n).to(device)
#         A_A_T = A_matrix @ A_matrix.T
#     else:
#         A_A_T = A_A_T.to(device)
#     _, s, _ = torch.svd(A_A_T, compute_uv=False)
    s = 2 ** 0.5
    if p_order == 3:
#         return s.max() ** 2 * 15 / gamma ** 3
#         return s.max() ** 2 * 15
        return s ** 4 * 15 / gamma ** 3
    elif p_order == 1:
#         return s.max() / gamma
#         return s.max()
        return s ** 2 / gamma
    else:
        raise NotImplementedError(f'Lipschitz constant calculation for p={p_order} is not implemented!')


def optimize(
        optimizer,
        closure,
        round_function,
        eps,
        M_matrix,
        gamma,
        max_steps=None,
        fgm_cr_1_list=None,
        fgm_cr_2_list=None,
        fgm_phi_list=None,
        fgm_f_list=None,
        device='cpu'
):
    i = 0

    import time
    start_time = time.time()

    cr_1_list = []
    cr_2_list = []
    phi_list = []
    f_list = []
    while True:
        optimizer.step(closure)
        torch.cuda.empty_cache()

        with torch.no_grad():
            X_hat_matrix_next = optimizer.state['default']['x_hat'][0]
            phi_value = optimizer.state['default']['phi_next'][0]
            f_value = f(X_hat_matrix_next, M_matrix, gamma, device)
            phi_list.append(phi_value.item())
            f_list.append(f_value.item())

            cr_1 = abs(phi_value + f_value)
            cr_2 = abs((M_matrix * (round_function(X_hat_matrix_next) - X_hat_matrix_next)).sum())
            cr_1_list.append(cr_1.detach().clone().item())
            cr_2_list.append(cr_2.detach().clone().item())
            # cr_2 = torch.norm(A_matrix @ x_hat - b)
            if i == 0:
                init_cr_1 = cr_1
                init_cr_2 = cr_2
                init_phi_value = phi_value.detach()
                init_f_value = f_value.item()

            clear_output(wait=True)

            time_whole = int(time.time() - start_time)
            time_h = time_whole // 3600
            time_m = time_whole % 3600 // 60
            time_s = time_whole % 3600 % 60
            print('\n'.join([
                f'Step #{i}',
                f'cr_1: {init_cr_1} -> {cr_1}',
                f'cr_2: {init_cr_2} -> {cr_2}',
                f'phi: {init_phi_value} -> {phi_value.item()}',
                f'f: {init_f_value} -> {f_value.item()}',
                f'time={time_h}h, {time_m}m, {time_s}s'
            ]))
            if fgm_cr_1_list is not None and fgm_cr_2_list is not None and fgm_phi_list is not None and fgm_f_list is not None:
                fig, ax = plt.subplots(2, 2, figsize=(20, 16))
                ax[0, 0].plot(fgm_cr_1_list, label='FGM')
                ax[0, 0].plot(cr_1_list, label='Tensor Method')
                ax[0, 0].set_xlabel('iter')
                ax[0, 0].set_ylabel('Dual gap')
                ax[0, 0].set_yscale('log')
                ax[0, 0].legend()

                ax[0, 1].plot(fgm_cr_2_list, label='FGM')
                ax[0, 1].plot(cr_2_list, label='Tensor Method')
                ax[0, 1].set_xlabel('iter')
                ax[0, 1].set_ylabel('Linear constraints')
                ax[0, 1].set_yscale('log')
                ax[0, 1].legend()

                ax[1, 0].plot(fgm_phi_list, label='FGM')
                ax[1, 0].plot(phi_list, label='Tensor Method')
                ax[1, 0].set_xlabel('iter')
                ax[1, 0].set_ylabel('Dual function value')
                ax[1, 0].legend()

                ax[1, 1].plot(fgm_f_list, label='FGM')
                ax[1, 1].plot(f_list, label='Tensor Method')
                ax[1, 1].set_xlabel('iter')
                ax[1, 1].set_ylabel('Primal function value')
                ax[1, 1].legend()
                plt.show()

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
    return i, cr_1_list, cr_2_list, phi_list, f_list


def run_experiment(
        M_p,
        gamma,
        eps,
        image_index=0,
        new_m=None,
        optimizer=None,
        max_steps=None,
        fgm_cr_1_list=None,
        fgm_cr_2_list=None,
        fgm_phi_list=None,
        fgm_f_list=None,
        device='cpu',
        debug=False
):
    images, labels = load_data()
    if new_m is not None:
        n = new_m ** 2
        m = new_m
    else:
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

    epsp = eps / 8

    p, q = mnist(epsp, p_list[image_index], q_list[image_index], images, m)
    p = torch.tensor(p, device=device, dtype=torch.double)
    q = torch.tensor(q, device=device, dtype=torch.double)

    p_ref, q_ref = mnist(0, p_list[image_index], q_list[image_index], images, m)
    p_ref = torch.tensor(p_ref, device=device, dtype=torch.double)
    q_ref = torch.tensor(q_ref, device=device, dtype=torch.double)

    ones = torch.ones(n, device=device, dtype=torch.double)
    
    if optimizer is None:
        lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=True, device=device)
        optimizer = PrimalDualAccelerated(
            [lamb],
            M_p=M_p,
            p_order=torch.tensor(3, device=device),
            eps=0.01,
            calculate_primal_var=lambda lamb: calculate_x(lamb, n, gamma, M_matrix_over_gamma, ones),
            debug=debug
        )
    else:
        lamb = optimizer.param_groups[0]['params'][0]
        
    closure = lambda: phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=optimizer)
    round_function = lambda X_matrix: B_round(X_matrix, p_ref, q_ref, ones)
    i, cr_1_list, cr_2_list, phi_list, f_list = optimize(
        optimizer,
        closure,
        round_function,
        eps,
        M_matrix,
        gamma,
        max_steps,
        fgm_cr_1_list,
        fgm_cr_2_list,
        fgm_phi_list,
        fgm_f_list,
        device
    )
    return optimizer, i, cr_1_list, cr_2_list, phi_list, f_list


if __name__ == '__main__':
    n = 784
    device = 'cpu'

    # A_A_T_path = 'A_A_T.pkl'
    # if not os.path.exists(A_A_T_path):
    #     A_matrix = calculate_A_matrix(n).to(device)
    #     A_A_T = A_matrix @ A_matrix.T
    #     torch.save(A_A_T, A_A_T_path)
    # else:
    #     A_A_T = torch.load(A_A_T_path)

    eps = 0.02
    gamma = 0.35
    image_index = 0

    M_p = calculate_lipschitz_constant(n, gamma, p_order=3, A_A_T=None, device=device)

    run_experiment(M_p, gamma, eps, image_index, max_steps=50, device=device, debug=True)
