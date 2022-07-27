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


# ok
def load_data(path='./data/'):
    mndata = MNIST(path)
    return mndata.load_training()


# ok
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

    # so there won't be any zeros
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


# ok
def calculate_M_matrix(m):
    M_matrix = np.arange(m)
    M_matrix = cartesian_product(M_matrix, M_matrix)
    M_matrix = cdist(M_matrix, M_matrix)
    M_matrix /= np.max(M_matrix)
    # M_matrix /= np.median(M_matrix)
    return torch.tensor(M_matrix, dtype=torch.double)


def calculate_x_old(lamb, n, gamma, M_matrix_over_gamma, ones):
    A = -M_matrix_over_gamma + (torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])) / gamma
    return torch.softmax(A.view(-1), dim=0)


# softmax
def calculate_x(lamb, n, gamma, M_matrix_over_gamma, ones):
    log_X = -M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])
    max_log_X = log_X.max()
    log_X_stable = log_X - max_log_X
    X_stable = torch.exp(log_X_stable)
    X_stable_sum = X_stable.sum()
    return X_stable / X_stable_sum, X_stable_sum, max_log_X


# ok
def phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q, X_stable_sum=None, max_log_X=None, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()
    if X_stable_sum is None or max_log_X is None:
        assert lamb.grad is None
        assert not M_matrix_over_gamma.requires_grad
        assert not ones.requires_grad
        assert not p.requires_grad
        assert not q.requires_grad
        if torch.is_tensor(gamma):
            assert not gamma.requires_grad
        A = -M_matrix_over_gamma + (torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:]))
        s = torch.logsumexp(A.view(-1), dim=0)
    else:
        s = torch.log(X_stable_sum) + max_log_X
    return gamma * (s - lamb[:n] @ p - lamb[n:] @ q)


def grad_phi(lamb, gamma, calculate_primal_var, p, q, ones, device='cpu'):
    X_stable, X_stable_sum, _ = calculate_primal_var(lamb)
    u_hat_stable, v_hat_stable = X_stable @ ones, X_stable.T @ ones

    grad_phi = gamma * torch.cat((-p + u_hat_stable, -q + v_hat_stable)).to(device)
    return grad_phi


# ok
def f(x, M_matrix, gamma, device='cpu'):
    x = x.detach().clone().to(device)
    y = x.clone().to(device)
    y[y == 0.] = 1.

    return (M_matrix * x).sum() + gamma * (x * torch.log(y)).sum()


# ok
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


def calculate_lipschitz_constant(gamma, p_order=3):
    # s = 2 ** 0.5
    if p_order == 1:
        return 2 * gamma
    elif p_order == 3:
        return 60 * gamma
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
        device='cpu',
        debug=False
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
            state = optimizer.state['default']
            X_hat_matrix_next = state['x_hat'][0]
            phi_value = state['phi_next'][0]
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
            # os.system('clear')

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
            if debug:
                grad_psi_norm = state['grad_psi_norm'].item()
                other_grad_psi_norm = state['other_grad_psi_norm'].item()
                if i == 0:
                    init_grad_psi_norm = grad_psi_norm
                    init_other_grad_psi_norm = other_grad_psi_norm
                print(f'grad_psi_norm: {init_grad_psi_norm} -> {grad_psi_norm}')
                print(f'other_grad_psi_norm: {init_other_grad_psi_norm} -> {other_grad_psi_norm}')

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
        device='cpu'
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

    epsp = eps / 8
    # epsp = eps

    p, q = mnist(epsp, p_list[image_index], q_list[image_index], images, m)
    p = torch.tensor(p, device=device, dtype=torch.double)
    q = torch.tensor(q, device=device, dtype=torch.double)

    p_ref, q_ref = mnist(0, p_list[image_index], q_list[image_index], images, m)
    p_ref = torch.tensor(p_ref, device=device, dtype=torch.double)
    q_ref = torch.tensor(q_ref, device=device, dtype=torch.double)

    ones = torch.ones(n, device=device, dtype=torch.double)

    if optimizer is None:
        lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=False, device=device)
        lamb.mul_(-1 / gamma).requires_grad_(True)

        caclulate_primal_var = lambda lamb: calculate_x(lamb, n, gamma, M_matrix_over_gamma, ones)
        optimizer = PrimalDualAccelerated(
            [lamb],
            M_p=M_p,
            p_order=torch.tensor(3, device=device),
            eps=0.01,
            calculate_primal_var=caclulate_primal_var,
            calculate_grad_phi=lambda lamb: grad_phi(lamb, gamma, caclulate_primal_var, p, q, ones, device)
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

    gamma = 1.2
    eps = 0.001
    image_index = 2
    new_m = 10

    M_p = calculate_lipschitz_constant(gamma, p_order=3)

    torch.autograd.set_detect_anomaly(True)
    run_experiment(M_p, gamma, eps, image_index, new_m, max_steps=500, device=device)
