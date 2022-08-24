import os
import time

from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.spatial.distance import cdist
from mnist import MNIST
from IPython.display import clear_output
import cv2

from OPTAMI.higher_order import PrimalDualAccelerated


def run_experiment(
    M_p,
    gamma,
    eps,
    image_index=0,
    new_m=None,
    optimizer=None,
    max_steps=None,
    device="cpu",
    tensorboard=False
):

    n, M_matrix, p, q, p_ref, q_ref = \
        init_data(image_index, new_m, eps, device)
    gamma = torch.tensor(gamma, device=device)
    M_matrix_over_gamma = M_matrix / gamma
    ones = torch.ones(n, device=device, dtype=torch.double)

    if optimizer is None:
        lamb = torch.zeros(
            n * 2, dtype=torch.double, requires_grad=False, device=device
        )
        lamb.mul_(-1 / gamma).requires_grad_(True)

        caclulate_primal_var = lambda lamb: calculate_x(
            lamb, n, gamma, M_matrix_over_gamma, ones
        )
        optimizer = PrimalDualAccelerated(
            [lamb],
            M_p=M_p,
            p_order=torch.tensor(3, device=device),
            calculate_primal_var=caclulate_primal_var
        )
    else:
        lamb = optimizer.param_groups[0]["params"][0]

    closure = lambda: phi(
        lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=optimizer
    )
    round_function = lambda X_matrix: B_round(X_matrix, p_ref, q_ref, ones)

    if tensorboard:
        writer = SummaryWriter(f'tensorboard/TM_gamma_{gamma}_M_p_{M_p}')
    else:
        writer = None

    i, cr_1_list, cr_2_list, phi_list, f_list = optimize(
        optimizer,
        closure,
        round_function,
        eps,
        M_matrix,
        gamma,
        max_steps,
        device,
        writer
    )
    return optimizer, i, cr_1_list, cr_2_list, phi_list, f_list
    
    
def init_data(image_index, new_m, eps, device):
    images, labels = load_data()
    if new_m is not None:
        n = new_m ** 2
        m = new_m
    else:
        n = len(images[0])
        m = int(np.sqrt(n))

    M_matrix = calculate_M_matrix(m)
    M_matrix = M_matrix.to(device)

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
 
    return n, M_matrix, p, q, p_ref, q_ref


def optimize(
    optimizer,
    closure,
    round_function,
    eps,
    M_matrix,
    gamma,
    max_steps=None,
    device="cpu",
    writer=None
):
    i = 0

    start_time = time.time()

    cr_1_arr = []
    cr_2_arr = []
    f_arr = []

    M_p = optimizer.param_groups[0]['M_p']
    if max_steps is not None:
        t = tqdm(desc=f'M_p = {M_p}', total=max_steps, leave=True)
    else:
        t = tqdm(desc=f'M_p = {M_p}', leave=True)

    if optimizer.keep_psi_data:
        phi_arr = state["phi_arr"]
    else:
        phi_arr = []
    while True:
        optimizer.step(closure)
        torch.cuda.empty_cache()

        with torch.no_grad():
            state = optimizer.state["default"]
            X_hat_matrix_next = state["x_hat"][0]
            phi_value = state['phi_next'][0]
            if not optimizer.keep_psi_data:
                phi_arr.append(phi_value)

            f_value = f(X_hat_matrix_next, M_matrix, gamma, device)
            f_arr.append(f_value.item())

            cr_1 = abs(phi_value + f_value)
            cr_2 = abs(
                (
                    M_matrix * (round_function(X_hat_matrix_next) - X_hat_matrix_next)
                ).sum()
            )
            cr_1_arr.append(cr_1.detach().clone().item())
            cr_2_arr.append(cr_2.detach().clone().item())
            # cr_2 = torch.norm(A_matrix @ x_hat - b)
            if i == 0:
                init_cr_1 = cr_1.item()
                init_cr_2 = cr_2.item()
                init_phi_value = phi_value.item()
                init_f_value = f_value.item()

            if writer is not None:
                dump_tensorboard_info(i, f_value, cr_1, cr_2, state, writer)

            tqdm_postfix_dict = {
                'dual gap': f'{init_cr_1:.3f}->{cr_1.item():.3f}',
                'eq': f'{init_cr_2:.3f}->{cr_2.item():.3f}',
                'phi': f'{init_phi_value:.3f}->{phi_value.item():.3f}',
                'f': f'{init_f_value:.3f}->{f_value.item():.3f}',
            }
            t.update()
            t.set_postfix(tqdm_postfix_dict)

            if cr_1 < eps and cr_2 < eps:
                break

            if max_steps is not None:
                if i == max_steps - 1:
                    break

            i += 1
    return i, cr_1_arr, cr_2_arr, phi_arr, f_arr


def load_data(path="./data/"):
    mndata = MNIST(path)
    return mndata.load_training()


def mnist(eps, p, q, images, m):
    p, q = np.float64(images[p]), np.float64(images[q])
    old_m = int(p.size**0.5)
    if old_m != m:
        p = cv2.resize(p.reshape(old_m, old_m), (m, m))
        q = cv2.resize(q.reshape(old_m, old_m), (m, m))
        p = p.reshape(-1)
        q = q.reshape(-1)
    n = m**2
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
    A = (
        -M_matrix_over_gamma
        + (torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])) / gamma
    )
    return torch.softmax(A.view(-1), dim=0)


# softmax
def calculate_x(lamb, n, gamma, M_matrix_over_gamma, ones):
    log_X = (
        -M_matrix_over_gamma + torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])
    )
    max_log_X = log_X.max()
    log_X_stable = log_X - max_log_X
    X_stable = torch.exp(log_X_stable)
    X_stable_sum = X_stable.sum()
    return X_stable / X_stable_sum, X_stable_sum, max_log_X


def phi(
    lamb,
    n,
    gamma,
    M_matrix_over_gamma,
    ones,
    p,
    q,
    X_stable_sum=None,
    max_log_X=None,
    optimizer=None,
):
    if optimizer is not None:
        optimizer.zero_grad()
    if X_stable_sum is None or max_log_X is None:
        assert not M_matrix_over_gamma.requires_grad
        assert not ones.requires_grad
        assert not p.requires_grad
        assert not q.requires_grad
        if torch.is_tensor(gamma):
            assert not gamma.requires_grad
        A = -M_matrix_over_gamma + (
            torch.outer(lamb[:n], ones) + torch.outer(ones, lamb[n:])
        )
        s = torch.logsumexp(A.view(-1), dim=0)
    else:
        s = torch.log(X_stable_sum) + max_log_X
    return gamma * (s - lamb[:n] @ p - lamb[n:] @ q)


def grad_phi(lamb, gamma, calculate_primal_var, p, q, ones, device="cpu"):
    X_stable, X_stable_sum, _ = calculate_primal_var(lamb)
    u_hat_stable, v_hat_stable = X_stable @ ones, X_stable.T @ ones

    grad_phi = gamma * torch.cat((-p + u_hat_stable, -q + v_hat_stable)).to(device)
    return grad_phi


def f(x, M_matrix, gamma, device="cpu"):
    x = x.detach().clone().to(device)
    y = x.clone().to(device)
    y[y == 0.0] = 1.0

    return (M_matrix * x).sum() + gamma * (x * torch.log(y)).sum()


def B_round(x, p_ref, q_ref, ones):
    r = p_ref / (x @ ones)
    r[r > 1] = 1.0
    F = torch.diag(r) @ x
    c = q_ref / (x.T @ ones)
    c[c > 1] = 1.0
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
        raise NotImplementedError(
            f"Lipschitz constant calculation for p={p_order} is not implemented!"
        )


def dump_tensorboard_info(i, f_value, cr_1, cr_2, state, writer):
    psi_value = state["psi_value"]
    phi_value = state["phi_arr"][-1]
    grad_psi_norm = state["grad_psi_norm"]
    v = state["v"][0]

    writer.add_scalar(tag="phi_value", scalar_value=phi_value.item(), global_step=i)
    writer.add_scalar("f_value", f_value, i)
    writer.add_scalar("|f_value + phi_value|", cr_1, i)
    writer.add_scalar("||Ax - b||", cr_2, i)
    writer.flush()


def get_time(start_time):
    time_whole = int(time.time() - start_time)
    time_h = time_whole // 3600
    time_m = time_whole % 3600 // 60
    time_s = time_whole % 3600 % 60
    return time_h, time_m, time_s


if __name__ == "__main__":
    n = 784
    device = "cpu"

    gamma = 1.2
    eps = 0.001
    image_index = 2
    new_m = 10

    M_p = calculate_lipschitz_constant(gamma, p_order=3)

    torch.autograd.set_detect_anomaly(True)
    run_experiment(M_p, gamma, eps, image_index, new_m, max_steps=500, device=device)
