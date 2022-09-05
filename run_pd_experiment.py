from math import factorial
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import time
import json
import argparse

from tqdm.auto import tqdm, trange
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.spatial.distance import cdist
from mnist import MNIST
from IPython.display import clear_output
import cv2

from OPTAMI.higher_order import PrimalDualTensorMethod
from OPTAMI.higher_order.gradient_norm import GradientNormTensorMethod


# TODO: Split this file to several distinct files
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


def dump_tensorboard_info(i, f_value, cr_1, cr_2, phi_value, writer):
    writer.add_scalar(tag="phi_value", scalar_value=phi_value, global_step=i)
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


def init_tqdm(M_p, max_steps):
    if max_steps is not None:
        t = tqdm(desc=f"M_p = {M_p}", total=max_steps, leave=True)
    else:
        t = tqdm(desc=f"M_p = {M_p}", leave=True)
    return t


def update_tqdm(
    t,
    init_cr_1,
    cr_1,
    init_cr_2,
    cr_2,
    init_phi_value,
    phi_value,
    init_f_value,
    f_value,
    init_grad_phi_norm=None,
    grad_phi_norm=None,
    A=None,
):
    postfix = {
        "dual gap": f"{init_cr_1:.3f}->{cr_1:.3f}",
        "eq": f"{init_cr_2:.3f}->{cr_2:.3f}",
        "phi": f"{init_phi_value:.3f}->{phi_value:.3f}",
        "f": f"{init_f_value:.3f}->{f_value:.3f}",
    }
    if init_grad_phi_norm is not None and grad_phi_norm is not None:
        postfix["grad_phi.norm()"] = f"{init_grad_phi_norm}->{grad_phi_norm}"
    if A is not None:
        postfix["A"] = A
    t.update()
    t.set_postfix(postfix)


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


def init_data(image_index, new_m, eps, device):
    images, labels = load_data()
    if new_m is not None:
        n = new_m**2
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


def init_primal_dual_tm(
    lamb, n, M_p, gamma, M_matrix_over_gamma, ones, device
) -> PrimalDualTensorMethod:
    calculate_primal_var = lambda lamb: calculate_x(
        lamb, n, gamma, M_matrix_over_gamma, ones
    )
    optimizer = PrimalDualTensorMethod(
        [lamb],
        M_p=M_p,
        p_order=torch.tensor(3, device=device),
        calculate_primal_var=calculate_primal_var,
    )
    return optimizer


def calculate_R_for_gradient_norm_tm(n, M_matrix, p, q, gamma) -> torch.float:
    N = n
    C = M_matrix
    C_norm = C.norm(p=torch.inf)
    log_factor = torch.log(min(p.min(), q.min()))
    R = (N / 2) ** 0.5 * (C_norm - gamma / 2 * gamma / 2 * log_factor)
    return R


# TODO: change all ones to torch.ones
def init_gradient_norm_tm(lamb, n, M_p, gamma, eps, M_matrix, p, q, ones, device):
    R = calculate_R_for_gradient_norm_tm(n, M_matrix, p, q, gamma)
    calculate_primal_var = lambda lamb: calculate_x(
        lamb, n, gamma, M_matrix / gamma, ones
    )
    eps = min(eps / (2 * R), eps)  # since \e_{eq} == \e_{f}
    optimizer = GradientNormTensorMethod(
        [lamb],
        M_p,
        eps,
        R,
        p_order=torch.tensor(3, device=device),
        calculate_primal_var=calculate_primal_var,
    )
    return optimizer


def run_primal_dual(
    optimizer, closure, eps, M_matrix, gamma, round_function, max_steps, device, writer
):
    cr_1_arr = []
    cr_2_arr = []
    f_arr = []

    group = optimizer.param_groups[0]
    state = optimizer.state["default"]
    M_p = group["M_p"]
    t = init_tqdm(M_p, max_steps)

    phi_arr = []
    i = 0
    while True:
        optimizer.zero_grad()
        optimizer.step(closure)
        torch.cuda.empty_cache()

        with torch.no_grad():
            X = state["x_hat"][0]
            phi_value = state["phi_next"][0].clone().item()

            phi_arr.append(phi_value)

            f_value = f(X, M_matrix, gamma, device).item()
            f_arr.append(f_value)

            cr_1 = abs(phi_value + f_value)
            cr_2 = abs((M_matrix * (round_function(X) - X)).sum().item())
            cr_1_arr.append(cr_1)
            cr_2_arr.append(cr_2)
            # cr_2 = torch.norm(A_matrix @ x_hat - b)
            if i == 0:
                init_cr_1 = cr_1
                init_cr_2 = cr_2
                init_phi_value = phi_value
                init_f_value = f_value

            if writer is not None:
                dump_tensorboard_info(i, f_value, cr_1, cr_2, phi_value, writer)

            update_tqdm(
                t,
                init_cr_1,
                cr_1,
                init_cr_2,
                cr_2,
                init_phi_value,
                phi_value,
                init_f_value,
                f_value,
            )

            if cr_1 < eps and cr_2 < eps:
                break

            if max_steps is not None:
                if i == max_steps - 1:
                    break

            i += 1
    return i, cr_1_arr, cr_2_arr, phi_arr, f_arr


def run_gradient_norm(
    optimizer, closure, eps, M_matrix, gamma, round_function, max_steps, device, writer, manual_restart
):
    dual_gap_arr = []
    constraint_arr = []
    f_arr = []
    phi_arr = []
    grad_phi_arr = []

    group = optimizer.param_groups[0]
    mu = group["mu"]
    lamb = group["params"][0].detach()
    lamb_0 = lamb.clone()
    state = optimizer.state["default"]
    M_p = group["M_p"]
    M_mu = group["M_mu"]
    R = state["R"]
    p_order = optimizer.p_order
    one_over_p = 1 / p_order
    eps = min(eps / (2 * R), eps)
    eps_tilde = (eps / 2) ** (1 + one_over_p) / (
        4 * factorial(p_order + 2) * M_mu**one_over_p
    )
    criterion = mu * R**2 / 2

    if manual_restart:
        restart_after = 50

    outer_tqdm = init_tqdm(M_p, max_steps)
    i = 0
    n_of_inner_steps = 0
    while criterion >= eps_tilde:
        if max_steps is not None:
            if i == max_steps - 1:
                break

        A = 0
        inner_max_steps = 1000
        postfix = {"A": A}
        inner_tqdm = trange(inner_max_steps, postfix=postfix)
        inner_i = 0
        while A < 4 / mu:
            optimizer.zero_grad()
            optimizer.step(closure)
            torch.cuda.empty_cache()

            if inner_i == inner_max_steps - 1:
                raise Exception(
                    f"Number of iterations of inner tensor method exceeds {inner_max_steps}"
                )

            with torch.no_grad():
                inner_tm_state = optimizer.inner_tensor_method.state["default"]
                A = inner_tm_state["A"].item()
                X = state["x"]
                phi_mu_value = state["phi_mu"]
                lamb = group['params'][0]
                phi_value = (phi_mu_value - mu / 2 * torch.norm(lamb - lamb_0)**2).item()
                grad_phi_mu_norm = state["grad_phi_mu"].norm()
                phi_arr.append(phi_value)
                f_value = f(X, M_matrix, gamma, device).item()
                f_arr.append(f_value)

                dual_gap = abs(phi_value + f_value)
                constraint = abs((M_matrix * (round_function(X) - X)).sum().item())
                dual_gap_arr.append(dual_gap)
                constraint_arr.append(constraint)
                # cr_2 = torch.norm(A_matrix @ x_hat - b)
                if inner_i == 0:
                    inner_init_dual_gap = dual_gap
                    inner_init_constraint = constraint
                    inner_init_phi_value = phi_value
                    inner_init_f_value = f_value
                    inner_init_grad_phi_mu_norm = grad_phi_mu_norm
                    if i == 0:
                        init_dual_gap = inner_init_dual_gap
                        init_constraint = inner_init_constraint
                        init_phi_value = inner_init_phi_value
                        init_f_value = inner_init_f_value
                        init_grad_phi_mu_norm = inner_init_grad_phi_mu_norm

                if writer is not None:
                    dump_tensorboard_info(
                        inner_i, f_value, dual_gap, constraint, phi_value, writer
                    )

                update_tqdm(
                    inner_tqdm,
                    inner_init_dual_gap,
                    dual_gap,
                    inner_init_constraint,
                    constraint,
                    inner_init_phi_value,
                    phi_value,
                    inner_init_f_value,
                    f_value,
                    inner_init_grad_phi_mu_norm,
                    grad_phi_mu_norm,
                    A,
                )

                inner_i += 1
                if manual_restart:
                    if inner_i == restart_after:
                        break

        inner_tqdm.close()

        assert state["k"] == inner_i, f'{state["k"]}, {inner_i}'
        n_of_inner_steps += inner_i

        R = state["R"]
        criterion = mu * R**2 / 2

        update_tqdm(
            outer_tqdm,
            init_dual_gap,
            dual_gap,
            init_constraint,
            constraint,
            init_phi_value,
            phi_value,
            init_f_value,
            f_value,
            init_grad_phi_mu_norm,
            grad_phi_mu_norm,
        )

        i += 1

    optimizer.final_tensor_step(closure)
    X = state["x"]
    phi_mu_value = state["phi_mu"]
    phi_value = (phi_mu_value - mu / 2 * torch.norm(lamb - lamb_0)).item()
    grad_phi_mu_norm = state["grad_phi_mu"].norm()
    phi_arr.append(phi_value)
    f_value = f(X, M_matrix, gamma, device).item()
    f_arr.append(f_value)

    dual_gap = abs(phi_value + f_value)
    constraint = abs((M_matrix * (round_function(X) - X)).sum().item())
    dual_gap_arr.append(dual_gap)
    constraint_arr.append(constraint)

    print("Final results:")
    print(f"Total number of inner steps = {n_of_inner_steps}")
    print(f"Number of outer steps = {i}")
    print(f"phi_value: {init_phi_value:.3f}->{phi_value:.3f}")
    print(f"grad_phi_mu.norm(): {init_grad_phi_mu_norm:.3f}->{grad_phi_mu_norm:.3f}")
    print(f"f_value: {init_f_value:.3f}->{f_value:.3f}")
    print(f"|f_value - phi_value|: {init_dual_gap:.3f}->{dual_gap:.3f}")
    print(f"||Ax - b||: {init_constraint:.3f}->{constraint:.3f}")

    return i, dual_gap_arr, constraint_arr, phi_arr, f_arr


def optimize(
    optimizer,
    closure,
    round_function,
    eps,
    M_matrix,
    gamma,
    max_steps=None,
    device="cpu",
    writer=None,
    manual_restart=False
):
    if type(optimizer) == PrimalDualTensorMethod:
        i, cr_1_arr, cr_2_arr, phi_arr, f_arr = run_primal_dual(
            optimizer,
            closure,
            eps,
            M_matrix,
            gamma,
            round_function,
            max_steps,
            device,
            writer,
        )
    elif type(optimizer) == GradientNormTensorMethod:
        i, cr_1_arr, cr_2_arr, phi_arr, f_arr = run_gradient_norm(
            optimizer,
            closure,
            eps,
            M_matrix,
            gamma,
            round_function,
            max_steps,
            device,
            writer,
            manual_restart
        )
    return i, cr_1_arr, cr_2_arr, phi_arr, f_arr


def run_experiment(
    M_p,
    gamma,
    eps,
    optimizer_type,
    image_index=0,
    new_m=None,
    max_steps=None,
    device="cpu",
    tensorboard=False,
    manual_restart=False
):

    n, M_matrix, p, q, p_ref, q_ref = init_data(image_index, new_m, eps, device)
    gamma = torch.tensor(gamma, device=device)
    M_matrix_over_gamma = M_matrix / gamma
    ones = torch.ones(n, device=device, dtype=torch.double)

    lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=False, device=device)
    lamb.mul_(-1 / gamma).requires_grad_(True)

    if optimizer_type == GradientNormTensorMethod:
        optimizer = init_gradient_norm_tm(
            lamb, n, M_p, gamma, eps, M_matrix, p, q, ones, device
        )
        group = optimizer.param_groups[0]
        mu = group["mu"]
        lamb_0 = lamb.detach().clone()
        closure = (
            lambda: phi(
                lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=optimizer
            )
            + mu / 2 * torch.norm(lamb - lamb_0) ** 2
        )

    elif optimizer_type == PrimalDualTensorMethod:
        optimizer = init_primal_dual_tm(
            lamb, n, M_p, gamma, M_matrix_over_gamma, ones, device
        )
        closure = lambda: phi(
            lamb, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=optimizer
        )

    round_function = lambda X_matrix: B_round(X_matrix, p_ref, q_ref, ones)

    if tensorboard:
        if optimizer_type == PrimalDualTensorMethod:
            name = "PD"
        elif optimizer_type == GradientNormTensorMethod:
            name = "GN"
        else:
            raise NotImplementedError
        writer = SummaryWriter(f"tensorboard/{name}_gamma_{gamma}_M_p_{M_p}")
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
        writer,
        manual_restart
    )
    return optimizer, i, cr_1_list, cr_2_list, phi_list, f_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments for Optimal Transport')
    parser.add_argument('tensor_method_type', type=str, help='"gn" or "pd"')
    parser.add_argument('M_p', type=float)
    parser.add_argument('--manual_restart', action='store_true')
    args = parser.parse_args()

    M_p = args.M_p
    if args.tensor_method_type == 'pd':
        tensor_method_type = PrimalDualTensorMethod
    elif args.tensor_method_type == 'gn':
        tensor_method_type = GradientNormTensorMethod
    else:
        raise NotImplementedError
    manual_restart = args.manual_restart
    if manual_restart:
        assert args.tensor_method_type == 'gn', 'Manual restart is possible only for GradientNormTensorMethod'

    device = "cuda:4"

    gamma = 0.5
    eps = 0.001
    image_index = 1
    # m = 10
    m = None
    
    # M_p = calculate_lipschitz_constant(gamma, p_order=3)
    
    #     torch.autograd.set_detect_anomaly(True)
    optimizer, i, cr_1_list, cr_2_list, phi_list, f_list = run_experiment(
        M_p,
        gamma,
        eps,
        tensor_method_type,
        image_index,
        m,
        max_steps=10000,
        device=device,
        tensorboard=True,
        manual_restart=manual_restart
    )
    result = {
        'dual_gap': cr_1_list,
        'constraint': cr_2_list,
        'phi': phi_list,
        'f': f_list
    }

    if manual_restart:
        assert args.tensor_method_type == 'gn'
        fn = f'{args.tensor_method_type}_M_p_{M_p}_mr.json'
    else:
        fn = f'{args.tensor_method_type}_M_p_{M_p}.json'
    with open(fn, 'w') as f:
        json.dump(result, f)