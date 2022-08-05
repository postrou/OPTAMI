import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output

def fast_gradient_method(
        n,
        L,
        M_matrix,
        primal_function,
        dual_function,
        primal_var_function,
        round_function,
        eps,
        eps_tilde,
        max_steps=None,
        device='cpu',
        writer=None
):
    dual_var_next = torch.zeros(2 * n, requires_grad=True, dtype=torch.double, device=device)
    y_next = torch.zeros(2 * n, dtype=torch.double, device=device)
    z_next = torch.zeros(2 * n, dtype=torch.double, device=device)
    k = 0
    one_over_L = 1 / L
    alpha_next = one_over_L
    sum_of_y = y_next
    cr_1_list = []
    cr_2_list = []
    phi_list = []
    f_list = []

    if max_steps is not None:
        t = tqdm(desc=f'M_1 = {L}', total=max_steps, leave=True)
    else:
        t = tqdm(desc=f'M_1 = {L}', leave=True)

    while True:
        y = y_next.detach().clone()
        z = z_next.detach().clone()
        tau = 2 / (k + 2)

        with torch.no_grad():
            dual_var_next = tau * z + (1 - tau) * y
            dual_var_next.requires_grad_(True)
            # dual_var_next.add_(z, alpha=tau).add_(y, alpha=1 - tau)
        assert dual_var_next.requires_grad

        phi_next = dual_function(dual_var_next)
        grad_phi_next = torch.autograd.grad(phi_next, dual_var_next)
        y_next = dual_var_next.detach().clone() - one_over_L * grad_phi_next[0]

        z_next = z - alpha_next * grad_phi_next[0]

        N = k + 1

        y_tilde = 1 / (N * (N + 3)) * (sum_of_y + (N + 1) ** 2 * y_next)
        sum_of_y += y_next

        primal_var, _, _ = primal_var_function(dual_var_next)

        phi_value = dual_function(y_tilde)
        f_value = primal_function(primal_var)
        phi_list.append(phi_value.item())
        f_list.append(f_value.item())
        cr_1 = torch.abs(phi_value + f_value)
        cr_2 = abs((M_matrix * (round_function(primal_var) - primal_var)).sum())
        cr_1_list.append(cr_1.item())
        cr_2_list.append(cr_2.item())

        if writer is not None:
            dump_tensorboard_info(k, phi_value, f_value, cr_1, cr_2, writer)
        #         cr_2 = torch.norm(A_matrix @ primal_var - b)
        if k == 0:
            init_cr_1 = cr_1
            init_cr_2 = cr_2
            init_phi_value = phi_value.item()
            init_f_value = f_value.item()
        # clear_output(wait=True)
        # print('\n'.join([
        #     f'Step #{k}',
        #     f'cr_1: {init_cr_1} -> {cr_1}',
        #     f'cr_2: {init_cr_2} -> {cr_2}',
        #     f'phi: {init_phi_value} -> {phi_value.item()}',
        #     f'f: {init_f_value} -> {f_value.item()}',
        # ]))

        tqdm_postfix_dict = {
            'dual gap': f'{init_cr_1:.3f}->{cr_1.item():.3f}',
            'eq': f'{init_cr_2:.3f}->{cr_2.item():.3f}',
            'phi': f'{init_phi_value:.3f}->{phi_value.item():.3f}',
            'f': f'{init_f_value:.3f}->{f_value.item():.3f}',
        }
        t.update()
        t.set_postfix(tqdm_postfix_dict)

        if cr_1 <= eps and cr_2 <= eps_tilde:
            break

        alpha_next = (k + 2) * one_over_L / 2
        k += 1

        if max_steps is not None:
            if k == max_steps:
                break

    return primal_var, dual_var_next, phi_list, f_list, cr_1_list, cr_2_list

def dump_tensorboard_info(i, phi_value, f_value, cr_1, cr_2, writer):
    writer.add_scalar(tag="phi_value", scalar_value=phi_value, global_step=i)
    writer.add_scalar("f_value", f_value, i)
    writer.add_scalar("|f_value + phi_value|", cr_1, i)
    writer.add_scalar("||Ax - b||", cr_2, i)
    writer.flush()
