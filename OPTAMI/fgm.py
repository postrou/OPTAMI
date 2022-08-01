import torch
from IPython.display import clear_output

def fast_gradient_method(
        n,
        L,
        M_matrix,
        primal_function,
        dual_function,
        primal_var_function,
        round_function,
        b,
        eps,
        eps_tilde,
        max_steps=None,
        device='cpu'
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
    F_list = []
    g_list = []
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

        F_value = dual_function(y_tilde)
        g_value = primal_function(primal_var)
        F_list.append(F_value.item())
        g_list.append(g_value.item())
        cr_1 = torch.abs(F_value + g_value)
        cr_2 = abs((M_matrix * (round_function(primal_var) - primal_var)).sum())
        cr_1_list.append(cr_1.item())
        cr_2_list.append(cr_2.item())
        #         cr_2 = torch.norm(A_matrix @ primal_var - b)
        if k == 0:
            init_cr_1 = cr_1
            init_cr_2 = cr_2
            init_phi = F_value.item()
            init_f = g_value.item()
        clear_output(wait=True)
        print('\n'.join([
            f'Step #{k}',
            f'cr_1: {init_cr_1} -> {cr_1}',
            f'cr_2: {init_cr_2} -> {cr_2}',
            f'phi: {init_phi} -> {F_value.item()}',
            f'f: {init_f} -> {g_value.item()}',
        ]))

        if cr_1 <= eps and cr_2 <= eps_tilde:
            break

        alpha_next = (k + 2) * one_over_L / 2
        k += 1

        if max_steps is not None:
            if k == max_steps:
                break

    return primal_var, dual_var_next, F_list, g_list, cr_1_list, cr_2_list