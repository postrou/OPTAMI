import numpy as np
import torch
from torch.optim.optimizer import Optimizer

import OPTAMI as opt
from OPTAMI.sup import tuple_to_vec as ttv


class PrimalDualAccelerated(Optimizer):

    def __init__(
            self,
            params,
            M_p=1e+3,
            eps=1e-1,
            p_order=3,
            subsolver=opt.BDGM,
            subsolver_bdgm=None,
            tol_subsolve=None,
            subsolver_args=None,
            calculate_primal_var=None,
            calculate_grad_phi=None,
            debug=False
    ):
        if not M_p >= 0.0:
            raise ValueError("Invalid L: {}".format(M_p))
        if calculate_primal_var is None:
            raise ValueError('We need function for primal (x) value calculation from lambda (dual) variable')

        M = M_p * 2
        M_squared, M_p_squared = M ** 2, M_p ** 2

        C = p_order / 2 * ((p_order + 1) / (p_order - 1) * (M_squared - M_p_squared)) ** 0.5
        A_factor = ((p_order - 1) * (M_squared - M_p_squared) /
                    (4 * (p_order + 1) * p_order ** 2 * M_squared)) ** (p_order / 2)

        p_fact = ttv.factorial(p_order)
        step_3_fst_factor = (p_fact / C) ** (1 / p_order)

        defaults = dict(
            M_p=M_p,
            M=M,
            C=C,
            A_factor=A_factor,
            step_3_fst_factor=step_3_fst_factor,
            eps=eps,
            p_order=p_order,
            subsolver=subsolver,
            subsolver_bdgm=subsolver_bdgm,
            tol_subsolve=tol_subsolve,
            subsolver_args=subsolver_args
        )
        super().__init__(params, defaults)

        self.debug = debug
        self._init_state()

        self._calculate_primal_var = calculate_primal_var
        self._calculate_grad_phi = calculate_grad_phi

    def _init_state(self):
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']

        # filling state
        state = self.state['default']
        for param in params:
            param_copy = param.clone().detach()
            assert len(param_copy.shape) <= 2, "May be some troubles with tensor of higher order"
        state['x_hat'] = []
        state['grad_phi_sum'] = [torch.zeros_like(param) for param in params]
        state['phi_next'] = None
        state['ksi'] = None
        state['A'] = 0.0
        state['k'] = 0

        if self.debug:
            state['A_arr'] = [0.0]
            state['phi_arr'] = [None]
            state['grad_phi_arr'] = [None]
            state['param_arr'] = [None]
            state['grad_psi_norm'] = None
            state['other_grad_psi_norm'] = torch.tensor(0)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure is None. Closure is necessary for this method. ")
        assert len(self.param_groups) == 1

        # initialisation
        group = self.param_groups[0]
        params = group['params']
        M = group['M']
        state = self.state['default']
        A = state['A']
        k = state['k']

        # step 3
        #   (checked)
        v = self._estim_seq_subproblem(k, group)

        # step 4 (we won't need a)
        #   (checked)
        A_next = self._calculate_A(k + 1, group)
        state['A'] = A_next
        if self.debug:
            state['A_arr'].append(A_next)

        # step 5
        #   (checked)
        A_over_A_next = A / A_next
        self._update_param_point(v, A_over_A_next, params)

        # step 6
        subsolver = group['subsolver']
        subsolver_bdgm = group['subsolver_bdgm']
        tol_subsolve = group['tol_subsolve']
        subsolver_args = group['subsolver_args']
        optimizer = subsolver(
            params,
            L=M,
            subsolver_bdgm=subsolver_bdgm,
            tol_subsolve=tol_subsolve,
            subsolver_args=subsolver_args
        )
        optimizer.step(closure)
        if self.debug:
            state['param_arr'].append(params[0].detach().clone())

        # step 7 (since we'll need this function only on step 3 on next k,
        #   here we only calculate \phi(\lambda_{k + 1}) and \nabla \phi(\lambda_{k + 1})
        #   (checked)
        self._calculate_closure_and_its_grad(closure, A, A_next, params)

        # step 8
        #   (checked)
        self._calculate_x_hat_next(k, A_over_A_next, params)

        state['k'] += 1

    def _calculate_A(self, k, param_group):
        A_factor = param_group['A_factor']
        p_order = param_group['p_order']
        return A_factor * (k / (p_order + 1)) ** (p_order + 1)

    def _estim_seq_subproblem(self, k, param_group):
        params = param_group['params']
        p_order = param_group['p_order']
        fst_factor = param_group['step_3_fst_factor']

        state = self.state['default']
        grad_phi_sum = state['grad_phi_sum']

        v = [torch.zeros_like(param) for param in params]
        if k != 0:
            for i, param in enumerate(params):
                grad_sum = grad_phi_sum[i]
                grad_sum_norm = grad_sum.norm()
                snd_factor = grad_sum / grad_sum_norm ** (1 - 1 / p_order)
                v[i] = -fst_factor * snd_factor

        if self.debug:
            # test that grad is zero
            v_val = v[0].detach().clone().requires_grad_(True)
            A_arr = state['A_arr']
            phi_arr = state['A_arr']
            grad_phi_arr = state['grad_phi_arr']
            param_arr = state['param_arr']
            ksi = self._psi(v_val, A_arr, phi_arr, grad_phi_arr, param_arr, param_group)
            grad_psi = torch.autograd.grad(ksi, v_val)[0]
            # # assert torch.allclose(grad_psi, torch.zeros_like(grad_psi), atol=1e-7)
            state['grad_psi_norm'] = grad_psi.norm()

            if k != 0:
                C = param_group['C']
                grad_sum = torch.zeros_like(params[0])
                for i in range(1, k + 1):
                    A = A_arr[i]
                    A_prev = A_arr[i - 1]
                    lamb = param_arr[i]
                    grad_phi = self._calculate_grad_phi(lamb)
                    # assert torch.allclose(grad_phi, grad_phi_arr[i])
                    grad_sum += (A - A_prev) * grad_phi

                other_fst_factor = -(ttv.factorial(p_order) / C) ** (1 / p_order)
                other_snd_factor = grad_sum / grad_sum.norm() ** ((p_order - 1) / p_order)

                other_v_val = other_fst_factor * other_snd_factor
                other_v_val.requires_grad_(True)
                assert torch.allclose(v_val, other_v_val)

                other_psi = self._psi(other_v_val, A_arr, phi_arr, grad_phi_arr, param_arr, param_group)
                other_grad_psi = torch.autograd.grad(other_psi, other_v_val)[0]
                state['other_grad_psi_norm'] = other_grad_psi.norm()

        return v

    def _update_param_point(self, v, A_over_A_next, params):
        with torch.no_grad():
            for i, param in enumerate(params):
                param.mul_(A_over_A_next).add_(v[i], alpha=1 - A_over_A_next)

    def _calculate_closure_and_its_grad(self, closure, A, A_next, params):
        state = self.state['default']
        outputs = closure()
        phi_next = outputs.detach().clone()
        if type(outputs) == list:
            state['phi_next'] = phi_next
        else:
            state['phi_next'] = [phi_next]

        if self.debug:
            state['phi_arr'].append(phi_next)

        # add new gradient to the sum
        grad_phi_next = torch.autograd.grad(outputs=outputs, inputs=params, retain_graph=False)
        # grad_phi_next = [self._calculate_grad_phi(params[0])]
        for i, grad in enumerate(grad_phi_next):
            state['grad_phi_sum'][i] += (A_next - A) * grad

        if self.debug:
            state['grad_phi_arr'].append(grad_phi_next[0])

    def _calculate_x_hat_next(self, k, A_over_A_next, params):
        with torch.no_grad():
            for i, param in enumerate(params):
                X_matrix, _, _ = self._calculate_primal_var(param)
                state = self.state['default']
                if k == 0:
                    X_hat_matrix_next = X_matrix  # A = 0
                    state['x_hat'].append(X_hat_matrix_next)
                else:
                    X_hat_matrix = state['x_hat'][i]
                    X_hat_matrix_next = (1 - A_over_A_next) * X_matrix + A_over_A_next * X_hat_matrix
                    state['x_hat'][i] = X_hat_matrix_next

    def _psi(self, lamb, A_arr, phi_arr, grad_phi_arr, param_arr, param_group):
        assert lamb.requires_grad
        p_order = param_group['p_order']
        C = param_group['C']

        state = self.state['default']
        k = state['k']

        fact = 1
        for i in range(2, p_order + 2):
            fact *= i

        ksi_0 = C / fact * lamb.norm() ** (p_order + 1)
        result = ksi_0

        if k != 0:
            for i in range(1, k + 1):
                A = A_arr[i]
                A_prev = A_arr[i - 1]
                phi = phi_arr[i].detach().clone()
                grad_phi = grad_phi_arr[i].detach().clone()
                param = param_arr[i].detach().clone()
                result += (A - A_prev) * (phi + grad_phi @ (lamb - param))

        return result
