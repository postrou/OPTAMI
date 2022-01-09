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
            calculate_primal_var=None
    ):
        if not M_p >= 0.0:
            raise ValueError("Invalid L: {}".format(M_p))
        if calculate_primal_var is None:
            raise ValueError('We need function for primal (x) value calculation from lambda (dual) variable')

        M = M_p * (p_order + 2)
        M_squared, M_p_squared = M ** 2, M_p ** 2

        C = p_order / 2 * ((p_order + 1) / (p_order - 1) * (M_squared - M_p_squared)) ** 0.5
        A_factor = ((p_order - 1) * (M_squared - M_p_squared) /
                    (4 * (p_order + 1) * p_order ** 2 * M_squared)) ** (p_order / 2)

        p_fact = ttv.factorial(p_order)
        step_3_fst_factor = (C / p_fact) ** (1 / p_order)

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

        self._init_state()

        self._calculate_primal_var = calculate_primal_var

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
        state['grad_phi_k_sum'] = [torch.zeros_like(param) for param in params]
        state['phi_next'] = None
        state['A'] = 0.0
        state['k'] = 0

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure is None. Closure is necessary for this method. ")
        assert len(self.param_groups) == 1

        # initialisation
        # print('Step 1: Initialization...')
        group = self.param_groups[0]
        params = group['params']
        M = group['M']
        state = self.state['default']
        A = state['A']
        k = state['k']

        # step 3
        # print('Step 3: Computation of v_k...')
        v = self._estim_seq_subproblem(k, group)

        # step 4 (we won't need a)
        # print('Step 4: Computation of A_{k + 1}...')
        A_next = self._calculate_A(k + 1, group)
        state['A'] = A_next

        # step 5
        # print('Step 5: Computation of y_k...')
        A_over_A_next = A / A_next
        self._update_param_point(v, A_over_A_next, params)

        # step 6
        # print('Step 6: Computation of \\lambda_{k + 1} (tensor step)...')
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

        # step 7 (since we'll need this function only on step 3 on next k,
        #   here we only calculate \phi(\lambda_{k + 1}) and \nabla \phi(\lambda_{k + 1})
        # print('Step 7: Computation of \\nabla \\phi(\\lambda_{k + 1}...')
        self._calculate_closure_and_its_grad(closure, A, A_next, params)

        # step 8
        # print('Step 8: Computation of \\hat x_{k + 1}...')
        self._calculate_x_hat_next(k, A_over_A_next, params)

    def _calculate_A(self, k, param_group):
        A_factor = param_group['A_factor']
        p_order = param_group['p_order']
        return A_factor * (k / (p_order + 1)) ** (p_order + 1)

    def _estim_seq_subproblem(self, k, param_group):
        params = param_group['params']
        p_order = param_group['p_order']
        fst_factor = param_group['step_3_fst_factor']

        state = self.state['default']
        grad_phi_k_sum = state['grad_phi_k_sum']

        if k == 0:
            return [torch.zeros_like(param) for param in params]

        results = [torch.empty_like(param) for param in params]
        for i, param in enumerate(params):
            grad_sum = grad_phi_k_sum[i]
            grad_sum_norm = grad_sum.norm()
            print(grad_sum_norm)
            snd_factor = grad_sum / grad_sum_norm ** (1 - 1 / p_order)
            results[i] = -fst_factor * snd_factor
        return results

    def _update_param_point(self, v, A_over_A_next, params):
        with torch.no_grad():
            for i, param in enumerate(params):
                param.mul_(A_over_A_next).add_(v[i], alpha=1 - A_over_A_next)

    def _calculate_closure_and_its_grad(self, closure, A, A_next, params):
        state = self.state['default']
        outputs = closure()
        grad_phi_next = torch.autograd.grad(outputs=outputs, inputs=params, retain_graph=False)
        if type(outputs) == list:
            state['phi_next'] = outputs
        else:
            state['phi_next'] = [outputs]

        # add new gradient to the sum
        for i, grad in enumerate(grad_phi_next):
            state['grad_phi_k_sum'][i] += (A_next - A) * grad

    def _calculate_x_hat_next(self, k, A_over_A_next, params):
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
