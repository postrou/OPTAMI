from copy import deepcopy

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
            subsolver_args=None
    ):
        if not M_p >= 0.0:
            raise ValueError("Invalid L: {}".format(M_p))

        M = M_p * (p_order + 2)
        M_squared, M_p_squared = M ** 2, M_p ** 2

        C = p_order / 2 * np.sqrt((p_order + 1) / (p_order - 1) * (M_squared - M_p_squared))
        A_factor = ((p_order - 1) * (M_squared - M_p_squared) /
                    (4 * (p_order + 1) * p_order ** 2 * M_squared)) ** (p_order / 2)

        defaults = dict(
            M_p=M_p,
            M=M,
            C=C,
            A_factor=A_factor,
            eps=eps,
            p_order=p_order,
            subsolver=subsolver,
            subsolver_bdgm=subsolver_bdgm,
            tol_subsolve=tol_subsolve,
            subsolver_args=subsolver_args
        )
        super().__init__(params, defaults)

        self._init_state()

    def _init_state(self):
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        params = group['params']

        # filling state
        state = self.state['default']
        grad_phi_k = []
        for param in params:
            param_copy = param.clone().detach()
            assert all(param_copy == 0), 'Initial point should be all zeros!'
            grad_phi_k.append(param_copy) # since we won't need \grad \phi(\lambda_0)
        state['y'] = []
        state['x_hat'] = None
        state['grad_phi_k'] = [tuple(grad_phi_k)]

        A = self._calculate_A(0, group)
        state['A_k'] = [A]

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure is None. Closure is necessary for this method. ")
        assert len(self.param_groups) == 1

        # initialisation
        group = self.param_groups[0]
        params = group['params']
        M_p = group['M_p']
        state = self.state['default']
        A_k = state['A_k']
        A = A_k[-1]
        k = len(A_k) - 1    # because we keep A_next too

        # step 3
        v = self._estim_seq_subproblem(k, group)

        # step 4 (we won't need a)
        A_next = self._calculate_A(k + 1, group)

        # step 5
        A_over_A_next = A / A_next
        self._update_param_point(v, A_over_A_next, params)

        # step 6
        subsolver = group['subsolver']
        subsolver_bdgm = group['subsolver_bdgm']
        tol_subsolve = group['tol_subsolve']
        subsolver_args = group['subsolver_args']
        optimizer = subsolver(
            params,
            L=M_p,
            subsolver_bdgm=subsolver_bdgm,
            tol_subsolve=tol_subsolve,
            subsolver_args=subsolver_args
        )
        optimizer.step(closure)

        # step 7 (since here we'll need this function only on step 3 on next k,
        #   here we only calculate \nabla \phi(\lambda_{k + 1})
        self._calculate_closure_grad(closure, params)

        # step 8
        self._calculate_x_hat_next(A_over_A_next)

    def _calculate_A(self, k, param_group):
        A_factor = param_group['A_factor']
        p_order = param_group['p_order']
        return A_factor * (k / (p_order + 1)) ** (p_order + 1)

    def _estim_seq_subproblem(self, k, param_group):
        p_order = param_group['p_order']
        C = param_group['C']

        state = self.state['default']
        A_k = state['A_k']
        grad_phi_k = state['grad_phi_k']

        if k == 0:
            return [torch.zeros_like(param) for param in param_group['params']]

        p_fact = ttv.factorial(p_order)
        A_prev = A_k[0]
        one_over_p = 1 / p_order
        fst_factor = (p_fact / C) ** one_over_p
        results = [0.0, 0.0]
        for i, param in enumerate(param_group):
            grad_sum = torch.zeros_like(param)
            for j in range(1, k):
                A = A_k[j]
                grad_phi = torch.tensor(grad_phi_k[j])
                grad_sum += (A - A_prev) * grad_phi

            snd_factor = grad_sum / (grad_sum.norm() ** (1 - one_over_p))
            results[i] = -fst_factor * snd_factor
        return results

    def _update_param_point(self, v, A_over_A_next, params):
        with torch.no_grad():
            for i, param in enumerate(params):
                param.mul_(A_over_A_next).add_(v[i], alpha=1 - A_over_A_next)

    def _calculate_closure_grad(self, closure, params):
        state = self.state['default']
        outputs = closure()
        grad_phi_next = torch.autograd.grad(outputs=outputs, inputs=params, retain_graph=False)
        state['grad_phi_k'].append(grad_phi_next)

    def _calculate_x_hat_next(self, A_over_A_next):
        x = self._calculate_x()
        state = self.state['default']
        x_hat = state['x_hat']
        if x_hat is None:  # it means k == 0
            x_hat_next = x  # a == A_next
        else:
            x_hat_next = (1 - A_over_A_next) * x + A_over_A_next * x_hat
        state['x_hat'] = x_hat_next
        # return x_hat_next

    def _calculate_x(self):
        raise NotImplementedError()
