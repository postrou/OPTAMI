import torch
from torch.optim.optimizer import Optimizer

import OPTAMI


class PrimalDualAccelerated(Optimizer):

    def __init__(
            self,
            params,
            M_p: float,
            p_order: int = 3,
            tensor_step_method: Optimizer = None,
            tensor_step_kwargs: dict = None,
            subsolver: Optimizer = None,
            subsolver_args: dict = None,
            max_iters: int = None,
            verbose: bool = None,
            calculate_primal_var=None,
            keep_psi_data=False
    ):
        self.p_order = p_order
        self.tensor_step_method = tensor_step_method
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.max_iters = max_iters
        self.tensor_step_kwargs = tensor_step_kwargs
        self.verbose = verbose
        self.keep_psi_data = keep_psi_data
        self._calculate_primal_var = calculate_primal_var

        if not M_p >= 0.0:
            raise ValueError("Invalid L: {}".format(M_p))
        if calculate_primal_var is None:
            raise ValueError('We need function for primal (x) value calculation from lambda (dual) variable')

        M = M_p * 2
        M_squared, M_p_squared = M ** 2, M_p ** 2

        C = p_order / 2 * ((p_order + 1) / (p_order - 1) * (M_squared - M_p_squared)) ** 0.5
        A_factor = ((p_order - 1) * (M_squared - M_p_squared) /
                    (4 * (p_order + 1) * p_order ** 2 * M_squared)) ** (p_order / 2)

        p_fact = 1
        for i in range(2, p_order + 1):
            p_fact *= i
        step_3_fst_factor = (p_fact / C) ** (1 / p_order)

        defaults = dict(
            M_p=M_p,
            M=M,
            C=C,
            A_factor=A_factor,
            step_3_fst_factor=step_3_fst_factor,
        )
        super().__init__(params, defaults)

        self._init_state()

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
        state['psi'] = None
        state['A'] = 0.0
        state['k'] = 0
        state['v'] = [torch.zeros_like(param) for param in params]
        if self.keep_psi_data:
            state['A_arr'] = [state['A']]
            state['phi_arr'] = [None]
            state['grad_phi_arr'] = [None]
            state['param_arr'] = [params[0].detach().clone()]
            state['psi_value'] = None

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure is None. Closure is necessary for this method.")

        # initialisation
        group = self.param_groups[0]
        params = group['params']
        state = self.state['default']
        A = state['A']
        k = state['k']

        # step 3
        self._estim_seq_subproblem(k, group)

        # step 4 (we won't need a_i)
        A_next = self._calculate_A(k + 1, group)
        state['A'] = A_next

        # step 5
        A_over_A_next = A / A_next
        self._update_param_point(state['v'], A_over_A_next, params)

        # step 6
        self._perform_tensor_step(closure)

        # step 7 (since we'll need this function only on step 3 on next k,
        #   here we only calculate \phi(\lambda_{k + 1}) and \nabla \phi(\lambda_{k + 1})
        phi_next, grad_phi_next = self._calculate_closure_and_its_grad(closure, A, A_next, params)

        # step 8
        self._calculate_x_hat_next(k, A_over_A_next, params)

        if self.keep_psi_data:
            state['phi_arr'].append(phi_next)
            state['grad_phi_arr'].append(grad_phi_next)
            state['A_arr'].append(A_next)
            state['param_arr'].append(params[0].detach().clone())
            self._fill_psi_information()
            
        state['k'] += 1

    def _calculate_A(self, k, param_group):
        A_factor = param_group['A_factor']
        return A_factor * (k / (self.p_order + 1)) ** (self.p_order + 1)

    def _estim_seq_subproblem(self, k, param_group):
        params = param_group['params']
        fst_factor = param_group['step_3_fst_factor']

        state = self.state['default']
        grad_phi_sum = state['grad_phi_sum']

        v = state['v']
        if k != 0:
            for i, param in enumerate(params):
                grad_sum = grad_phi_sum[i]
                grad_sum_norm = grad_sum.norm()
                snd_factor = grad_sum / (grad_sum_norm ** (1 - 1 / self.p_order))
                v[i] = -fst_factor * snd_factor

        return v

    def _update_param_point(self, v, A_over_A_next, params):
        with torch.no_grad():
            for i, param in enumerate(params):
                param.mul_(A_over_A_next).add_(v[i], alpha=1 - A_over_A_next)

    def _perform_tensor_step(self, closure):
        group = self.param_groups[0]
        params = group['params']
        M = group['M']
        if self.tensor_step_method is None:
            if self.p_order == 3:
                self.tensor_step_method = OPTAMI.BasicTensorMethod(
                    params=params,
                    L=M,
                    subsolver=self.subsolver,
                    subsolver_args=self.subsolver_args,
                    max_iters=self.max_iters,
                    verbose=self.verbose
                )
            elif self.p_order == 2:
                self.tensor_step_method = OPTAMI.CubicReguralizedNewton(
                    params=params,
                    L=M,
                    subsolver=self.subsolver,
                    subsolver_args=self.subsolver_args,
                    max_iters=self.max_iters,
                    verbose=self.verbose
                )
            else:
                raise NotImplementedError(f'Method for p = {self.p_order} \
                                          is not implemented!')
        self.tensor_step_method.step(closure)

    def _calculate_closure_and_its_grad(self, closure, A, A_next, params):
        state = self.state['default']
        outputs = closure()
        phi_next = outputs.detach().clone()
        if type(outputs) == list:
            state['phi_next'] = phi_next
        else:
            state['phi_next'] = [phi_next]

        # add new gradient to the sum
        outputs.backward()
        for i, param in enumerate(params):
            grad_phi_next = param.grad.clone()
            state['grad_phi_sum'][i] += (A_next - A) * grad_phi_next
        self.zero_grad()

        return phi_next, grad_phi_next

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

    def _fill_psi_information(self):
        group = self.param_groups[0]
        params = group['params']
        state = self.state['default']

        assert torch.equal(params[0], group['params'][0])
        psi_value = self.psi(
            params[0],
            state['A_arr'],
            state['phi_arr'],
            state['grad_phi_arr'],
            state['param_arr'],
            group
        )
        grad_psi = torch.autograd.grad(psi_value, params)[0]
        state['psi_value'] = psi_value
        state['grad_psi_norm'] = grad_psi.norm()

    def psi(self, lamb, A_arr, phi_arr, grad_phi_arr, param_arr, param_group):
        assert lamb.requires_grad
        assert lamb.norm().requires_grad, 'There is torch.no_grad() somewhere!'
        C = param_group['C']

        state = self.state['default']
        k = state['k'] - 1  # since we will use _psi after first step

        fact = 1
        for i in range(2, self.p_order + 2):
            fact *= i

        psi_0 = C / fact * lamb.norm() ** (self.p_order + 1)
        result = psi_0

        if k != 0:
            for i in range(1, k + 1):
                A = A_arr[i]
                A_prev = A_arr[i - 1]
                phi = phi_arr[i].detach().clone()
                grad_phi = grad_phi_arr[i].detach().clone()
                param = param_arr[i].detach().clone()
                result += (A - A_prev) * (phi + grad_phi @ (lamb - param))

        return result
