import unittest
from copy import deepcopy

import torch.autograd.functional as AF
from tqdm import trange

from OPTAMI.sup import tuple_to_vec as ttv
from OPTAMI import BDGM
from run_pd_experiment import *


class PrimalDualAcceleratedTester(PrimalDualAccelerated):
    def __init__(self,
            params,
            M_p=1e+3,
            eps=1e-1,
            p_order=3,
            subsolver=BDGM,
            subsolver_bdgm=None,
            tol_subsolve=None,
            subsolver_args=None,
            calculate_primal_var=None,
            calculate_grad_phi=None):
        super().__init__(
            params,
            M_p,
            eps,
            p_order,
            subsolver,
            subsolver_bdgm,
            tol_subsolve,
            subsolver_args,
            calculate_primal_var)
        self._calculate_grad_phi = calculate_grad_phi

        state = self.state['default']

        state['A_arr'] = [state['A']]
        state['phi_arr'] = [None]
        state['grad_phi_arr'] = [None]
        state['param_arr'] = [None]
        state['grad_psi_norm'] = None
        state['other_grad_psi_norm'] = torch.tensor(0)
        state['v'] = None

    def step(self, closure=None):
        super().step(closure)

        params = self.param_groups[0]['params']
        state = self.state['default']
        state['param_arr'].append(params[0].detach().clone())
        state['A_arr'].append(state['A'])

    def _estim_seq_subproblem(self, k, param_group):
        v = super()._estim_seq_subproblem(k, param_group)

        self.state['default']['v'] = v[0]
        return v

    def _calculate_closure_and_its_grad(self, closure, A, A_next, params):
        phi_next, grad_phi_next = super()._calculate_closure_and_its_grad(closure, A, A_next, params)

        state = self.state['default']
        state['grad_phi_arr'].append(grad_phi_next[0])
        state['phi_arr'].append(phi_next)

        return phi_next, grad_phi_next

    def psi(self, lamb, A_arr, phi_arr, grad_phi_arr, param_arr, param_group):
        assert lamb.requires_grad
        assert lamb.norm().requires_grad, 'There is torch.no_grad() somewhere!'
        p_order = param_group['p_order']
        C = param_group['C']

        state = self.state['default']
        k = state['k'] - 1  # since we will use _psi after first step

        fact = 1
        for i in range(2, p_order + 2):
            fact *= i

        psi_0 = C / fact * lamb.norm() ** (p_order + 1)
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


class PrimalDualTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_steps = 500

        self.device = 'cpu'
        self.gammas = [0.001, 0.5, 1, 1.5]
        eps = 0.001
        image_index = 2
        new_m = 10

        images, labels = load_data()
        if new_m is not None:
            n = new_m ** 2
            m = new_m
        else:
            n = len(images[0])
            m = int(np.sqrt(n))

        p_list = [34860, 31226, 239, 37372, 17390]
        q_list = [45815, 35817, 43981, 54698, 49947]

        # x_array = np.linspace(1 / 2e-2, 1 / 4e-4, 6)
        # epslist = 1 / x_array
        epsp = eps / 8

        p, q = mnist(epsp, p_list[image_index], q_list[image_index], images, m)
        p = torch.tensor(p, device=self.device, dtype=torch.double)
        q = torch.tensor(q, device=self.device, dtype=torch.double)

        self.optimizers = []
        self.closures = []
        self.primal_var_funcs = []
        self.grad_phi_funcs = []
        self.phi_funcs = []
        for gamma in self.gammas:
            M_matrix = calculate_M_matrix(m).to(self.device)
            M_matrix_over_gamma = M_matrix / gamma
            M_p = calculate_lipschitz_constant(gamma, p_order=3)

            ones = torch.ones(n, device=self.device, dtype=torch.double)
            lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=False, device=self.device)
            lamb.mul_(-1 / gamma).requires_grad_(True)

            self.primal_var_funcs.append(
                lambda dual_var, gamma=gamma, M_matrix_over_gamma=M_matrix_over_gamma, ones=ones:
                calculate_x(dual_var, n, gamma, M_matrix_over_gamma, ones)
            )

            self.grad_phi_funcs.append(
                lambda dual_var, gamma=gamma, primal_var_func=self.primal_var_funcs[-1], ones=ones:
                grad_phi(dual_var, gamma, primal_var_func, p, q, ones, self.device)
            )

            self.phi_funcs.append(
                lambda dual_var, gamma=gamma, M_matrix_over_gamma=M_matrix_over_gamma, ones=ones:
                phi(dual_var, n, gamma, M_matrix_over_gamma, ones, p, q)
            )

            optimizer = PrimalDualAcceleratedTester(
                [lamb],
                M_p=M_p,
                p_order=torch.tensor(3, device=self.device),
                eps=0.01,
                calculate_primal_var=self.primal_var_funcs[-1],
                calculate_grad_phi=self.grad_phi_funcs[-1]
            )

            closure = \
                lambda dual_var=lamb, gamma=gamma, M_matrix_over_gamma=M_matrix_over_gamma, ones=ones, optimizer=optimizer: \
                    phi(
                        dual_var,
                        n,
                        gamma,
                        M_matrix_over_gamma,
                        ones,
                        p,
                        q,
                        optimizer=optimizer
                    )

            self.optimizers.append(optimizer)
            self.closures.append(closure)

    def test_subproblem(self):
        for gamma, optimizer, closure in zip(self.gammas, self.optimizers, self.closures):
            for i in trange(1, self.n_steps + 1, desc=f'test_subproblem, gamma = {gamma}'):
                optimizer.step(closure)

                if self.device != 'cpu':
                    torch.cuda.empty_cache()

                param_group = optimizer.param_groups[0]
                params = param_group['params']
                p_order = param_group['p_order']
                C = param_group['C']

                state = optimizer.state['default']
                A_arr = state['A_arr']
                phi_arr = state['A_arr']
                grad_phi_arr = state['grad_phi_arr']
                param_arr = state['param_arr']
                grad_sum = state['grad_phi_sum'][0]

                v = state['v'].detach().clone().requires_grad_(True)
                psi = optimizer.psi(v, A_arr, phi_arr, grad_phi_arr, param_arr, param_group)
                grad_psi = torch.autograd.grad(psi, v)[0]

                other_v = torch.zeros_like(v).requires_grad_(True)
                other_grad_sum = torch.zeros_like(params[0])
                other_grads = torch.zeros(i, len(grad_psi), dtype=torch.double)
                for j in range(1, i + 1):
                    A = A_arr[j]
                    A_prev = A_arr[j - 1]
                    lamb = param_arr[j]
                    grad_phi = grad_phi_arr[j]
                    other_grad_phi = optimizer._calculate_grad_phi(lamb)
                    self.assertTrue(torch.allclose(grad_phi, other_grad_phi),
                                    f'Autograd gradient and explicit gradient are not equal on step {j}! '
                                    f'Maximal element is {torch.abs(grad_phi - other_grad_phi).argmax()},'
                                    f'{torch.abs(grad_phi - other_grad_phi).max()}')

                    other_grads[j - 1] = (A - A_prev) * other_grad_phi
                    other_grad_sum += (A - A_prev) * other_grad_phi
                    if j == i - 1:
                        # since v calculation doesn't include grad_phi_next
                        other_fst_factor = (ttv.factorial(p_order) / C) ** (1 / p_order)
                        # other_snd_factor = other_grad_sum / (other_grad_sum.norm() ** ((p_order - 1) / p_order))
                        other_snd_factor = other_grad_sum / (other_grad_sum.norm() ** ((p_order - 1) / p_order))

                        other_v = -other_fst_factor * other_snd_factor
                        other_v.requires_grad_(True)

                other_psi = optimizer.psi(other_v, A_arr, phi_arr, grad_phi_arr, param_arr, param_group)
                other_grad_psi = torch.autograd.grad(other_psi, other_v)[0]

                stable_grad_sum = other_grads.sum(dim=0)
                self.assertTrue(torch.allclose(other_grad_sum, stable_grad_sum), '\n'.join([
                    'Stable grad sum and method grad sum are not close!',
                    f'other_grad_sum.norm() = {other_grad_sum.norm()}',
                    f'stable_grad_sum.norm() = {stable_grad_sum.norm()}'
                ]))

                self.assertTrue(torch.allclose(other_grad_sum, grad_sum), '\n'.join([
                    'Grad sums are not close!',
                    f'grad_sum.norm() = {grad_sum.norm()}',
                    f'other_grad_sum.norm() = {other_grad_sum.norm()}'
                ]))

                self.assertTrue(torch.allclose(v, other_v), '\n'.join([
                    'v values are not close!',
                    f'v.norm() = {v.norm()}',
                    f'other_v.norm() = {other_v.norm()}',
                ]))

                self.assertTrue(torch.allclose(grad_psi, other_grad_psi), '\n'.join([
                    f'Grad psi values are not close!',
                    f'grad_psi.norm() = {grad_psi.norm()}',
                    f'other_grad_psi.norm() = {other_grad_psi.norm()}',
                    f'v.norm() = {v.norm()}',
                    f'other_v.norm() = {other_v.norm()}',
                    f'grad_sum.norm() = {grad_sum.norm()}',
                    f'other_grad_sum.norm() = {other_grad_sum.norm()}',
                    f'stable_grad_sum.norm() = {stable_grad_sum.norm()}'
                ]))

    def test_hessian(self):
        for gamma, optimizer, closure, phi_func in zip(self.gammas, self.optimizers, self.closures, self.phi_funcs):
            for _ in trange(self.n_steps, desc=f'test_hessian, gamma = {gamma}'):
                optimizer.step(closure)
                if self.device != 'cpu':
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    lamb = optimizer.param_groups[0]['params'][0]
                    phi_hess = AF.hessian(phi_func, lamb)
                    phi_hess_norm = torch.norm(phi_hess, 'fro')
                    self.assertNotAlmostEqual(phi_hess_norm.item(), 0.0, msg='Phi hessian norm equals zero!')
                    self.assertFalse(torch.isnan(phi_hess_norm), msg='Phi hessian norm is nan!')


if __name__ == '__main__':
    unittest.main()
