import unittest

import torch.autograd.functional as AF
from tqdm import trange

from OPTAMI.sup import tuple_to_vec as ttv
from run_pd_experiment import *


class PrimalDualAcceleratedTester(PrimalDualAccelerated):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state = self.state['default']
        param_group = self.param_groups[0]
        params = param_group['params']

        state['A_arr'] = [0.0]
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

        self.state['default']['v'] = v[0].detach().clone().requires_grad_(True)
        return v

    def _calculate_closure_and_its_grad(self, closure, A, A_next, params):
        phi_next, grad_phi_next = super()._calculate_closure_and_its_grad(closure, A, A_next, params)

        state = self.state['default']
        state['grad_phi_arr'].append(grad_phi_next[0])
        state['phi_arr'].append(phi_next)

        return phi_next, grad_phi_next

    def _psi(self, lamb, A_arr, phi_arr, grad_phi_arr, param_arr, param_group):
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
        self.n_steps = 100

        self.device = 'cpu'
        gamma = 1
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

        M_matrix = calculate_M_matrix(m).to(self.device)
        M_matrix_over_gamma = M_matrix / gamma
        M_p = calculate_lipschitz_constant(gamma, p_order=3)

        ones = torch.ones(n, device=self.device, dtype=torch.double)
        lamb = torch.zeros(n * 2, dtype=torch.double, requires_grad=True, device=self.device)
        primal_var_func = lambda lamb: calculate_x(lamb, n, gamma, M_matrix_over_gamma, ones)
        grad_phi_func = lambda lamb: grad_phi(lamb, gamma, primal_var_func, p, q, ones, self.device)
        self.phi_func = lambda lamb: phi(lamb, n, gamma, M_matrix_over_gamma, ones, p, q)

        self.optimizer = PrimalDualAcceleratedTester(
            [lamb],
            M_p=M_p,
            p_order=torch.tensor(3, device=self.device),
            eps=0.01,
            calculate_primal_var=primal_var_func,
            calculate_grad_phi=grad_phi_func
        )

        self.closure = lambda: phi(
            lamb,
            n,
            gamma,
            M_matrix_over_gamma,
            ones,
            p,
            q,
            optimizer=self.optimizer
        )

    def test_hessian(self):
        for i in trange(self.n_steps, desc='test_hessian'):
            self.optimizer.step(self.closure)
            if self.device != 'cpu':
                torch.cuda.empty_cache()

            with torch.no_grad():
                lamb = self.optimizer.param_groups[0]['params'][0]
                phi_hess = AF.hessian(self.phi_func, lamb)
                phi_hess_norm = torch.norm(phi_hess, 'fro')
                self.assertNotAlmostEqual(phi_hess_norm.item(), 0.0, msg='Phi hessian norm equals zero!')
                self.assertFalse(torch.isnan(phi_hess_norm), msg='Phi hessian norm is nan!')

    def test_subproblem(self):
        for i in trange(self.n_steps, desc='test_subproblem'):
            self.optimizer.step(self.closure)

            if self.device != 'cpu':
                torch.cuda.empty_cache()

            param_group = self.optimizer.param_groups[0]
            params = param_group['params']
            p_order = param_group['p_order']
            fst_factor = param_group['step_3_fst_factor']

            state = self.optimizer.state['default']
            A_arr = state['A_arr']
            phi_arr = state['A_arr']
            grad_phi_arr = state['grad_phi_arr']
            param_arr = state['param_arr']

            v = state['v']
            grad_sum = state['grad_phi_sum'][0]
            psi = self.optimizer._psi(v, A_arr, phi_arr, grad_phi_arr, param_arr, param_group)
            grad_psi = torch.autograd.grad(psi, v)[0]
            # state['grad_psi_norm'] = grad_psi.norm()

            if i > 0:
                C = param_group['C']
                other_grad_sum = torch.zeros_like(params[0])
                other_grads = torch.zeros(i, len(grad_psi), dtype=torch.double)
                for j in range(1, i + 1):
                    A = A_arr[j]
                    A_prev = A_arr[j - 1]
                    lamb = param_arr[j]
                    other_grad_phi = self.optimizer._calculate_grad_phi(lamb)
                    # assert torch.allclose(grad_phi, grad_phi_arr[j])
                    other_grads[j - 1] = (A - A_prev) * other_grad_phi
                    other_grad_sum += (A - A_prev) * other_grad_phi

                self.assertTrue(torch.allclose(other_grad_sum, other_grads.sum(dim=0)), 'Grad sums are not equal!')
                self.assertTrue(torch.allclose(other_grad_sum, grad_sum), 'Grad sums are not equal!')

                other_fst_factor = (ttv.factorial(p_order) / C) ** (1 / p_order)
                other_snd_factor = other_grad_sum / (other_grad_sum.norm() ** ((p_order - 1) / p_order))

                other_v_val = -other_fst_factor * other_snd_factor
                other_v_val.requires_grad_(True)
                assert torch.allclose(v, other_v_val)


if __name__ == '__main__':
    unittest.main()
