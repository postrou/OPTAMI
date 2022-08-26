from math import ceil, factorial, floor
from sys import float_repr_style
import unittest

import torch
from tqdm.auto import trange

from OPTAMI import GradientNormTensorMethod
from run_pd_experiment import init_data, init_gradient_norm_tm, phi


class GradientNormTMTestCase(unittest.TestCase):
    def setUp(self) -> None:
        M_p = 1e-3
        image_index = 2
        new_m = 10
        eps = 0.001
        device = 'cpu'
        gamma = torch.tensor(0.1, device=device)

        n, M_matrix, p, q, p_ref, q_ref = init_data(image_index, new_m, eps, device)
        M_matrix_over_gamma = M_matrix / gamma
        ones = torch.ones(n, device=device, dtype=torch.double)

        lamb = torch.zeros(
            n * 2, dtype=torch.double, requires_grad=False, device=device
        )
        lamb.mul_(-1 / gamma).requires_grad_(True)

        self.optimizer = init_gradient_norm_tm(lamb, M_p, eps, n, gamma, M_matrix, p, q, device)
        group = self.optimizer.param_groups[0]
        mu = group["mu"]
        lamb_0 = lamb.detach().clone()
        self.closure = (
            lambda: phi(
                lamb,
                n,
                gamma,
                M_matrix_over_gamma,
                ones,
                p,
                q,
                optimizer=self.optimizer,
            )
            + mu / 2 * torch.norm(lamb - lamb_0) ** 2
        )

        return super().setUp()

    def test_sanity(self):
        state = self.optimizer.state['default']

        self.assertEqual(state['k'], 0)

    def test_inner_tensor_method_iterations(self):
        group = self.optimizer.param_groups[0]
        M_p = group["M_p"]
        mu = group["mu"]
        state = self.optimizer.state["default"]
        p_order = self.optimizer.p_order
        p_fact = factorial(p_order)
        power = (3 * (p_order + 1) ** 2 + 4) / 4
        c = 2**power * (p_order + 1) / p_fact


        fst_term_power = 2 / (3 * p_order + 1)
        tqdm_postfix = {'N_k': 0, 'N_k_upper': 0}
        with trange(10, desc='Inner method iterations', postfix=tqdm_postfix) as t:
            for _ in t:
                k = state["k"]
                if k > 0:
                    state["R"] /= 2**k

                R = state["R"]
                N_k_upper = max(
                    ceil((8 * c * M_p * R ** (p_order - 1) / mu) ** fst_term_power), 1
                )
                self.optimizer._run_inner_tensor_method(self.closure, mu, verbose_ls=True)
                inner_tm_state = self.optimizer.inner_tensor_method.state['default']
                N_k = inner_tm_state['k']

                self.assertGreater(N_k, 0, 'N_k == 0!')

                self.assertLessEqual(
                    N_k,
                    N_k_upper,
                    "\n".join(
                        [
                            f"Number of iterations of inner tensor method exceeds upper bound!",
                            f"N_k = {N_k}",
                            f"N_k upper bound = {N_k_upper}",
                        ]
                    ),
                )

                tqdm_postfix['N_k'] = N_k
                tqdm_postfix['N_k_upper'] = N_k_upper
                t.set_postfix(tqdm_postfix)
                state["k"] += 1
