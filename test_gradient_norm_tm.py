from math import ceil, factorial
from re import A
import unittest

import torch
from tqdm.auto import trange
from OPTAMI.higher_order.gradient_norm import GradientNormTensorMethod

from run_pd_experiment import init_data, init_gradient_norm_tm, phi


class GradientNormTMTestCase(unittest.TestCase):
    def setUp(self) -> None:
        M_p = 1e-3
        image_index = 2
        new_m = 10
        eps = 0.001
        device = "cpu"
        gamma = torch.tensor(0.1, device=device)

        n, M_matrix, p, q, p_ref, q_ref = init_data(image_index, new_m, eps, device)
        M_matrix_over_gamma = M_matrix / gamma
        ones = torch.ones(n, device=device, dtype=torch.double)

        lamb = torch.zeros(
            n * 2, dtype=torch.double, requires_grad=False, device=device
        )
        lamb.mul_(-1 / gamma).requires_grad_(True)

        self.optimizer = init_gradient_norm_tm(
            lamb, n, M_p, gamma, eps, M_matrix, p, q, ones, device
        )
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

    def test_sanity(self) -> None:
        state = self.optimizer.state["default"]

        self.assertEqual(state["k"], 0)
        self.assertEqual(state["inner_k"], 0)
        self.assertGreater(state["R"], 0)
        self.assertIsNone(state["x"])
        self.assertIsNone(state["phi_mu"])
        self.assertIsNone(state["grad_phi_mu"])

    def test_inner_tensor_method_iterations(self) -> None:
        group = self.optimizer.param_groups[0]
        M_p = group["M_p"]
        mu = group["mu"]
        state = self.optimizer.state["default"]
        p_order = self.optimizer.p_order
        p_fact = factorial(p_order)
        power = (3 * (p_order + 1) ** 2 + 4) / 4
        c = 2**power * (p_order + 1) / p_fact

        fst_term_power = 2 / (3 * p_order + 1)
        tqdm_postfix = {"N_k": 0, "N_k_upper": 0}
        with trange(10, desc="Inner method iterations", postfix=tqdm_postfix) as t:
            for _ in t:
                inner_k = state["inner_k"]
                if inner_k > 0:
                    state["R"] /= 2**inner_k

                R = state["R"]
                N_k_upper = max(
                    ceil((8 * c * M_p * R ** (p_order - 1) / mu) ** fst_term_power), 1
                )
                self.optimizer._run_inner_tensor_method(
                    self.closure, mu, verbose_ls=True
                )
                N_k = state["inner_k"]

                self.assertGreater(N_k, 0, "N_k == 0!")

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

                tqdm_postfix["N_k"] = N_k
                tqdm_postfix["N_k_upper"] = N_k_upper
                t.set_postfix(tqdm_postfix)
                state["k"] += 1

    # def test_fourth_order_polynom(self):
    #     x = torch.zeros(10, requires_grad=True)
    #     closure = lambda: x.sub(10).pow(4).sum()
    #     M_p = 1e-1
    #     eps = 0.1
    #     R = torch.tensor([10.] * 10).norm()
    #     calculate_primal_var = lambda x: x.norm()
    #     optimizer = GradientNormTensorMethod(
    #         [x],
    #         M_p,
    #         eps,
    #         R,
    #         3,
    #         calculate_primal_var=calculate_primal_var
    #     )
        
    #     output = closure()
    #     output.backward()
    #     grad = x.grad.clone()
    #     tqdm_postfix = {'grad.norm()': grad.norm().item()}
    #     with trange(10, postfix=tqdm_postfix) as t:
    #         for i in t:
    #             optimizer.zero_grad()
    #             optimizer.step(closure, verbose_ls=True)
    #             grad = optimizer.state['default']['grad_phi_mu']
    #             tqdm_postfix["grad.norm()"] = grad.norm().item()
    #             t.set_postfix(tqdm_postfix)
    #     optimizer.final_tensor_step(closure)
    #     grad = optimizer.state['default']['grad_phi_mu']
    #     print(f'final grad.norm() = {grad.norm()}')
    
    def test_tensor_step(self):
        x = torch.zeros(10, requires_grad=True)
        closure = lambda: x.sub(10).pow(4).sum()
        M_p = 1e-1
        eps = 0.1
        R = torch.tensor([10.] * 10).norm()
        calculate_primal_var = lambda x: x.norm()
        optimizer = GradientNormTensorMethod(
            [x],
            M_p,
            eps,
            R,
            3,
            calculate_primal_var=calculate_primal_var
        )
        prev_value = closure().detach().clone()
        optimizer.final_tensor_step(closure)
        next_value = closure().detach().clone()
        
        self.assertGreaterEqual(prev_value, next_value)
        print(prev_value, next_value)
        
    # def test_gradient_norm_minimization(self) -> None:
    #     group = self.optimizer.param_groups[0]
    #     param = group['params'][0]
    #     init_output = self.closure()
    #     init_output.backward()
    #     init_grad = param.grad.clone()
    #     grads = [init_grad]
    #     with trange(
    #         10,
    #         desc=f'grad.norm(): {init_grad.norm()}'
    #     ) as t:
    #         for i in t:
    #             self.optimizer.zero_grad()
    #             self.optimizer.step(self.closure)
    #             output = self.closure()
    #             output.backward()
    #             grad = param.grad.clone()
    #             if i > 0:
    #                 prev_grad = grads[-1]
    #                 self.assertGreaterEqual(
    #                     prev_grad.norm(),
    #                     grad.norm(),
    #                     'Gradient norm does not decrease!'
    #                 )
    #             grads.append(grad)
    #             t.set_description(
    #                 f'grad.norm(): {init_grad.norm()} -> {grad.norm()}'
    #             )
