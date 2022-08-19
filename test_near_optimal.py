import unittest

import torch

from OPTAMI import NearOptimal
from run_pd_experiment import *


class NearOptimalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        params = torch.zeros(100, requires_grad=True)
        L_p = 1
        p_order = 3

        self.optimizer = NearOptimal([params], L_p, p_order)
        self.closure = lambda: (params + 10).norm()

        return super().setUp()

    def test_sanity(self):
        state = self.optimizer.state["default"]
        group = self.optimizer.param_groups[0]
        param_copy = group["params"][0].detach().clone()

        self.assertEqual(state["A"], 0.0)
        self.assertEqual(state["a"], 0.0)
        self.assertEqual(state["k"], 0)
        self.assertTrue(state["x"].equal(param_copy))
        self.assertTrue(state["y"].equal(param_copy))
        self.assertEqual(state["theta"], 0.0)
        self.assertEqual(state["lambda"], 0.0)
        self.assertEqual(state["ls_count"], 0)
        self.assertEqual(state["dzeta"], 0.0)

    def test_lambda_and_y_computation_first_iter(self) -> None:
        state = self.optimizer.state["default"]
        y = state["y"]
        A = state['A']
        self.optimizer._compute_lambda_and_y(self.closure)
        A_next = state['A']
        a_next = state['a']
        lamb_next = state["lambda"]
        y_next = state["y"]
        upper = self.optimizer.ls_upper
        lower = self.optimizer.ls_lower
        p_order = self.optimizer.p_order
        L_p = self.optimizer.param_groups[0]["L_p"]
        fact = self.optimizer.fact
        inequality = lamb_next * L_p * y_next.norm() ** (p_order - 1) / fact
        self.assertGreaterEqual(
            inequality,
            lower,
            "\n".join(
                [
                    f"After one iteration of (lambda, y) computation lambda is less than lower bound:",
                    f"inequality = {inequality}",
                    f"lower = {lower}",
                ]
            ),
        )
        self.assertLessEqual(
            inequality,
            upper,
            "\n".join(
                [
                    f"After one iteration of (lambda, y) computation lambda is greater than upper bound:",
                    f"inequality = {inequality}",
                    f"upper = {upper}",
                ]
            ),
        )
        self.assertFalse(
            y.equal(y_next),
            "\n".join(
                [
                    "After one iteration of (lambda, y) computation y and y_next are equal:",
                    f"y.norm() = {y.norm()}",
                ]
            ),
        )

    def test_lambda_and_y_computation_second_iter(self):
        state = self.optimizer.state["default"]
        group = self.optimizer.param_groups[0]
        param = group["params"][0]
        state["A"] = torch.rand(1).item()
        state["a"] = state["A"]
        state["k"] = 1
        state["x"] = torch.rand_like(param)
        state["y"] = torch.rand_like(param)
        A = state["A"]
        x = state["x"]
        y = state["y"]

        self.optimizer._compute_lambda_and_y(self.closure, verbose_ls=True)

        y_next = state["y"]
        a_next = state["a"]
        theta = state["theta"]
        A_next = state["A"]
        self.assertEqual(
            A_next,
            a_next + A,
            "\n".join(
                [
                    "A_next != A + a_next:",
                    f"A_next = {A_next},",
                    f"A = {A},",
                    f"a_next = {a_next},",
                    f"A + a_next = {A + a_next}",
                ]
            ),
        )
        self.assertEqual(
            theta,
            A / A_next,
            f"Wrong value of theta or A or A_next:\n\
                           theta = {theta},\n\
                           A / A_next = {A / A_next}",
        )

        x_tilde = A / A_next * y + a_next / A_next * x
        x_theta = theta * y + (1 - theta) * x
        self.assertTrue(
            torch.allclose(x_tilde, x_theta),
            f"x_tilde and x_theta are note equal:\n\
            x_tilde.norm() = {x_tilde.norm()},\n\
            x_theta.norm() = {x_theta.norm()}",
        )
        self.assertFalse(
            torch.allclose(y, y_next),
            "\n".join(
                [
                    "After one iteration of line search y and y_next are equal:",
                    f"y.norm() = {y.norm()}",
                ]
            ),
        )

        ls_count = state["ls_count"]
        self.assertGreater(
            ls_count,
            0,
            f"At first iteration of line search number of inner cycles is still zero",
        )

        upper = self.optimizer.ls_upper
        lower = self.optimizer.ls_lower
        p_order = self.optimizer.p_order
        L_p = self.optimizer.param_groups[0]["L_p"]
        fact = self.optimizer.fact
        norm_factor = torch.norm(y_next - x_tilde) ** (p_order - 1)
        lamb_next = state['lambda']
        inequality = (lamb_next * L_p * norm_factor / fact).item()
        self.assertGreaterEqual(
            inequality,
            lower,
            "\n".join(
                [
                    f"After one iteration of line search lambda is less than lower bound:",
                    f"inequality = {inequality}",
                    f"lower = {lower}",
                ]
            ),
        )
        self.assertLessEqual(
            inequality,
            upper,
            "\n".join(
                [
                    f"After one iteration of line search lambda is greater then upper bound:",
                    f"inequality = {inequality}",
                    f"upper = {upper}",
                ]
            ),
        )
