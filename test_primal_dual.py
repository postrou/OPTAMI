import unittest

import torch.autograd.functional as AF
from torch.optim.optimizer import Optimizer
from tqdm import trange

from OPTAMI import PrimalDualTensorMethod
from run_pd_experiment import *


class PrimalDualAcceleratedTester(PrimalDualTensorMethod):
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
        calculate_grad_phi=None,
    ):
        super().__init__(
            params,
            M_p,
            p_order,
            tensor_step_method,
            tensor_step_kwargs,
            subsolver,
            subsolver_args,
            max_iters,
            verbose,
            calculate_primal_var,
            keep_psi_data=True,
        )
        self._calculate_grad_phi = calculate_grad_phi

        state = self.state["default"]

        state["grad_psi_norm"] = None
        state["other_grad_psi_norm"] = torch.tensor(0)


class PrimalDualTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_steps = 100

        self.device = "cpu"
        self.gammas = [0.001, 0.5, 1, 1.5]
        image_index = 2
        new_m = 10
        images, labels = load_data()
        if new_m is not None:
            n = new_m**2
            m = new_m
        else:
            n = len(images[0])
            m = int(np.sqrt(n))

        p_list = [34860, 31226, 239, 37372, 17390]
        q_list = [45815, 35817, 43981, 54698, 49947]

        eps = 0.001
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
            lamb = torch.zeros(
                n * 2, dtype=torch.double, requires_grad=False, device=self.device
            )
            lamb.mul_(-1 / gamma).requires_grad_(True)

            self.primal_var_funcs.append(
                lambda dual_var, gamma=gamma, M_matrix_over_gamma=M_matrix_over_gamma, ones=ones: calculate_x(
                    dual_var, n, gamma, M_matrix_over_gamma, ones
                )
            )

            self.grad_phi_funcs.append(
                lambda dual_var, gamma=gamma, primal_var_func=self.primal_var_funcs[
                    -1
                ], ones=ones: grad_phi(
                    dual_var, gamma, primal_var_func, p, q, ones, self.device
                )
            )

            self.phi_funcs.append(
                lambda dual_var, gamma=gamma, M_matrix_over_gamma=M_matrix_over_gamma, ones=ones: phi(
                    dual_var, n, gamma, M_matrix_over_gamma, ones, p, q
                )
            )

            optimizer = PrimalDualAcceleratedTester(
                [lamb],
                M_p=M_p,
                p_order=torch.tensor(3, device=self.device),
                calculate_primal_var=self.primal_var_funcs[-1],
                calculate_grad_phi=self.grad_phi_funcs[-1],
            )

            closure = lambda dual_var=lamb, gamma=gamma, M_matrix_over_gamma=M_matrix_over_gamma, ones=ones, optimizer=optimizer: phi(
                dual_var, n, gamma, M_matrix_over_gamma, ones, p, q, optimizer=optimizer
            )

            self.optimizers.append(optimizer)
            self.closures.append(closure)

    def test_subproblem(self):
        for gamma, optimizer, closure in zip(
            self.gammas, self.optimizers, self.closures
        ):
            p_fact = 1
            for i in range(2, optimizer.p_order + 1):
                p_fact *= i

            with trange(1, self.n_steps + 1) as t:
                for i in t:
                    optimizer.step(closure)

                    if self.device != "cpu":
                        torch.cuda.empty_cache()

                    param_group = optimizer.param_groups[0]
                    params = param_group["params"]
                    C = param_group["C"]

                    state = optimizer.state["default"]
                    A_arr = state["A_arr"]
                    phi_arr = state["A_arr"]
                    grad_phi_arr = state["grad_phi_arr"]
                    param_arr = state["param_arr"]
                    grad_sum = state["grad_phi_sum"][0]

                    v = state["v"][0].detach().clone().requires_grad_(True)
                    t.set_description(
                        f"test_subproblem, gamma = {gamma}, v.norm()={v.norm()}"
                    )
                    psi = optimizer.psi(
                        v, A_arr, phi_arr, grad_phi_arr, param_arr, param_group
                    )
                    psi.backward()
                    grad_psi = v.grad

                    v_test = torch.zeros_like(v).requires_grad_(True)
                    grad_sum_test = torch.zeros_like(params[0])
                    grads_test = torch.zeros(i, len(grad_psi), dtype=torch.double)
                    for j in range(1, i + 1):
                        # test grads
                        A = A_arr[j]
                        A_prev = A_arr[j - 1]
                        self.assertGreater(A, A_prev)

                        lamb = param_arr[j]
                        grad_phi = grad_phi_arr[j]
                        explicit_grad_phi = optimizer._calculate_grad_phi(lamb)
                        self.assertTrue(
                            torch.allclose(grad_phi, explicit_grad_phi),
                            f"Autograd gradient and explicit gradient are not equal on step {j}! "
                            f"Maximal element is {torch.abs(grad_phi - explicit_grad_phi).argmax()},"
                            f"{torch.abs(grad_phi - explicit_grad_phi).max()}",
                        )

                        grads_test[j - 1] = (A - A_prev) * explicit_grad_phi
                        grad_sum_test += (A - A_prev) * explicit_grad_phi
                        if j == i - 1:
                            # since v calculation doesn't include grad_phi_next
                            test_fst_factor = (p_fact / C) ** (1 / optimizer.p_order)
                            self.assertEqual(
                                test_fst_factor,
                                param_group["step_3_fst_factor"],
                                "\n".join(
                                    [
                                        f"Step 3 first factors are not equal!",
                                        f'original = {param_group["step_3_fst_factor"]}',
                                        f"test = {test_fst_factor}",
                                    ]
                                ),
                            )
                            other_snd_factor = grad_sum_test / (
                                grad_sum_test.norm() ** (1 - 1 / optimizer.p_order)
                            )

                            v_test = -test_fst_factor * other_snd_factor
                            v_test.requires_grad_(True)

                    psi_test = optimizer.psi(
                        v_test, A_arr, phi_arr, grad_phi_arr, param_arr, param_group
                    )
                    grad_psi_test = torch.autograd.grad(psi_test, v_test)[0]

                    stable_grad_sum = grads_test.sum(dim=0)
                    self.assertTrue(
                        torch.allclose(grad_sum_test, stable_grad_sum),
                        "\n".join(
                            [
                                "Stable grad sum and method grad sum are not close!",
                                f"grad_sum_test.norm() = {grad_sum_test.norm()}",
                                f"stable_grad_sum.norm() = {stable_grad_sum.norm()}",
                            ]
                        ),
                    )

                    self.assertTrue(
                        torch.allclose(grad_sum_test, grad_sum),
                        "\n".join(
                            [
                                "Grad sums are not close!",
                                f"grad_sum.norm() = {grad_sum.norm()}",
                                f"grad_sum_test.norm() = {grad_sum_test.norm()}",
                            ]
                        ),
                    )

                    self.assertTrue(
                        torch.allclose(v, v_test),
                        "\n".join(
                            [
                                "v values are not close!",
                                f"v.norm() = {v.norm()}",
                                f"v_test.norm() = {v_test.norm()}",
                            ]
                        ),
                    )

                    self.assertTrue(
                        torch.allclose(grad_psi, grad_psi_test),
                        "\n".join(
                            [
                                f"Grad psi values are not close!",
                                f"grad_psi.norm() = {grad_psi.norm()}",
                                f"grad_psi_test.norm() = {grad_psi_test.norm()}",
                                f"v.norm() = {v.norm()}",
                                f"v_test.norm() = {v_test.norm()}",
                                f"grad_sum.norm() = {grad_sum.norm()}",
                                f"grad_sum_test.norm() = {grad_sum_test.norm()}",
                                f"stable_grad_sum.norm() = {stable_grad_sum.norm()}",
                            ]
                        ),
                    )

    def test_hessian(self):
        for gamma, optimizer, closure, phi_func in zip(
            self.gammas, self.optimizers, self.closures, self.phi_funcs
        ):
            for _ in trange(self.n_steps, desc=f"test_hessian, gamma = {gamma}"):
                optimizer.step(closure)
                if self.device != "cpu":
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    lamb = optimizer.param_groups[0]["params"][0]
                    phi_hess = AF.hessian(phi_func, lamb)
                    phi_hess_norm = torch.norm(phi_hess, "fro")
                    self.assertNotAlmostEqual(
                        phi_hess_norm.item(), 0.0, msg="Phi hessian norm equals zero!"
                    )
                    self.assertFalse(
                        torch.isnan(phi_hess_norm), msg="Phi hessian norm is nan!"
                    )


if __name__ == "__main__":
    unittest.main()
