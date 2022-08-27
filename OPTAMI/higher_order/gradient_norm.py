from math import factorial

import torch
from torch.optim import Optimizer

import OPTAMI


class GradientNormTensorMethod(Optimizer):
    def __init__(
        self,
        params: list,
        M_p: float,
        eps: float,
        R: torch.float,
        p_order: int = 3,
        tensor_step_method: Optimizer = None,
        tensor_step_kwargs: dict = None,
        subsolver: Optimizer = None,
        subsolver_args: dict = None,
        max_iters: int = None,
        verbose: bool = None,
    ):
        assert torch.all(
            params[0] == 0
        ), "R upper bound is written only for \
            x_0 = 0!"

        self.p_order = p_order
        self.tensor_step_method = tensor_step_method
        self.tensor_step_kwargs = tensor_step_kwargs
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.max_iters = max_iters
        self.verbose = verbose

        self.inner_tensor_method = None

        M_mu = (p_order + 2) * M_p
        mu = eps / (4 * R)
        one_over_p = 1 / p_order
        eps_tilde = (eps / 2) ** (1 + one_over_p) / (
            4 * factorial(p_order + 2) * M_mu**one_over_p
        )
        defaults = dict(
            M_p=M_p,
            M_mu=M_mu,
            mu=mu,
            eps_tilde=eps_tilde,
        )
        super().__init__(params, defaults)

        self._init_state(R)

    def _init_inner_tensor_method(self):
        group = self.param_groups[0]
        params = group["params"]
        M_p = group["M_p"]
        self.inner_tensor_method = OPTAMI.NearOptimalTensorMethod(
            params,
            M_p,
            self.p_order,
            self.tensor_step_method,
            self.tensor_step_kwargs,
            self.subsolver,
            self.subsolver_args,
            self.max_iters,
            self.verbose,
        )

    def _init_state(self, R):
        assert len(self.param_groups) == 1
        params = self.param_groups[0]["params"]
        assert len(params) == 1

        # filling state
        state = self.state["default"]
        state["k"] = 0
        state["inner_k"] = 0
        state["R"] = R

    def step(self, closure) -> None:
        assert len(self.param_groups) == 1
        state = self.state["default"]

        # steps 2-6
        self._run_inner_cycle(closure)

        # step 7
        self._perform_tensor_step(closure)

        state["k"] += 1

    def _run_inner_cycle(self, closure):
        group = self.param_groups[0]
        mu = group["mu"]
        eps_tilde = group["eps_tilde"]
        state = self.state["default"]
        R = state["R"]

        # step 2
        while mu * R**2 / 2 >= eps_tilde:
            inner_k = state["inner_k"]
            if inner_k > 0:
                # step 3
                state["R"] /= 2**inner_k
            # step 4
            self._run_inner_tensor_method(closure, mu)
            inner_tm_state = self.inner_tensor_method.state["default"]
            N_k = inner_tm_state["k"]
            assert N_k > 0, "Inner tensor method did not make any iterations"

            # step 5
            state["inner_k"] += 1

    def _run_inner_tensor_method(self, closure, mu, verbose_ls=False):
        A = 0
        # step 4
        self._init_inner_tensor_method()
        while A < 4 / mu:
            self.inner_tensor_method.step(closure, verbose_ls)
            inner_tm_state = self.inner_tensor_method.state["default"]
            A = inner_tm_state["A"]

    def _perform_tensor_step(self, closure):
        group = self.param_groups[0]
        params = group["params"]
        M_mu = group["M_mu"]
        if self.tensor_step_method is None:
            if self.p_order == 3:
                self.tensor_step_method = OPTAMI.BasicTensorMethod(
                    params=params,
                    L=M_mu,
                    subsolver=self.subsolver,
                    subsolver_args=self.subsolver_args,
                    max_iters=self.max_iters,
                    verbose=self.verbose,
                )
            elif self.p_order == 2:
                self.tensor_step_method = OPTAMI.CubicRegularizedNewton(
                    params=params,
                    L=M_mu,
                    subsolver=self.subsolver,
                    subsolver_args=self.subsolver_args,
                    max_iters=self.max_iters,
                    verbose=self.verbose,
                )
            else:
                raise NotImplementedError(
                    f"Method for p = {self.p_order} \
                                          is not implemented!"
                )
        self.tensor_step_method.step(closure)
