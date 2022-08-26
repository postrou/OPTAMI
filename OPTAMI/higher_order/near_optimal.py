from math import factorial

import torch
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm

import OPTAMI


class NearOptimalTensorMethod(Optimizer):
    MONOTONE = True

    def __init__(
        self,
        params,
        L_p: float,
        p_order: int = 3,
        tensor_step_method: Optimizer = None,
        tensor_step_kwargs: dict = None,
        subsolver: Optimizer = None,
        subsolver_args: dict = None,
        max_iters: int = None,
        verbose: bool = None,
    ):
        self.p_order = p_order
        self.tensor_step_method = tensor_step_method
        self.subsolver = subsolver
        self.subsolver_args = subsolver_args
        self.max_iters = max_iters
        self.tensor_step_kwargs = tensor_step_kwargs
        self.verbose = verbose

        self.ls_lower = 1 / 2
        self.ls_upper = p_order / (p_order + 1)
        self.fact = factorial(p_order - 1)

        if L_p < 0:
            raise ValueError("Invalid M_p: {}".format(L_p))

        defaults = dict(
            L_p=L_p,
        )
        super().__init__(params, defaults)

        self._init_state()

    def _init_state(self):
        assert len(self.param_groups) == 1
        params = self.param_groups[0]["params"]
        assert len(params) == 1

        # filling state
        state = self.state["default"]
        state["A"] = 0.0
        state["a"] = 0.0
        state["k"] = 0
        param_copy = params[0].clone().detach()
        state["x"] = param_copy
        state["y"] = param_copy
        state["f"] = 0.0
        state["theta"] = 0.0
        state["lambda"] = 0
        state["ls_count"] = 0
        state["dzeta"] = 0.0

    def step(self, closure, verbose_ls=False) -> None:
        assert len(self.param_groups) == 1
        assert (
            len(self.param_groups[0]["params"]) == 1
        ), f"This method is implemented only for single parameter!"
        state = self.state["default"]

        # step 4
        self._compute_lambda_and_y(closure, verbose_ls)

        # step 5
        self._perform_gradient_step(closure)

        state["k"] += 1

    def _compute_lambda_and_y(self, closure, verbose_ls=False) -> None:
        group = self.param_groups[0]
        param = group["params"][0]
        L_p = group["L_p"]
        state = self.state["default"]
        k = state["k"]
        if k == 0:
            x_tilde = param.detach().clone()
            self._perform_tensor_step(closure)
            y_next = param.detach().clone()
            y_next_norm_pow = torch.norm(y_next - x_tilde)**(self.p_order - 1)
            state["y"] = y_next
            lamb_next = 7 / 12 * self.fact / (L_p * y_next_norm_pow)
        else:
            lamb_next = self._line_search(closure, verbose_ls)
        A = state["A"]
        a_next = (lamb_next + (lamb_next**2 + 4 * lamb_next * A) ** 0.5) / 2
        state["a"] = a_next
        state["A"] = A + a_next
        state["lambda"] = lamb_next

    def _line_search(self, closure, verbose_ls=False) -> float:
        group = self.param_groups[0]
        param = group["params"][0]
        L_p = group["L_p"]
        state = self.state["default"]
        A = state["A"]
        x = state["x"]
        y = state["y"]
        ls_count = 0
        theta_min = 0
        theta_max = 1
        dzeta = 0
        upper = self.ls_upper
        lower = self.ls_lower
        upper_ineq = dzeta > upper
        lower_ineq = dzeta < lower

        if verbose_ls:
            init_lamb = state["lambda"]
            init_dzeta = dzeta
            init_theta = 1
            init_theta_min = theta_min
            init_theta_max = theta_max
            tqdm_postfix = ", ".join(
                [
                    f"lamb: {init_lamb}",
                    f"dzeta: {init_dzeta}",
                    f"lower: {lower}",
                    f"upper: {upper}",
                    f"theta: {init_theta}",
                    f"theta_min: {init_theta_min}",
                    f"theta_max: {init_theta_max}",
                ]
            )
            t = tqdm(desc="Step #0", postfix=tqdm_postfix)

        while upper_ineq or lower_ineq:
            theta = (theta_max + theta_min) / 2
            with torch.no_grad():
                param.zero_().add_(y, alpha=theta).add_(x, alpha=1 - theta)
            x_tilde = param.detach().clone()

            self._perform_tensor_step(closure)

            y_theta = group['params'][0].detach()
            z_theta = y_theta - x_tilde
            norm_z_theta_pow = z_theta.norm() ** (self.p_order - 1)

            lamb_next = (1 - theta) ** 2 * A / theta
            dzeta = lamb_next * L_p * norm_z_theta_pow / self.fact
            upper_ineq = dzeta > upper
            lower_ineq = dzeta < lower
            if upper_ineq:
                theta_min = theta
            elif lower_ineq:
                theta_max = theta

            ls_count += 1
            if ls_count == 10000:
                raise Exception('Number of lenear search iterations is greater \
                    than 10000!')

            if verbose_ls:
                t.set_description(f"Step #{ls_count}", refresh=False)
                tqdm_postfix = ", ".join(
                    [
                        f"lambda: {init_lamb} -> {lamb_next}",
                        f"dzeta: {init_dzeta} -> {dzeta.item()}",
                        f"lower: {lower}",
                        f"upper: {upper}",
                        f"theta: {init_theta} -> {theta}",
                        f"theta_min: {init_theta_min} -> {theta_min}",
                        f"theta_max: {init_theta_max} -> {theta_max}",
                    ]
                )
                t.set_postfix_str(tqdm_postfix)

        if verbose_ls:
            t.close()
        state["y"] = y_theta.clone()
        state["theta"] = theta
        state["ls_count"] = ls_count
        state["dzeta"] = dzeta

        return lamb_next

    def _perform_tensor_step(self, closure):
        group = self.param_groups[0]
        params = group["params"]
        L_p = group["L_p"]
        if self.tensor_step_method is None:
            if self.p_order == 3:
                self.tensor_step_method = OPTAMI.BasicTensorMethod(
                    params=params,
                    L=L_p,
                    subsolver=self.subsolver,
                    subsolver_args=self.subsolver_args,
                    max_iters=self.max_iters,
                    verbose=self.verbose,
                )
            elif self.p_order == 2:
                self.tensor_step_method = OPTAMI.CubicReguralizedNewton(
                    params=params,
                    L=L_p,
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

    def _perform_gradient_step(self, closure):
        param = self.param_groups[0]["params"][0]
        state = self.state["default"]
        a_next = state["a"]
        self.zero_grad()
        f = closure()
        f.backward()
        grad_f = param.grad.clone()
        state["x"] -= a_next * grad_f
        state["f"] = f.detach().clone()
