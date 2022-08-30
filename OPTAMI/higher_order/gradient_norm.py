from math import factorial

import torch
from torch.optim import Optimizer
from tqdm.auto import trange

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
        calculate_primal_var=None,
    ):
        assert torch.all(
            params[0] == 0
        ), "R upper bound is written only for \
            x_0 = 0!"

        #         if calculate_primal_var is None:
        #             raise ValueError(
        #                 "We need function for primal (x) value calculation from lambda (dual) variable"
        #             )
        self._calculate_primal_var = calculate_primal_var

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
        defaults = dict(
            M_p=M_p,
            M_mu=M_mu,
            mu=mu,
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
        state["x"] = None
        state["phi_mu"] = None
        state["grad_phi_mu"] = None

    def step(self, closure, verbose_ls=False) -> None:
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        mu = group["mu"]
        state = self.state["default"]
        k = state["k"]

        # step 3
        if k > 0:
            state["R"] /= 2

        # step 4
        # self._run_inner_tensor_method(closure, mu, verbose_inner, verbose_ls)
        self._init_inner_tensor_method()
        self.inner_tensor_method.step(closure, verbose_ls)

        # step 5
        state["k"] += 1
        self._update_state(closure)

    # def _run_inner_tensor_method(self, closure, mu, verbose_inner=False, verbose_ls=False):
    #     state = self.state['default']
    #     A = 0
    #     # step 4
    #     self._init_inner_tensor_method()
    #     inner_tm_state = self.inner_tensor_method.state['default']

    #     max_iters = 1000
    #     if verbose_inner:
    #         postfix = {'A': inner_tm_state['A']}
    #         rng = trange(max_iters, postfix=postfix)
    #     else:
    #         rng = range(max_iters)
    #     for i in rng:
    #         self.inner_tensor_method.step(closure, verbose_ls)
    #         A = inner_tm_state["A"]
    #         if A >= 4 / mu:
    #             break

    #         if verbose_inner:
    #             postfix['A'] = A
    #             rng.set_postfix(postfix)
    #     else:
    #         raise Exception(f'Number of iterations of inner tensor method exceeds {max_iters}')
    #     if verbose_inner:
    #         rng.close()

    #     N_k = inner_tm_state['k']
    #     assert N_k > 0, "Inner tensor method did not make any iterations"
    #     state['inner_k'] = N_k

    def _update_state(self, closure):
        group = self.param_groups[0]
        param = group["params"][0]
        state = self.state["default"]

        if self._calculate_primal_var is not None:
            state["x"], _, _ = self._calculate_primal_var(param)
        phi = closure()
        state["phi_mu"] = phi.item()
        phi.backward()
        state["grad_phi_mu"] = param.grad.clone()
        self.zero_grad()

    def final_tensor_step(self, closure):
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
