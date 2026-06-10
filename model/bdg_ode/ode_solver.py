"""ODE solver helpers.

原 BDG-ODE 仓库使用 ``torchdiffeq.odeint`` 来做连续时间积分。
这里默认也走 torchdiffeq；手写 Euler 只作为 fallback，方便在没有安装依赖时定位问题。
"""

from __future__ import annotations

import torch
from torch import nn

from config import (
    EULER_STEP_SIZE,
    ODE_ATOL,
    ODE_METHOD,
    ODE_RTOL,
    ODE_SOLVER_BACKEND,
    ODE_STEPS,
    ODE_TERMINAL_TIME,
    ODE_USE_ADJOINT,
)
from model.bdg_ode.dynamics import BDGODEFunc


class _TorchDiffEqWrapper(nn.Module):
    """把我们自己的 BDGODEFunc 包装成 torchdiffeq 需要的 f(t, y)。"""

    def __init__(self, func: BDGODEFunc, relation_adjs: dict[str, torch.Tensor]):
        super().__init__()
        self.func = func
        self.relation_adjs = relation_adjs

    def forward(
        self,
        _t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bull, bear = state
        return self.func(bull, bear, self.relation_adjs)


def integrate_bdg_ode(
    func: BDGODEFunc,
    bull0: torch.Tensor,
    bear0: torch.Tensor,
    relation_adjs: dict[str, torch.Tensor],
    steps: int = ODE_STEPS,
    step_size: float = EULER_STEP_SIZE,
    terminal_time: float = ODE_TERMINAL_TIME,
    method: str = ODE_METHOD,
    rtol: float = ODE_RTOL,
    atol: float = ODE_ATOL,
    use_adjoint: bool = ODE_USE_ADJOINT,
    backend: str = ODE_SOLVER_BACKEND,
) -> tuple[torch.Tensor, torch.Tensor]:
    """积分 BDG-ODE 状态，返回终止时刻的 bull/bear 状态。"""
    if backend == "torchdiffeq":
        return torchdiffeq_integrate(
            func=func,
            bull0=bull0,
            bear0=bear0,
            relation_adjs=relation_adjs,
            steps=steps,
            terminal_time=terminal_time,
            method=method,
            rtol=rtol,
            atol=atol,
            use_adjoint=use_adjoint,
        )
    if backend == "manual_euler":
        return euler_integrate(
            func=func,
            bull0=bull0,
            bear0=bear0,
            relation_adjs=relation_adjs,
            steps=steps,
            step_size=step_size,
        )
    raise ValueError(f"Unsupported ODE_SOLVER_BACKEND: {backend}")


def torchdiffeq_integrate(
    func: BDGODEFunc,
    bull0: torch.Tensor,
    bear0: torch.Tensor,
    relation_adjs: dict[str, torch.Tensor],
    steps: int = ODE_STEPS,
    terminal_time: float = ODE_TERMINAL_TIME,
    method: str = ODE_METHOD,
    rtol: float = ODE_RTOL,
    atol: float = ODE_ATOL,
    use_adjoint: bool = ODE_USE_ADJOINT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用 torchdiffeq.odeint 对齐原 BDG-ODE 的 ODEBlock。"""
    try:
        from torchdiffeq import odeint, odeint_adjoint
    except ImportError as exc:
        raise ImportError("torchdiffeq is required when ODE_SOLVER_BACKEND='torchdiffeq'") from exc

    if steps < 1:
        raise ValueError("ODE_STEPS must be >= 1")

    wrapped = _TorchDiffEqWrapper(func, relation_adjs)
    integration_time = torch.linspace(
        0.0,
        terminal_time,
        steps + 1,
        device=bull0.device,
        dtype=bull0.dtype,
    )
    solver = odeint_adjoint if use_adjoint else odeint
    bull_path, bear_path = solver(
        wrapped,
        (bull0, bear0),
        integration_time,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    return bull_path[-1], bear_path[-1]


def euler_integrate(
    func: BDGODEFunc,
    bull0: torch.Tensor,
    bear0: torch.Tensor,
    relation_adjs: dict[str, torch.Tensor],
    steps: int = ODE_STEPS,
    step_size: float = EULER_STEP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """不依赖 torchdiffeq 的固定步长 Euler fallback。"""
    bull, bear = bull0, bear0
    for _ in range(steps):
        d_bull, d_bear = func(bull, bear, relation_adjs)
        bull = bull + step_size * d_bull
        bear = bear + step_size * d_bear
    return bull, bear



