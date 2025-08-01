import torch

from typing import Optional

from .ssm import SSM

class MIMOSSM(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 step_scale: float = 1.0,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=True,
                 B_C_init='orthogonal',
                 stability='abs'):
        super().__init__()
        self.d_in = d_in
        self.d_state = d_state
        self.d_out = d_out
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.step_scale = step_scale
        self.previous_step_scale = step_scale
        self.complex_output = complex_output

        self.seq = SSM(
            d_in,
            d_state,
            d_out,
            dt_min,
            dt_max,
            step_scale,
            input_bias=input_bias,
            bias_init=bias_init,
            output_bias=output_bias,
            complex_output=complex_output,
            B_C_init=B_C_init,
            ensure_stability=stability
        )

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self, signal):
        # return torch.vmap(lambda s: self.seq(s))(signal)
        return self.seq(signal)

    def set_step_scale(self, step_scale):
        self.step_scale = step_scale
        self.seq.step_scale = step_scale