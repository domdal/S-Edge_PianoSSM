from typing import Optional

import torch
from torch.nn import functional as F
import numpy as np

from .associative_scan import apply_ssm
from .init import make_linear_eigenvalues, init_log_steps, S5_init


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt

def discretize_zoh(Lambda, B, B_bias, Delta, bias):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B      (complex64): input matrix + bias                (P, H + 1)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H + 1)
    """
    if bias:
        B_concat = torch.cat((B, B_bias.unsqueeze(1)), dim=-1)
    else:
        B_concat = B
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = ((Lambda_bar - 1)/Lambda)[..., None] * B_concat
    return Lambda_bar, B_bar

def discretize_zoh_norm_angle(norm_angle, B, B_bias, Delta, bias):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B      (complex64): input matrix + bias                (P, H + 1)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H + 1)
    """
    if bias:
        B_concat = torch.cat((B, B_bias.unsqueeze(1)), dim=-1)
    else:
        B_concat = B
    Lambda_bar = torch.exp(torch.exp(norm_angle) * Delta)
    B_bar = ((Lambda_bar - 1)*torch.exp(-norm_angle))[..., None] * B_concat
    return Lambda_bar, B_bar

def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt

class SSM(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 dt_min: float,
                 dt_max: float,
                 step_scale: float = 1.0,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=False,
                 B_C_init='orthogonal',
                 ensure_stability='abs',
                 symmetric=True,
                 what2train = 'phase_norm'
                 ): 
        """The Modified S5 SSM
        Args:
            d_in        (int32):     Number of features of input
            d_state     (int32):     state size
            d_out       (int32):     Number of output features
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        self.symmetric = symmetric

        # lambdaInit  (float32): Initial diagonal state matrix       (P,2)
        self.Lambda = torch.nn.Parameter(make_linear_eigenvalues(d_state, symmetric=self.symmetric))
        self.log_step = torch.nn.Parameter(init_log_steps(d_state, dt_min, dt_max))
        self.discretize = discretize_zoh_norm_angle

        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.step_scale = step_scale
        self.ensure_stability = ensure_stability
        self.what2train = what2train

        if self.input_bias:
            if bias_init == 'zero':
                self.B_bias = torch.nn.Parameter(
                    torch.zeros(d_state, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.B_bias = torch.nn.Parameter(
                    torch.rand(d_state, 2, dtype=torch.float))
        else:
            self.B_bias = torch.nn.Parameter(torch.zeros(
                d_state, 2, dtype=torch.float), requires_grad=False)

        if self.output_bias:
            if bias_init == 'zero':
                self.C_bias = torch.nn.Parameter(
                    torch.zeros(d_out, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.C_bias = torch.nn.Parameter(
                    torch.rand(d_out, 2, dtype=torch.float))
        else:
            self.C_bias = torch.nn.Parameter(torch.zeros(
                d_out, 2, dtype=torch.float), requires_grad=False)
        if B_C_init == 'S5':
            lamb, B, C = S5_init(d_in, d_out, d_state)
            self.Lambda.data = lamb
            self.B = torch.nn.Parameter(B,requires_grad=True)
            self.C = torch.nn.Parameter(2*C,requires_grad=True)
            self.B_bias = torch.nn.Parameter(self.B_bias.data[:self.Lambda.shape[0],...],requires_grad=True)
            self.log_step = torch.nn.Parameter(init_log_steps(self.Lambda.shape[0], dt_min, dt_max))

            print('S5 init')
            print('A', self.Lambda.shape)
            print('B', self.B.shape)
            print('C', self.C.shape)
            print('B_bias', self.B_bias.shape)
            print('C_bias', self.C_bias.shape)


        elif B_C_init == 'orthogonal':
            gain = np.sqrt(4/12)
            B_r = torch.empty(d_state, d_in)
            B_i = torch.empty(d_state, d_in)
            if d_in == 1:
                B_r = torch.nn.init.normal_(B_r, std=gain*10)
                B_i = torch.nn.init.normal_(B_i, std=gain*10)
            else:
                B_r = torch.nn.init.orthogonal_(B_r.T, gain=gain*10).T
                B_i = torch.nn.init.orthogonal_(B_i.T, gain=gain*10).T

            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))

            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_uniform_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_uniform_(B_i,  nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_uniform_(C_r,  nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_uniform_(C_i,  nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_normal_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_normal_(B_i,  nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_normal_(C_r,  nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_normal_(C_i,  nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_uniform_(
                B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_uniform_(
                B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_uniform_(
                C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_uniform_(
                C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_normal_(
                B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_normal_(
                B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_normal_(
                C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_normal_(
                C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        # print('A', self.Lambda.shape)
        # print('B', self.B.shape)
        # print('C', self.C.shape)
        # print('B_bias', self.B_bias.shape)
        # print('C_bias', self.C_bias.shape)
        
        if self.what2train == 'phase_norm' or self.what2train == 'real_eigenvalues':
            real_dynamics = self.Lambda * torch.exp(self.log_step).view(-1, 1)
            eigenvalues = as_complex(real_dynamics)
            self.norm = torch.nn.Parameter(torch.log(torch.abs(eigenvalues)).detach())
            self.angle = torch.nn.Parameter(torch.angle(eigenvalues).detach())

            # self.Real_Lambda = torch.nn.Parameter(real_dynamics.detach())

            B_c = as_complex(self.B)
            B_bias_c = as_complex(self.B_bias)
            Lambda_bars, B_bars = self.discretize(as_complex(self.Lambda), B_c, B_bias_c,  self.step_scale * torch.exp(self.log_step), self.input_bias)

            B_bias_b = B_bars * (eigenvalues/(torch.exp(eigenvalues)-1))[..., None]

            #B_concat = torch.cat((B, B_bias.unsqueeze(1)), dim=-1)
            if self.input_bias:
                self.B2 = torch.nn.Parameter(B_bias_b[..., 0:-1].clone().detach())
                self.B2_bias = torch.nn.Parameter(B_bias_b[..., -1].clone().detach())
            else:
                self.B2 = torch.nn.Parameter(B_bias_b.clone().detach())
                self.B2_bias = None#torch.nn.Parameter(torch.zeros_like(B_bias_b[..., -1].clone().detach()))
            # eigenvalues = torch.exp(1j*self.angle)
            # B_bias_b = B_bars * (eigenvalues/(torch.exp(eigenvalues)-1))[..., None]
            
            # self.B3 = torch.nn.Parameter(B_bias_b[..., 0:-1].clone().detach())
            # self.B3_bias = torch.nn.Parameter(B_bias_b[..., -1].clone().detach())

            # print('orig_init', real_dynamics)
            # print('fake_init',torch.exp(self.norm + 1j*self.angle))

            delattr(self, 'B')
            delattr(self, 'B_bias')
            delattr(self, 'Lambda')
            delattr(self, 'log_step')

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C.shape[-2]))

    def forward_rnn(self, signal, prev_state):
        assert False, "not implemented for norm angle SSMs"
        Lambda_c = as_complex(self.Lambda)
        if self.ensure_stability == 'relu':
            Lambda_c = torch.complex(-F.relu(-Lambda_c.real), Lambda_c.imag)
            # Lambda_c.real = -F.relu(-Lambda_c.real) # Ensure stability
        elif self.ensure_stability == 'abs':
            Lambda_c = torch.complex(-torch.abs(Lambda_c.real), Lambda_c.imag)
        else:
            # raise not implemented error
            raise NotImplementedError(
                'Only relu and abs stability are implemented')

        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        cinput_sequence = signal.type(C_c.dtype)

        step = self.step_scale * torch.exp(self.log_step)
        # print('step', step)
        Lambda_bar, B_bars = self.discretize(
            Lambda_c, B_c, B_bias_c, step, self.input_bias)

        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bar[:, 0])

        # print('Lambda_bar', Lambda_bar.shape)
        # print('input', cinput_sequence)
        # print('B_bar_forward_rnn', B_bar)
        # print('B_bias_bar_forward_rnn', B_bias_bar)
        # print('C_c', C_c)

        Bu = B_bar @ cinput_sequence + B_bias_bar
        x = Lambda_bar * prev_state + Bu
        y = C_c @ x + C_bias_c

        if self.complex_output:
            y_out = y
        else:
            y_out = y.real
        return y_out, x

    def forward(self, signal):
        with torch.no_grad():
            if self.ensure_stability == 'relu':
                eigenvalue = torch.exp(self.norm.data + 1j*self.angle.data) 
                eigenvalue.real = -F.relu(-eigenvalue.real) 
                self.angle.data = torch.angle(eigenvalue)
                # self.Lambda.data[:, 0] = -F.relu(-self.Lambda.data[:, 0])
                # Lambda_c.real = -F.relu(-Lambda_c.real) # Ensure stability
            elif self.ensure_stability == 'abs':
                eigenvalue = torch.exp(self.norm.data + 1j*self.angle.data) 
                eigenvalue.real = -torch.abs(-eigenvalue.real) 
                self.angle.data = torch.angle(eigenvalue)
                # self.Lambda.data[:, 0] = -torch.abs(self.Lambda.data[:, 0])
                # Lambda = torch.complex(-torch.abs(Lambda.real), Lambda.imag)

            norm_angle = torch.complex(self.norm, self.angle)
            
            
            if not self.symmetric:
                assert False, "not implemented for norm angle SSMs with non-symmetric eigenvalues"
                # self.Lambda.data[:, 1] = torch.abs(self.Lambda.data[:, 1])

        # Lambda = as_complex(self.Lambda)

        step = self.step_scale # * torch.exp(self.log_step)
        # print('Lambda', Lambda.shape)

        B_c = self.B2
        B_bias_c = self.B2_bias
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        Lambda_bars, B_bars = self.discretize(
            norm_angle, B_c, B_bias_c, step, self.input_bias)
        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bars[:, 0])
        # forward = apply_ssm
        return apply_ssm(Lambda_bars, B_bar, B_bias_bar, C_c, C_bias_c, signal, self.complex_output)
