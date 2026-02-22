import torch 
import torch.nn as nn
from torch_brain.nn import FeedForward
from torch_brain.registry import ModalitySpec
import numpy as np
 
from torch_brain.utils import (
    prepare_for_readout,
)

from torch_brain.utils.binning import bin_spikes_and_location, binnnn

def generate_sinusoidal_position_embs(num_timesteps, dim):
    position = torch.arange(num_timesteps).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe = torch.empty(num_timesteps, dim)
    pe[:, 0:dim // 2] = torch.sin(position * div_term)
    pe[:, dim//2:] = torch.cos(position * div_term)
    return pe


class TransformerNeuralDecoder(nn.Module):
    def __init__(
        self, readout_spec: ModalitySpec,
        num_units, bin_size, sequence_length,   # data properties
        dim_hidden, n_layers, n_heads,    # transformer properties
    ):
        """Initialize the neural net components"""
        super().__init__()

        self.sequence_length = sequence_length
        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size
        self.num_units = num_units
        
        self.readout_spec = readout_spec
        dim_output = self.readout_spec.dim 

        # Create the read-in/out linear layers
        self.readin = nn.Linear(num_units, dim_hidden)
        self.readout = nn.Linear(dim_hidden, dim_output)

        # Create the position embeddings
        # Note that these are kept constant in this implementation, i.e. _not_ learnable
        self.position_embeddings = nn.Parameter(
            data=generate_sinusoidal_position_embs(self.num_timesteps, dim_hidden),
            requires_grad=False, # can set to True here, but might overfit
        )

        # Create the transformer layers:
        # each composed of the Attention and the feedforward (FFN) blocks
        self.transformer_layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=dim_hidden,
                    num_heads=n_heads,
                    batch_first=True,
                ),
                FeedForward(dim=dim_hidden),
            ])
            for _ in range(n_layers)
        ])

    def forward(self, x, output_timestamps,
            output_mask,):
        """Produces predictions from a binned spiketrain.
        This is pure PyTorch code.

        Shape of x: (B, T, N)
        """

        # Read-in: converts our input marix to transformer tokens; one token for each timestep
        x = self.readin(x)  # (B, T, N) -> (B, T, D)

        # Add position embeddings to the tokens
        x = x + self.position_embeddings[None, ...]  # -> (B, T, D)

        # Transformer
        for attn, ffn in self.transformer_layers:
            x = x + attn(x, x, x, need_weights=False)[0]
            x = x + ffn(x)

        # Readout: converts tokens to 2d vectors; each vector signifying (v_x, v_y) at that timestep
        x = self.readout(x)  # (B, T, D) -> (B, T, d_out)

        return x

    def tokenize(self, data):
        """ Need padding for edge cases**** """

        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # the weight and mask created by prepare_for_readout doesnt match with the binning we use, create them manually here for now
        # you can check thus by printing shape of y and mask and weight
        output_timestamps, output_values, _output_weights, _eval_mask = (
            prepare_for_readout(data, self.readout_spec)
        )         

        x3, y3, bin_centers, bin_edges = bin_spikes_and_location(
            spikes_times=spike_timestamps,
            spike_unit_index=spike_unit_index,
            locations=output_values.squeeze(),
            location_times=output_timestamps.squeeze(),
            num_units=self.num_units,
            S=self.sequence_length,
            bin_size_ms=self.bin_size*1000.
        )
        artifical_timestamps_of_bins = bin_edges[:-1] # torch.arange(0.0, 1.0, (bin_size/1000.))

        output_mask = np.ones(len(y3), dtype=np.bool_)
        eval_mask = np.ones(len(y3), dtype=np.bool_)
        output_weights = np.ones(len(y3), dtype=np.float32)

        data_dict = {
            "model_inputs": {
                "x": torch.tensor(x3.T, dtype=torch.float32),
                # For output_timestamps, one option is to pass the original [0, 1] values, that follows the sampling rate of data (time is relative to global, it does not start from 0)
                # another option is to start from 0 and add a bin_size at each bin
                # another option is to start from (bin_size/2) and adda bin size
                "output_timestamps": artifical_timestamps_of_bins,  # output_timestamps, 
                ########### should not be used in calculations, just logging
                # "input_timestamps": spike_timestamps, 
                # "input_unit_index": spike_unit_index, 
                # "input_mask"
                "output_mask": output_mask, 
            },
            "target_values": torch.tensor(y3[:, None], dtype=torch.float32), # output_values
            "target_weights": output_weights, 
            # For evaluation when overlapping windows
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
            "eval_mask": eval_mask,
        }
        return data_dict