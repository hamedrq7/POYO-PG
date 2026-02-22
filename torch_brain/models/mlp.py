from typing import Optional

import numpy as np
from temporaldata import IrregularTimeSeries
import torch.nn as nn 
import torch 
from torch_brain.utils.binning import bin_spikes
from torch_brain.registry import ModalitySpec

from torch_brain.utils import (
    prepare_for_readout,
)


from torch_brain.utils.binning import bin_spikes_and_location, binnnn

class MLPNeuralDecoder(nn.Module):
    def __init__(self, readout_spec: ModalitySpec, num_units, bin_size, sequence_length, hidden_dim):
        """Initialize the neural net layers."""
        super().__init__()
        self.readout_spec = readout_spec
        self.num_timesteps = int(sequence_length / bin_size)
        self.bin_size = bin_size
        self.num_units = num_units
        self.sequence_length = sequence_length

        self.net = nn.Sequential(
            nn.Linear(self.num_timesteps * num_units, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.readout_spec.dim * self.num_timesteps),
        )
        print(self.net)

    def forward(
            self, 
            x, 
            output_timestamps,
            output_mask,  
        ):
        """Produces predictions from a binned spiketrain.
        This is pure PyTorch code.

        Shape of x: (B, T, N)
        """ 
        # print(x.shape, x.flatten(1).shape)
        x = x.flatten(1)                          # (B, T, N)    -> (B, T*N)
        x = self.net(x)                           # (B, T*N)     -> (B, T*D_out)
        x = x.reshape(-1, self.num_timesteps, self.readout_spec.dim)  # (B, T*D_out) -> (B, T, D_out)
        return x


    def tokenize(self, data):
        """ Need padding for edge cases**** """

        """tokenizes a data sample, which is a sliced Data object"""
        start, end = 0, self.sequence_length
        
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # the weight and mask created by prepare_for_readout doesnt match with the binning we use, create them manually here for now
        # you can check thus by printing shape of y and mask and weight
        output_timestamps, output_values, _output_weights, _eval_mask = (
            prepare_for_readout(data, self.readout_spec)
        )         
        
        # print('built-in binning')
        # x1 = bin_spikes(
        #     spikes=data.spikes,
        #     num_units=len(data.units),
        #     bin_size=0.025, # self.bin_size,
        #     # num_bins=self.num_timesteps,
        #     eps = 1e-3
        # ).T
        # print('x', x1)
        # print('y', output_values)
        
        # # print('using new binning')
        # bin_size = 25 # (1/39.06263603480421) * 1000. # 25 # ms
        # x2, y2 = binnnn(spike_timestamps, spike_unit_index, output_values, output_timestamps, num_units=self.num_units, bin_size=bin_size)

        #         
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
        # print(x2.shape, y2.shape)
        # print(x3.shape, y3.shape)
        # print(artifical_timestamps_of_bins)
        # compare_binning_heatmaps(x2.T, y2, x3, y3)

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

    def original_tokenize(self, data):
        """tokenizes a data sample, which is a sliced Data object"""

        # A. Extract and bin neural activity (data.spikes)
        spikes = data.spikes
        x = bin_spikes(
            spikes=spikes,
            num_units=len(data.units),
            bin_size=self.bin_size,
            # num_bins=self.num_timesteps,
            eps = 1e-3
        ).T
        # Final shape of x here is (timestamps, num_neurons)

        # B. Extract the corresponding cursor velocity, which will act as targets
        #    for training the MLP.
        y = data.position.pos
        # Final shape of y is (timestamps x 2)
        # Note that in this example we have choosen the bin size to match the
        # sampling rate of the recorded cursor velocity.

        # print('---')
        # # print(data)
        # print(spikes.timestamps.min(), spikes.timestamps.max())
        # print(data.position.timestamps.min(), data.position.timestamps.max())
        # print(data.position.timestamps[0:5])
        # print('---')

        # Finally, we output the "tokenized" data in the form of a dictionary.
        data_dict = {
            "model_inputs": {
                "x": torch.tensor(x, dtype=torch.float32),
                # Models in torch_brain typically follow the convention that
                # fields that are input to model.forward() are stored in
                # "model_inputs". Although you are free to deviate from this,
                # we have found that this convention generally produces cleaner
                # training loops.
            },
            "target_values": torch.tensor(y, dtype=torch.float32),
        }
        return data_dict