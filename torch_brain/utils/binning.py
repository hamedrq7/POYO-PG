from typing import Optional

import numpy as np
from temporaldata import IrregularTimeSeries


def bin_spikes_and_location(
    spikes_times: np.ndarray,          # shape [N,]
    spike_unit_index: np.ndarray,      # shape [N,] int in [0, C-1]
    locations: np.ndarray,             # shape [M,]
    location_times: np.ndarray,        # shape [M,]
    num_units: int,                    # C
    S: float,                          # sequence length (seconds)
    bin_size_ms: float                 # bin size (milliseconds)
):
    """
    Bins spikes into counts [C, B] and resamples location onto bin centers [B,].

    - Spike binning: counts spikes in half-open intervals [b*Δ, (b+1)*Δ)
    - Location binning: linear interpolation of (location_times, locations)
      evaluated at bin centers t_b = (b + 0.5)*Δ

    Returns:
        spike_counts: (C, B) int32
        loc_binned:   (B,) float32
        bin_centers:  (B,) float64
        bin_edges:    (B+1,) float64
    """
    spikes_times = np.asarray(spikes_times, dtype=np.float64)
    spike_unit_index = np.asarray(spike_unit_index, dtype=np.int64)
    locations = np.asarray(locations, dtype=np.float64)
    location_times = np.asarray(location_times, dtype=np.float64)

    if spikes_times.ndim != 1 or spike_unit_index.ndim != 1:
        raise ValueError("spikes_times and spike_unit_index must be 1D arrays.")
    if locations.ndim != 1 or location_times.ndim != 1:
        raise ValueError("locations and location_times must be 1D arrays.")
    if spikes_times.shape[0] != spike_unit_index.shape[0]:
        raise ValueError("spikes_times and spike_unit_index must have same length.")
    if locations.shape[0] != location_times.shape[0]:
        raise ValueError("locations and location_times must have same length.")
    if num_units <= 0:
        raise ValueError("num_units must be positive.")
    if S <= 0:
        raise ValueError("S must be positive.")
    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive.")

    delta = bin_size_ms / 1000.0  # seconds
    B = int(S / delta)            # as you specified
    if B <= 0:
        raise ValueError("B computed as int(S/delta) is 0; choose larger S or smaller bin_size_ms.")
    S_eff = B * delta             # effective covered duration [0, S_eff)

    # ---- Spike counts (C, B) ----
    # Keep spikes in [0, S_eff) so bin index is in [0, B-1]
    mask_t = (spikes_times >= 0.0) & (spikes_times < S_eff)
    st = spikes_times[mask_t]
    su = spike_unit_index[mask_t]

    # Optional sanity mask for unit indices
    mask_u = (su >= 0) & (su < num_units)
    st = st[mask_u]
    su = su[mask_u]

    bin_idx = (st / delta).astype(np.int64)  # floor for non-negative st
    spike_counts = np.zeros((num_units, B), dtype=np.int32)
    np.add.at(spike_counts, (su, bin_idx), 1)

    # ---- Location at bin centers (B,) ----
    bin_edges = np.arange(B + 1, dtype=np.float64) * delta
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # (b+0.5)*Δ

    # Assumes location_times is sorted; if not, sort it (safe, small cost)
    if location_times.size == 0:
        raise ValueError("location_times is empty.")
    if not np.all(np.diff(location_times) >= 0):
        order = np.argsort(location_times)
        location_times = location_times[order]
        locations = locations[order]

    # Linear interpolation onto bin centers.
    # np.interp clamps outside range to endpoints; since your times are within [0,S], this is fine.
    loc_binned = np.interp(bin_centers, location_times, locations).astype(np.float32)

    return spike_counts, loc_binned, bin_centers, bin_edges

import matplotlib.pyplot as plt

def compare_binning_heatmaps(x1, y1, x2, y2):
    """
    x1: [N, T1]
    y1: [T1]
    x2: [N, T2]
    y2: [T2]
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 18))

    # ---- Top Left: x1 ----
    im0 = axes[0, 0].imshow(x1, aspect='auto')
    axes[0, 0].set_title("Algo 1 - Binned Spikes (x1)")
    axes[0, 0].set_xlabel("Time Bin")
    axes[0, 0].set_ylabel("Neuron")
    fig.colorbar(im0, ax=axes[0, 0])

    # ---- Top Right: x2 ----
    im1 = axes[0, 1].imshow(x2, aspect='auto')
    axes[0, 1].set_title("Algo 2 - Binned Spikes (x2)")
    axes[0, 1].set_xlabel("Time Bin")
    axes[0, 1].set_ylabel("Neuron")
    fig.colorbar(im1, ax=axes[0, 1])

    # ---- Bottom Left: y1 ----
    im2 = axes[1, 0].imshow(y1[np.newaxis, :], aspect='auto')
    axes[1, 0].set_title("Algo 1 - Binned Location (y1)")
    axes[1, 0].set_xlabel("Time Bin")
    axes[1, 0].set_yticks([])  # Hide single row axis
    fig.colorbar(im2, ax=axes[1, 0])

    # ---- Bottom Right: y2 ----
    im3 = axes[1, 1].imshow(y2[np.newaxis, :], aspect='auto')
    axes[1, 1].set_title("Algo 2 - Binned Location (y2)")
    axes[1, 1].set_xlabel("Time Bin")
    axes[1, 1].set_yticks([])
    fig.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

def binnnn(spikes_times, spike_unit_index, locations, locations_times, num_units, bin_size):
    # bin size is ms 
    binned_spike_times = np.array(np.floor(spikes_times*1000/bin_size), dtype='int');
    # print(binned_spike_times)
    # print('maybe set sequence length to length of binned_spike_times and then add padding when needed? ')

    spike_by_neuron = np.zeros((binned_spike_times.max() - binned_spike_times.min()+1, num_units));
    
    # print(spike_by_neuron.shape)

    for it in range(binned_spike_times.shape[0]):
        spike_by_neuron[binned_spike_times[it]-binned_spike_times.min(), spike_unit_index[it]] += 1;
        
    binned_locations_times = np.array(np.floor(locations_times*1000/bin_size), dtype='int');

    locations_vec = np.zeros(spike_by_neuron.shape[0]);

    for it in range(len(binned_locations_times)):
        locations_vec[binned_locations_times[it] - binned_spike_times.min()] = locations[it];
    
    return spike_by_neuron, locations_vec

def bin_spikes(
    spikes: IrregularTimeSeries,
    num_units: int,
    bin_size: float,
    max_spikes: Optional[int] = None,
    right: bool = True,
    eps: float = 1e-3,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    r"""Bins spikes into time bins of size `bin_size`. If the total time spanned by
    the spikes is not a multiple of `bin_size`, the spikes are truncated to the nearest
    multiple of `bin_size`. If `right` is True, the spikes are truncated from the left
    end of the time series, otherwise they are truncated from the right end.

    Notes:
    - The number of units cannot be inferred from a subset of spikes,
      so `num_units` must be provided explicitly.
    - Floating-point roundoff can cause `(end - start) / bin_size` to be
      very close to an integer without being exact (e.g. 9.99999999).
      The `eps` parameter is added before flooring to make the bin-count
      computation numerically robust.

    Args:
        spikes: IrregularTimeSeries object containing the spikes.
        num_units: Number of units in the population.
        bin_size: Size of the time bins in seconds.
        max_spikes: If provided, the maximum number of spikes per bin. Any bins
            exceeding this count will be clipped.
        right: If True, any excess spikes are truncated from the left end of the time
            series. Otherwise, they are truncated from the right end.
        eps : float, default=1e-3
            Small numerical tolerance added when computing the number of bins
            to avoid floating-point precision issues.
        dtype: Data type of the returned array.
    """
    start = spikes.domain.start[0]
    end = spikes.domain.end[-1]

    # Compute how much time must be discarded so that the duration
    # is an exact multiple of `bin_size`. The epsilon stabilizes
    # the floor operation under floating-point roundoff.
    discard = (end - start) - np.floor(((end - start) / bin_size) + eps) * bin_size
    # In theory, `discard` should always be non-negative.
    # Floating-point roundoff may make it slightly negative,
    # in that case, we avoid reslicing to prevent dropping the last spike.
    if discard > 0:
        if right:
            start += discard
        else:
            end -= discard
        # reslice
        spikes = spikes.slice(start, end)

    num_bins = round((end - start) / bin_size)

    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins), dtype=dtype)
    # Handle timestamps when the domain start is non-zero
    ts = spikes.timestamps - spikes.domain.start[0]
    bin_index = np.floor(ts * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)
    if max_spikes is not None:
        np.clip(binned_spikes, None, max_spikes, out=binned_spikes)

    return binned_spikes
