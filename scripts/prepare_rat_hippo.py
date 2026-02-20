"""Load data, processes it, save it."""

import argparse
import datetime
import logging

import numpy as np

import os
import logging


def find_files_by_extension(folder_path, extension):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(root, file)


from temporaldata import Data, IrregularTimeSeries, Interval, ArrayDict
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)

def extract_spikes_and_position_from_hippo(matfile, mode: 'str'):
    ## load spike info
    f = matfile
    spikes_times = np.array(f['sessInfo']['Spikes']['SpikeTimes'])[0];
    spikes_cells = np.array(f['sessInfo']['Spikes']['SpikeIDs'])[0];
    pyr_cells = np.array(f['sessInfo']['Spikes']['PyrIDs'])[0];
    
    ## load location info ## all in maze
    locations_2d = np.array(f['sessInfo']['Position']['TwoDLocation']).T;
    locations = np.array(f['sessInfo']['Position']['OneDLocation'])[0];
    print(locations.shape)

    locations_times = np.array(f['sessInfo']['Position']['TimeStamps'])[:,0];
    # linspeed_raw = np.array(f['sessInfo']['Position']['linspeed_raw'])[0];
    # linspeed_sm = np.array(f['sessInfo']['Position']['linspeed_sm'])[0];
    
    ## load maze epoch range
    maze_epoch = np.array(f['sessInfo']['Epochs']['MazeEpoch'])[:,0];
    wake_epoch = np.array(f['sessInfo']['Epochs']['Wake']);

    time_in_maze = ((spikes_times >= maze_epoch[0])*(spikes_times <= maze_epoch[1]));

    spikes_times = spikes_times[time_in_maze];
    spikes_cells = spikes_cells[time_in_maze];

    cell_mask = np.isin(spikes_cells, pyr_cells);
    spikes_times = spikes_times[cell_mask];
    spikes_cells = spikes_cells[cell_mask];

    cell_dic = {};
    for i,v in enumerate(pyr_cells):
        cell_dic[int(v)] = i;

    # locations_times: shape (M,), strictly increasing
    # locations:       shape (M,)
    # spikes_times:    shape (N,), sorted, no NaNs
    # spikes_cells:    shape (N,)

    # --- Step 0: masks and interval validity ---
    valid_loc_mask = ~np.isnan(locations)          # per-sample validity (M,)
    valid_intervals = valid_loc_mask[:-1]          # per-interval validity (M-1,)
    # If you prefer requiring BOTH endpoints valid, use:
    # valid_intervals = valid_loc_mask[:-1] & valid_loc_mask[1:]

    # --- Step 1: compute how much time is "removed" up to each location sample ---
    dt = np.diff(locations_times)                  # (M-1,)
    removed_dt = dt * (~valid_intervals)           # (M-1,) durations to remove
    cum_removed = np.concatenate([[0.0], np.cumsum(removed_dt)])  # (M,)

    # For a time t in interval i = [t_i, t_{i+1}), subtract cum_removed[i]

    # --- Step 2: assign each spike to a location interval index ---
    interval_idx = np.searchsorted(locations_times, spikes_times, side="right") - 1

    # --- Step 3: keep only spikes that land in a valid, in-range interval ---
    in_range = (interval_idx >= 0) & (interval_idx < len(valid_intervals))
    keep_spikes = in_range.copy()
    keep_spikes[in_range] &= valid_intervals[interval_idx[in_range]]

    # --- Step 4: remap (shift) spike timestamps to "gap-closed" time ---
    filtered_spike_time = spikes_times[keep_spikes] - cum_removed[interval_idx[keep_spikes]]
    filtered_spike_id   = spikes_cells[keep_spikes]

    # --- Step 5: filter + remap location samples too ---
    filtered_locations = locations[valid_loc_mask]
    filtered_location_time = locations_times[valid_loc_mask] - cum_removed[valid_loc_mask]

    # -------------------------------------------------------------------
    # NEW Step 6: drop spikes earlier than the first valid location time
    # -------------------------------------------------------------------
    t0_loc = filtered_location_time[0]   # first valid location time in gap-closed timeline
    eps = 1e-12                          # tiny tolerance for float comparisons

    keep2 = filtered_spike_time >= (t0_loc - eps)
    filtered_spike_time = filtered_spike_time[keep2]
    filtered_spike_id   = filtered_spike_id[keep2]

    # -------------------------------------------------------------------
    # NEW Step 7: shift everything so the timeline starts at 0
    # -------------------------------------------------------------------
    filtered_location_time = filtered_location_time - t0_loc
    filtered_spike_time    = filtered_spike_time - t0_loc

    spike_unit_index = np.zeros_like(filtered_spike_id, dtype = np.int32)
    
    for it in range(filtered_spike_id.shape[0]):
        spike_unit_index[it] = cell_dic[filtered_spike_id[it]]

    unit_ids = []
    for k, v in cell_dic.items(): 
        unit_ids.append(f'unit_{v}')

    return (
        np.array(unit_ids),
        filtered_spike_time, 
        spike_unit_index,
        filtered_location_time,
        filtered_locations[:, None],
    )

def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument(
        "--scramble",
        action="store_true",
        help="If passed, scrambles the timestep for positional data"
    )

    args = parser.parse_args()
    input_dir = args.input_dir # 'D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear/'
    output_dir = args.output_dir # 'D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear_processed/hippo_processed/rat_hippocampus'

    import h5py

    # iterate over the .nwb files and extract the data from each
    # for file_path in find_files_by_extension(db.raw_folder_path, ".nwb"):
    for file_path in find_files_by_extension(input_dir, ".mat"):
        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.

        # open file
        # # io = NWBHDF5IO(file_path, "r")
        # # nwbfile = io.read()

        # remember to close 
        matfile = h5py.File(file_path, "r")

        from pathlib import Path
        _filename = Path(file_path).stem.lower()

        from brainsets.descriptions import BrainsetDescription

        brainset_description = BrainsetDescription(
            id="rat_hippo",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://example.com/dataset",
            description="Description of your dataset..."
        )


        from brainsets.descriptions import SubjectDescription
        from brainsets.taxonomy import Species, Sex

        subject = SubjectDescription(
            id=f"{_filename}",
            species=Species.MACACA_MULATTA,  # or other species from taxonomy
            sex=Sex.MALE,  # or Sex.FEMALE, Sex.OTHER, Sex.UNKNOWN
        )
        
        from brainsets.descriptions import SessionDescription
        sess_id = f"{_filename}"
        session = SessionDescription(
            id=sess_id,
            recording_date=datetime.datetime(2024, 1, 1),
        )

        from brainsets.descriptions import DeviceDescription
        from brainsets.taxonomy import RecordingTech

        device = DeviceDescription(
            id="device_1",
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
        )

        unit_ids, spike_times, spike_unit_index, location_times, locations = extract_spikes_and_position_from_hippo(matfile=matfile, mode="idk")

        start = max(spike_times.min(), location_times.min())
        end = min(spike_times.max(), location_times.max())

        train_start_time = start
        train_end_time = (end * 0.6)
        valid_start_time = train_end_time
        valid_end_time = (end * 0.8)
        test_start_time = valid_end_time
        test_end_time = end 

        if args.scramble: 
            MODE = 'train_only__full_permutation'
            # scrambled_position_timesteps = np.random.permutation(position_timesteps)

            if MODE == "train_only__full_permutation":
                train_mask = (location_times >= train_start_time) & \
                            (location_times < train_end_time)

                train_indices = np.where(train_mask)[0]
                print('train_indices', train_indices.shape)
                shuffled_values = locations.copy()
                print(locations[train_indices].shape)
                shuffled_values[train_indices] = np.random.permutation(
                    locations[train_indices]
                )
                outside = ~((location_times >= train_start_time) & (location_times < train_end_time))
                assert np.all(shuffled_values[outside] == locations[outside])

                locations = shuffled_values.copy()

            elif MODE == "circular":
                print('not implemented yet')
                exit()
                # # MODE = 'circular'
                # # shift = np.random.randint(len(position_timesteps))
                # # scrambled_position_value = np.roll(position_value, shift)
                # shuffled_values = position_values.copy()
                # train_vals = position_values[train_indices]
                # shift = np.random.randint(len(train_vals))
                # shuffled_values[train_indices] = np.roll(train_vals, shift)

        # create the ArrayDict object for the units
        units = ArrayDict(id=np.array(unit_ids))
        
        spikes = IrregularTimeSeries(
            timestamps=spike_times,
            unit_index=spike_unit_index,
            domain=Interval(start=spike_times.min(), end=spike_times.max()),
        )
        spikes.sort()
        

        #### What to do here? your data is already filtered ### [?]
        # # extract data about trial structure
        # trials = extract_trials(nwbfile, task)
        # I think CEBRA treats this as a single trial, with invalid parts cut from data 
        # trials = Interval(spike_times[0], spike_times[-1])

        sampling_rate = 39.06263603480421
        # [?] in your main code, you used a `RegularTimeSeries`
        print(locations.shape)
        position = IrregularTimeSeries( 
            timestamps=location_times,
            pos=locations,
            # subtask_index=subtask_index,
            domain=Interval(start=location_times.min(), end=location_times.max()),
        )
        
        # # # extract behavior
        # # cursor = extract_behavior(nwbfile, trials, task)

        # close file
        matfile.close()

        data = Data(
            # metadata
            brainset=brainset_description,
            subject=subject,
            session=session,
            device=device,
            # neural activity
            spikes=spikes,
            units=units,
            # behavior
            # trials=trials,
            position=position,
            # domain
            domain=Interval(start=start, end=end),
        )


        print('train_start_time', train_start_time)
        print('train_end_time', train_end_time)
        print('valid_start_time', valid_start_time)
        print('valid_end_time', valid_end_time)
        print('test_start_time', test_start_time)
        print('test_end_time', test_end_time)
        data.set_train_domain(Interval(train_start_time, train_end_time))
        data.set_valid_domain(Interval(valid_start_time, valid_end_time))
        data.set_test_domain(Interval(test_start_time, test_end_time))
        

        plot_interval(data, sess_id, scrambled = args.scramble)

        import os   
        # save data to disk
        path = os.path.join(output_dir, f"{sess_id}.h5") if not args.scramble else os.path.join(output_dir, f"{sess_id}_scrambled.h5")
        with h5py.File(path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def _build_spike_image(spike_timesteps, spike_unit_index, n_units, t_min, t_max, dt):
    """
    Build a (n_units, n_bins) binary image for imshow, where each bin indicates
    whether a unit spiked in that time bin (clipped to 1 if multiple spikes land in same bin).
    """
    spike_timesteps = np.asarray(spike_timesteps)
    spike_unit_index = np.asarray(spike_unit_index)

    n_bins = int(np.floor((t_max - t_min) / dt)) + 1
    img = np.zeros((n_units, n_bins), dtype=np.uint8)

    # Convert timesteps to bin indices
    bin_idx = np.floor((spike_timesteps - t_min) / dt).astype(np.int64)
    unit_idx = spike_unit_index.astype(np.int64)

    valid = (
        np.isfinite(bin_idx) & np.isfinite(unit_idx) &
        (bin_idx >= 0) & (bin_idx < n_bins) &
        (unit_idx >= 0) & (unit_idx < n_units)
    )
    bin_idx = bin_idx[valid]
    unit_idx = unit_idx[valid]

    img[unit_idx, bin_idx] = 1
    return img


def plot_neural_splits(
    splits,
    dt=None,
    split_names=("train", "valid", "test"),
    figsize=(12, 8),
    cmap="gray_r",
):
    """
    Plot (spikes raster, position trace) for multiple splits in a 3x2 grid.

    - Left column (spikes): imshow raster (units x time)
    - Right column (position): position vs time

    Axes sharing between splits:
    - Spike plots share x and y with each other (left column)
    - Position plots share x and y with each other (right column)

    Parameters
    ----------
    splits : list/tuple of length 3
        Each split is a dict with keys:
          - "spike_timesteps": np.ndarray shape (N,)
          - "spike_unit_index": np.ndarray shape (N,)
          - "position_value": np.ndarray shape (M,)
          - "position_timesteps": np.ndarray shape (M,)

    dt : float or None
        Time bin size for spike raster. If None, inferred from position_timesteps (median diff), else 1.0.

    split_names : tuple/list of str
        Titles for each row.

    figsize : tuple
        Figure size.

    cmap : str
        Colormap for spike raster.
    """
    if len(splits) != 3:
        raise ValueError("Expected exactly 3 splits: [train, valid, test].")

    # --- Decide dt if not provided ---
    if dt is None:
        all_pos_diffs = []
        for sp in splits:
            pt = np.asarray(sp["position_timesteps"])
            if pt.size >= 2:
                diffs = np.diff(pt)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    all_pos_diffs.append(np.median(diffs))
        dt = float(np.median(all_pos_diffs)) if all_pos_diffs else 1.0

    def _finite_min(x):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        return float(x.min()) if x.size else None

    def _finite_max(x):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        return float(x.max()) if x.size else None

    # --- Global time window across splits (shared x-axis) ---
    tmins, tmaxs = [], []
    for sp in splits:
        st = np.asarray(sp["spike_timesteps"])
        pt = np.asarray(sp["position_timesteps"])
        cmin = [v for v in (_finite_min(st), _finite_min(pt)) if v is not None]
        cmax = [v for v in (_finite_max(st), _finite_max(pt)) if v is not None]
        if not cmin or not cmax:
            raise ValueError("Each split must have finite spike/position timesteps.")
        tmins.append(min(cmin))
        tmaxs.append(max(cmax))

    t_min = min(tmins)
    t_max = max(tmaxs)

    # --- Global unit count across splits (shared spike y-axis) ---
    n_units = 0
    for sp in splits:
        ui = np.asarray(sp["spike_unit_index"])
        ui = ui[np.isfinite(ui)].astype(int)
        if ui.size:
            n_units = max(n_units, int(ui.max()) + 1)
    if n_units == 0:
        n_units = 1

    # --- Global position y-lims across splits (shared position y-axis) ---
    pos_min, pos_max = np.inf, -np.inf
    for sp in splits:
        pv = np.asarray(sp["position_value"])
        pv = pv[np.isfinite(pv)]
        if pv.size:
            pos_min = min(pos_min, float(pv.min()))
            pos_max = max(pos_max, float(pv.max()))
    if not np.isfinite(pos_min) or not np.isfinite(pos_max):
        pos_min, pos_max = 0.0, 1.0

    # --- Figure layout: right column 2x wider; share axes within each column ---
    fig, axes = plt.subplots(
        3, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 2]},
        sharex="col",
        sharey="col",
    )

    # --- Plot each split ---
    for r, (sp, name) in enumerate(zip(splits, split_names)):
        st = np.asarray(sp["spike_timesteps"])
        ui = np.asarray(sp["spike_unit_index"])
        pv = np.asarray(sp["position_value"])
        pt = np.asarray(sp["position_timesteps"])

        # Left: spike raster image
        ax_spk = axes[r, 0]
        raster = _build_spike_image(st, ui, n_units=n_units, t_min=t_min, t_max=t_max, dt=dt)
        ax_spk.imshow(
            raster,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            extent=[t_min, t_min + raster.shape[1] * dt, 0, n_units],
        )
        ax_spk.set_ylabel("Neuron #")
        ax_spk.set_title(f"{name}: spikes")

        # Right: position trace
        ax_pos = axes[r, 1]
        ax_pos.plot(pt, pv, linewidth=1)
        ax_pos.set_title(f"{name}: position")

    # --- Set shared limits ONCE (top row) so all shared axes follow ---
    axes[0, 0].set_xlim(t_min, t_max)
    axes[0, 0].set_ylim(0, n_units)

    axes[0, 1].set_xlim(t_min, t_max)
    axes[0, 1].set_ylim(pos_min, pos_max)

    # Bottom x-labels only
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")

    fig.tight_layout()
    return fig, axes
def plot_interval(data, name, scrambled=False): 
    splits = []
    for phase in ["train", "valid", "test"]:
        interval = getattr(data, f'{phase}_domain')
        d = data.slice(interval.start[0], interval.end[0], reset_origin=False)
        splits.append({
            'spike_timesteps': d.spikes.timestamps, 
            'spike_unit_index': d.spikes.unit_index,
            'position_value': d.position.pos, 
            'position_timesteps': d.position.timestamps,
        })
        print(d.position.timestamps[0:5])


    fig, axes = plot_neural_splits(splits, dt=0.025)

    fig.savefig(f"{name}_neural_splits_scram={scrambled}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.clf()

"""
import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(9,3), dpi=150)
plt.subplots_adjust(wspace = 0.3)
ax = plt.subplot(141)
ax.imshow(spike_by_neuron_use[:1000].T, aspect = 'auto', cmap = 'gray_r')
plt.ylabel('Neuron #')
plt.xlabel('Time [s]')
plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))

ax2 = plt.subplot(142)
ax2.scatter(np.arange(1000), locations_vec[:1000], c = 'gray', s=1)
plt.ylabel('Position [m]')
plt.xlabel('Time [s]')
plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))

ax2 = plt.subplot(143)
ax2.scatter(np.arange(1000), locations_vec[:1000], c = 'gray', s=1)
plt.ylabel('Right')
plt.xlabel('Time [s]')
plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))

ax2 = plt.subplot(144)
ax2.scatter(np.arange(1000), locations_vec[:1000], c = 'gray', s=1)
plt.ylabel('Left')
plt.xlabel('Time [s]')
plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))

plt.show()

# plt.savefig('raw_data.png')
# plt.clf()

"""
if __name__ == "__main__":
    main()
