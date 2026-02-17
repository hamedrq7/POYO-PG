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

    args = parser.parse_args()
    input_dir = args.input_dir # 'D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear/'
    output_dir = args.output_dir # 'D:/Pose/Neuro Code/data/NoveltySessInfoMatFiles/linear_processed/hippo_processed'

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


        train_start_time = start
        train_end_time = end // 2.
        valid_start_time = train_end_time
        valid_end_time = end // (10/7.)
        test_start_time = valid_end_time
        test_end_time = end 
        # print('train_start_time', train_start_time)
        # print('train_end_time', train_end_time)
        # print('valid_start_time', valid_start_time)
        # print('valid_end_time', valid_end_time)
        # print('test_start_time', test_start_time)
        # print('test_end_time', test_end_time)
        data.set_train_domain(Interval(train_start_time, train_end_time))
        data.set_valid_domain(Interval(valid_start_time, valid_end_time))
        data.set_test_domain(Interval(test_start_time, test_end_time))

        
        import os 
        # save data to disk
        path = os.path.join(output_dir, f"{sess_id}.h5")
        with h5py.File(path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


if __name__ == "__main__":
    main()
