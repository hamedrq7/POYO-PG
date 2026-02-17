# D:\Pose\Neuro Code\poyo-reference\hippocampus.py
"""
Load hippocampus dataset from CEBRA's preprocessed dataset and format for POYO,
converting to event-based spikes and adding necessary attributes for transforms.
Splits data into 70% train, 25% test, and 5% valid.
Ensures target dimensions match model output (2 features).
"""

import argparse
import datetime
import logging
import pathlib
import os
from typing import List, Tuple

import torch
import numpy as np
from pathlib import Path

import cebra.datasets

from poyo.data import (
    Data,
    IrregularTimeSeries,
    Interval,
    DatasetBuilder,
    ArrayDict,
)
from poyo.taxonomy import Task, RecordingTech, Sex, Species


raw_folder = "./raw"
if not os.path.exists(raw_folder):
    os.makedirs(raw_folder)
    print(f"Created folder: {raw_folder}")
else:
    print(f"Folder {raw_folder} already exists.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_hippocampus_data() -> Tuple[List[str], List]:
    """Load hippocampus data using CEBRA's dataset classes."""
    hippocampus_a = cebra.datasets.init("rat-hippocampus-single-achilles")
    hippocampus_b = cebra.datasets.init("rat-hippocampus-single-buddy")
    hippocampus_c = cebra.datasets.init("rat-hippocampus-single-cicero")
    hippocampus_g = cebra.datasets.init("rat-hippocampus-single-gatsby")

    names = ["achilles", "buddy", "cicero", "gatsby"]
    hippocampus = [hippocampus_a, hippocampus_b, hippocampus_c, hippocampus_g]

    return names, hippocampus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./", help="Directory to save outputs."
    )
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="If set, will save a POYO Dataset structure. Otherwise, saves a dict of PT tensors.",
    )
    args = parser.parse_args()

    # Initialize dataset builder
    db = DatasetBuilder(
        raw_folder_path="./raw",  # Not used since we're using CEBRA's processed data
        processed_folder_path=str(pathlib.Path(args.output_dir) / "hippocampus"),
        experiment_name="hippocampus",
        origin_version="1.0",
        derived_version="1.0.0",
        source="https://crcns.org/data-sets/hc/hc-11",
        description="Hippocampal recordings (event-based) for POYO with unit_index and subtask_index",
        metadata_version="0.0.2",
    )

    # Load data from CEBRA
    names, hippocampus_data = load_hippocampus_data()

    # Process each rat's data
    for rat_idx, (name, data) in enumerate(zip(names, hippocampus_data)):
        logger.info(f"Processing rat: {name}")

        # Original data is binned rates: shape [Time x Units]
        neural_activity = data.neural.numpy()
        n_samples, n_units = neural_activity.shape

        # Timestamps at 30 Hz
        timestamps = np.arange(n_samples) / 30.0

        # Convert from (Time x Units) to event-based:
        # We'll do a simple approach: if neural_activity[t, u] > 0, treat that as a single spike at time t
        # (If you have multiple spikes in one bin, adapt accordingly).
        spike_times_list = []
        unit_idx_list = []
        for unit_i in range(n_units):
            spike_mask = neural_activity[:, unit_i] > 0
            # Indices of all time bins where that unit fired
            fired_indices = np.where(spike_mask)[0]

            # Extend times & unit indices
            spike_times_list.extend(timestamps[fired_indices])
            unit_idx_list.extend([unit_i] * len(fired_indices))

        # Convert to arrays and sort by time
        spike_times_arr = np.array(spike_times_list)
        unit_idx_arr = np.array(unit_idx_list)
        sort_idx = np.argsort(spike_times_arr)
        spike_times_arr = spike_times_arr[sort_idx]
        unit_idx_arr = unit_idx_arr[sort_idx]

        # Build an event-based spike IrregularTimeSeries
        spikes_ts = IrregularTimeSeries(
            timestamps=spike_times_arr,
            unit_index=unit_idx_arr,  # crucial for unit_dropout
            domain="auto",
        )

        # For demonstration, treat data.continuous_index as "behavior"
        behavior_array = data.continuous_index.numpy()

        # Log the shape before slicing
        logger.info(f"Original behavior_array shape: {behavior_array.shape}")

        # Define subtask_index
        # If your dataset has actual subtask information, extract and assign it here.
        # For this example, we'll create a placeholder where all subtasks are 0.
        # Replace this with actual subtask logic as needed.
        subtask_index = np.zeros(n_samples, dtype=int)

        # Slice behavior_array to have only the first 2 features
        if behavior_array.shape[1] < 2:
            logger.error(
                f"Behavior array has less than 2 features: {behavior_array.shape}"
            )
            # Handle the error as needed. Here, we'll skip processing this rat.
            continue

        sliced_behavior_array = behavior_array[:, :2]
        logger.info(
            f"Sliced behavior_array shape for target: {sliced_behavior_array.shape}"
        )

        # Create behavior IrregularTimeSeries with sliced behavior_array and subtask_index
        behavior_ts = IrregularTimeSeries(
            timestamps=timestamps,
            vel=sliced_behavior_array,  # Use sliced behavior_array
            subtask_index=subtask_index,  # Added to resolve AttributeError
            domain="auto",
        )

        epsilon = 1e-6

        total_duration = timestamps[-1]
        train_end_time = total_duration * 0.7
        test_end_time = total_duration * 0.95

        train_interval = Interval(
            start=np.array([timestamps[0]]),
            end=np.array([train_end_time - epsilon]),
            is_valid=np.array([True]),
        )
        test_interval = Interval(
            start=np.array([train_end_time]),
            end=np.array([test_end_time - epsilon]),
            is_valid=np.array([True]),
        )
        valid_interval = Interval(
            start=np.array([test_end_time]),
            end=np.array([timestamps[-1]]),
            is_valid=np.array([True]),
        )

        # Verify intervals contain spikes
        def interval_has_spikes(interval, spike_times):
            return np.any(
                (spike_times >= interval.start[0]) & (spike_times <= interval.end[0])
            )

        if not interval_has_spikes(test_interval, spike_times_arr):
            raise ValueError(f"Test interval for {name} contains no spikes")

            # -----------------------------
        # 2) If user wants a POYO Dataset
        # -----------------------------
        split_idx = int(0.7 * len(timestamps))
        if 1 == 1:
            with db.new_session() as session:
                # Create a Data object with event-based spikes and behavior

                pos_diff = np.abs(np.diff(data.continuous_index[:, 0].numpy()))
                threshold = np.percentile(
                    pos_diff, 95
                )  # Use 95th percentile for threshold
                trial_boundaries = np.where(pos_diff > threshold)[0]

                trial_starts = timestamps[trial_boundaries[:-1]]
                trial_ends = timestamps[trial_boundaries[1:]]

                # Filter out very short or long trials
                trial_durations = trial_ends - trial_starts
                valid_trials = (trial_durations > 1.0) & (trial_durations < 30.0)

                trials = Interval(
                    start=trial_starts[valid_trials],
                    end=trial_ends[valid_trials],
                    is_valid=np.ones(len(trial_starts[valid_trials]), dtype=bool),
                )

                # Create unit IDs for the sortset
                n_units = data.neural.shape[1]
                unit_ids = [f"unit_{i}" for i in range(n_units)]

                # Create separate trials for train and test data
                train_trials = Interval(
                    start=np.array([timestamps[0]]),
                    end=np.array([timestamps[split_idx - 1]]),
                    is_valid=np.array([True]),
                )

                test_trials = Interval(
                    start=np.array([timestamps[split_idx]]),
                    end=np.array([timestamps[-1]]),
                    is_valid=np.array([True]),
                )

                # Combine trials python Hippo_dataset_V2.py --save_dataset
                trials = Interval(
                    start=np.concatenate([train_trials.start, test_trials.start]),
                    end=np.concatenate([train_trials.end, test_trials.end]),
                    is_valid=np.concatenate(
                        [train_trials.is_valid, test_trials.is_valid]
                    ),
                )

                data_obj = Data(
                    spikes=spikes_ts,
                    cursor=behavior_ts,
                    trials=trials,  # no all-encompassing trial
                    domain=behavior_ts.domain,
                )

                # Attach the same unit IDs from the sortset to data_obj
                data_obj.units = ArrayDict(
                    id=np.array([f"unit_{i}" for i in range(n_units)]),
                    brain_area=np.array(["hippocampus"] * n_units),
                )

                # Register subject, session, sortset
                session.register_subject(
                    id=f"rat_{name}",
                    species=Species.RAT,
                    sex=Sex.UNKNOWN,
                    age=0.0,
                )
                session.register_session(
                    id=f"hippocampus_single_{name}",
                    recording_date=datetime.datetime.now(),
                    task=Task.NAVIGATION,
                )
                session.register_sortset(
                    id=f"hippocampus_single_{name}_units",
                    units=data_obj.units,
                    recording_tech=[RecordingTech.EXTRACELLULAR] * n_units,
                )

                # Register the Data object
                session.register_data(data_obj)

                session.register_split("train", train_trials)
                session.register_split("test", test_trials)

                # Define the splits (non-overlapping intervals)
                # session.register_split("train", train_interval)
                # session.register_split("test", test_interval)
                # session.register_split("valid", valid_interval)

                # Save to disk
                session.save_to_disk()

            # After all rats, finalize
            db.finish()

        # -----------------------------
        # 3) Otherwise, save a dictionary
        # -----------------------------
        else:
            # If you still want to produce a dictionary for other usage,
            # store spike_times and unit_index
            spike_times_pt = torch.from_numpy(spike_times_arr).float()
            unit_indices_pt = torch.from_numpy(unit_idx_arr).long()

            # Prepare a dict
            n_latents = 100  # example dimension for "latent"

            # Ensure behavior_array has at least 2 features before slicing
            if sliced_behavior_array.shape[1] != 2:
                logger.error(
                    f"Sliced behavior array does not have 2 features: {sliced_behavior_array.shape}"
                )
                # Handle the error as needed. Here, we'll skip saving this rat.
                continue

            poyo_inputs = {
                # Event-based spikes
                "spike_unit_index": unit_indices_pt,
                "spike_timestamps": spike_times_pt,
                "spike_type": torch.zeros(len(spike_times_pt), dtype=torch.long),
                "input_seqlen": n_samples,
                # Latent sequence
                "latent_index": torch.arange(n_latents),
                "latent_timestamps": torch.linspace(
                    timestamps[0], timestamps[-1], n_latents
                ),
                "latent_seqlen": n_latents,
                # Output sequence
                "session_index": torch.tensor([rat_idx]).repeat(n_samples),
                "output_timestamps": torch.from_numpy(timestamps).float(),
                "output_seqlen": n_samples,
                "output_batch_index": torch.arange(n_samples),
                "output_values": {
                    # Ensure that 'continuous' has only 2 features
                    "continuous": torch.from_numpy(sliced_behavior_array).float()
                },
                "output_weights": torch.ones(n_samples),
                # Metadata
                "n_neurons": n_units,
                "sequence_length": n_samples,
                "split_idx_train": train_end_time,
                "split_idx_test": test_end_time,
                "rat_name": name,
            }

            # Log the shape of the target
            logger.info(
                f"Target 'continuous' shape: {poyo_inputs['output_values']['continuous'].shape}"
            )

            save_path = (
                pathlib.Path(args.output_dir) / "hippocampus" / f"{name}_poyo_inputs.pt"
            )
            os.makedirs(save_path.parent, exist_ok=True)
            torch.save(poyo_inputs, save_path)
            logger.info(f"Saved poyo_inputs for {name} to {save_path}")


if __name__ == "__main__":
    main()
