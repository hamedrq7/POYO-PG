import math
import logging
from typing import List, Dict, Tuple, Optional, TypeVar, Iterator
from functools import cached_property

import torch
import torch.distributed as dist

from temporaldata import Interval
from torch_brain.data.dataset import DatasetIndex
import random 

def extend_dict_lists_randomly_v2(data_dict):
    """
    More explicit version that ensures minimal repetition.
    Creates full cycles of all elements, shuffling each cycle independently.
    """
    if not data_dict:
        return {}
    
    # Shuffle each list in place
    for key in data_dict:
        random.shuffle(data_dict[key])
    
    # Find the maximum length across all lists
    max_len = max(len(lst) for lst in data_dict.values())
    
    # Extend each list to match max_len
    result_dict = {}
    for key, lst in data_dict.items():
        if len(lst) == max_len:
            result_dict[key] = lst[:]
        else:
            num_to_add = max_len - len(lst)
            lst_len = len(lst)
            
            # Calculate full cycles and remainder
            full_cycles = num_to_add // lst_len
            remainder = num_to_add % lst_len
            
            extended = lst[:]
            
            # Add full cycles (each shuffled independently)
            for _ in range(full_cycles):
                cycle = lst[:]
                random.shuffle(cycle)
                extended.extend(cycle)
            
            # Add remaining elements (shuffled subset)
            if remainder > 0:
                cycle = lst[:]
                random.shuffle(cycle)
                extended.extend(cycle[:remainder])
            
            result_dict[key] = extended
    
    """
    
    # Example code to test the function

    # Create test dictionary with lists of different lengths
    test_dict = {
        'group_a': [1, 2, 3],
        'group_b': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # length 10
        'group_c': [14, 15, 16, 17],
        'group_d': [18, 19]
    }

    print("Original dictionary:")
    for key, lst in test_dict.items():
        print(f"  {key}: {lst} (length: {len(lst)})")

    # Apply the function
    extended_dict = extend_dict_lists_randomly_v2(test_dict.copy())

    print("\nAfter shuffling and random extension:")
    for key, lst in extended_dict.items():
        print(f"  {key}: {lst} (length: {len(lst)})")
        # Check for repetition pattern
        from collections import Counter
        counts = Counter(lst)
        max_repeat = max(counts.values())
        min_repeat = min(counts.values())
        print(f"    Max repetitions of any element: {max_repeat}, Min: {min_repeat}")

    # Verify all lists have the same length
    lengths = [len(lst) for lst in extended_dict.values()]
    max_len = max(lengths) if lengths else 0
    print(f"\nAll lengths: {lengths}")
    print(f"All same length? {all(l == max_len for l in lengths)}")

    # Specific test case you mentioned
    print("\n--- Testing specific case: 16 elements needed, list length 10 ---")
    test_specific = {
        'test': list(range(1, 11))  # list of length 10
    }
    # We'll create another list of length 26 to force 16 additions
    test_specific['target'] = list(range(100, 126))  # length 26

    result_specific = extend_dict_lists_randomly_v2(test_specific)

    print(f"Original list: {result_specific['test'][:10]}")
    print(f"Full extended list: {result_specific['test']}")
    print(f"Length: {len(result_specific['test'])}")

    # Check repetitions
    counts = Counter(result_specific['test'])
    print("Element counts:")
    for elem, count in sorted(counts.items()):
        print(f"  {elem}: appears {count} time(s)")

    max_repeat = max(counts.values())
    min_repeat = min(counts.values())
    print(f"Max repetitions: {max_repeat}, Min repetitions: {min_repeat}")
    print(f"Expected: each element appears at least once, no more than twice")
    """
    return result_dict

def interleave_round_robin_with_keys(dict_of_lists):
    """
    Same as above but also returns the source keys for each element.
    
    Returns:
        Tuple of (interleaved_list, source_keys_list)
    """
    # Sort keys to ensure deterministic order
    keys = sorted(dict_of_lists.keys())
    lists = [dict_of_lists[key] for key in keys]
    
    result = []
    sources = []
    max_len = max(len(lst) for lst in lists) if lists else 0
    
    for i in range(max_len):
        for idx, lst in enumerate(lists):
            if i < len(lst):
                result.append(lst[i])
                sources.append(keys[idx])
    
    return result, sources

def interleave_round_robin_with_chunk_and_keys(dict_of_lists, k=1):
    """
    Same as above but also returns the source keys for each element.
    """
    keys = sorted(dict_of_lists.keys())
    lists = [dict_of_lists[key] for key in keys]
    
    result = []
    sources = []
    
    max_len = max(len(lst) for lst in lists) if lists else 0
    num_chunks = (max_len + k - 1) // k
    
    for chunk_idx in range(num_chunks):
        for idx, lst in enumerate(lists):
            start = chunk_idx * k
            end = min(start + k, len(lst))
            
            # Add the chunk of elements
            for item in lst[start:end]:
                result.append(item)
                sources.append(keys[idx])
    
    return result, sources


class BalancedRandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`sampling_intervals` parameter. :obj:`sampling_intervals` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        batch_size: int, 
        subject_ids: List, 
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short
        self.batch_size = batch_size
        self.subject_ids = subject_ids 

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len

    def __iter__(self):
        # print('CALLED')
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        session_indices = {}
        for session_name, sampling_intervals in self.sampling_intervals.items():
            indices = []
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                # sample a random offset
                left_offset = (
                    torch.rand(1, generator=self.generator).item() * self.window_length
                )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(
                        start + left_offset,
                        end,
                        self.window_length,
                        dtype=torch.float64,
                    )
                    if t + self.window_length <= end
                ]

                if len(indices_) > 0:
                    indices.extend(indices_)
                    right_offset = end - indices[-1].end
                else:
                    right_offset = end - start - left_offset

                # if there is one sample worth of data, add it
                # this ensures that the number of samples is always consistent
                if right_offset + left_offset >= self.window_length:
                    if right_offset > left_offset:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )
                    else:
                        indices.append(
                            DatasetIndex(
                                session_name, start, start + self.window_length
                            )
                        )
            # 
            random.shuffle(indices)
            session_indices[session_name] = indices


            # print(session_name, len(indices))

        session_indices = extend_dict_lists_randomly_v2(session_indices)

        # for kkk in session_indices.keys():
        #     print(kkk, len(session_indices[kkk]))
        
        # all_indicies, sess_source = interleave_round_robin_with_keys(session_indices)
        k = self.batch_size // len(session_indices.keys())
        assert self.batch_size % len(session_indices.keys()) == 0, 'for now pass a batch size thats divisible by number of subjects. fix this later'
        
        all_indicies, sess_source = interleave_round_robin_with_chunk_and_keys(session_indices, k = k)

        for index in all_indicies:
            yield index
        
        # for idx in torch.randperm(len(indices), generator=self.generator):
        #     yield indices[idx]

class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`sampling_intervals` parameter. :obj:`sampling_intervals` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len

    def __iter__(self):
        # print('CALLED')
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        indices = []
        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                # sample a random offset
                left_offset = (
                    torch.rand(1, generator=self.generator).item() * self.window_length
                )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(
                        start + left_offset,
                        end,
                        self.window_length,
                        dtype=torch.float64,
                    )
                    if t + self.window_length <= end
                ]

                if len(indices_) > 0:
                    indices.extend(indices_)
                    right_offset = end - indices[-1].end
                else:
                    right_offset = end - start - left_offset

                # if there is one sample worth of data, add it
                # this ensures that the number of samples is always consistent
                if right_offset + left_offset >= self.window_length:
                    if right_offset > left_offset:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )
                    else:
                        indices.append(
                            DatasetIndex(
                                session_name, start, start + self.window_length
                            )
                        )

        # print(len(indices))
        # print(indices[0:3])
        # shuffle
        for idx in torch.randperm(len(indices), generator=self.generator):
            yield indices[idx]
        # for index in indices:
        #     yield index

class SequentialFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows sequentially, always in the same order. The
    sampling intervals are defined in the :obj:`sampling_intervals` parameter.
    :obj:`sampling_intervals` is a dictionary where the keys are the session ids and the
    values are lists of tuples representing the start and end of the intervals
    from which to sample.

    If the length of a sequence is not evenly divisible by the step, the last
    window will be added with an overlap with the previous window. This is to ensure
    that the entire sequence is covered.

    Args:
        sampling_intervals (Dict[str, List[Tuple[float, float]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        step (float, optional): Step size between windows. If None, it
            defaults to ``window_length``.
        drop_short (bool, optional): Whether to drop windows smaller than ``window_length``.
            Defaults to False.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        step: Optional[float] = None,
        drop_short=False,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.step = step or window_length
        self.drop_short = drop_short

        assert self.step > 0, "Step must be greater than 0."

    # we cache the indices since they are deterministic
    @cached_property
    def _indices(self) -> List[DatasetIndex]:
        indices = []
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(start, end, self.step, dtype=torch.float64)
                    if t + self.window_length <= end
                ]

                indices.extend(indices_)

                # we need to make sure that the entire interval is covered
                if indices_[-1].end < end:
                    indices.append(
                        DatasetIndex(session_name, end - self.window_length, end)
                    )

        if self.drop_short and total_short_dropped > 0:
            num_samples = len(indices)
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")

        return indices

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        yield from self._indices


class TrialSampler(torch.utils.data.Sampler):
    r"""Randomly samples a single trial interval from the given intervals.

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        shuffle (bool, optional): Whether to shuffle the indices. Defaults to False.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, List[Tuple[float, float]]],
        generator: Optional[torch.Generator] = None,
        shuffle: bool = False,
    ):
        self.sampling_intervals = sampling_intervals
        self.generator = generator
        self.shuffle = shuffle

    def __len__(self):
        return sum(len(intervals) for intervals in self.sampling_intervals.values())

    def __iter__(self):
        # Flatten the intervals from all sessions into a single list
        all_intervals = [
            (session_id, start, end)
            for session_id, intervals in self.sampling_intervals.items()
            for start, end in zip(intervals.start, intervals.end)
        ]

        indices = [
            DatasetIndex(session_id, start, end)
            for session_id, start, end in all_intervals
        ]

        if self.shuffle:
            # Yield a single DatasetIndex representing the selected interval
            for idx in torch.randperm(len(indices), generator=self.generator):
                yield indices[idx]
        else:
            yield from indices


class DistributedEvaluationSamplerWrapper(torch.utils.data.Sampler):
    r"""Wraps a sampler to be used in a distributed evaluation setting. Unlike the standard
    distributed samplers from PyTorch and PyTorch Lightning which ensure equal samples per rank
    by potentially dropping samples, this sampler preserves all samples by distributing them
    across ranks without dropping any, which is important to guarantee that evaluation is done
    on the complete dataset.

    .. warning::
        This wrapper assumes that there is no communication between ranks except at the
        begining or end of the evaluation, so it is only suitable for standard evaluation.
        This is because some ranks might end up performing more steps than others.

    Args:
        sampler (torch.utils.data.Sampler): The original sampler to wrap.
        num_replicas (int): Number of processes participating in the distributed
            evaluation.
        rank (int): Rank of the current process.

    Example ::

        >>> from torch_brain.data.sampler import SequentialFixedWindowSampler, DistributedEvaluationSamplerWrapper

        >>> sampling_intervals = {
        ...     "session_1": Interval(0, 100),
        ...     "session_2": Interval(0, 100),
        ... }

        >>> sampler = SequentialFixedWindowSampler(sampling_intervals=sampling_intervals, window_length=10)
        >>> dist_sampler = DistributedEvaluationSamplerWrapper(sampler)

    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def set_params(self, num_replicas, rank):
        logging.info(
            f"Setting distributed sampler params: "
            f"num_replicas={num_replicas}, rank={rank}"
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def _check_params(self):
        return (self.num_replicas is not None) and (self.rank is not None)

    def rank_len(self):
        r"""Returns the number of samples assigned to the current process."""
        total_len = len(self.sampler)
        evenly_split = total_len // self.num_replicas
        extra = int((total_len % self.num_replicas) < self.rank)
        return evenly_split + extra

    def __len__(self):
        r"""Returns the number of samples assigned to the current process if
        the rank and num_replicas are set. Otherwise, returns the total number
        of samples in the original sampler.
        """
        if not self._check_params():
            return len(self.sampler)
        else:
            return self.rank_len()

    def __iter__(self):
        assert (
            self._check_params()
        ), "Rank and num_replicas must be set before using the distributed sampler."
        indices = list(self.sampler)
        indices = indices[self.rank : len(indices) : self.num_replicas]
        return iter(indices)


class DistributedStitchingFixedWindowSampler(torch.utils.data.DistributedSampler):
    r"""A sampler designed specifically for evaluation that enables sliding window
    inference with prediction stitching across distributed processes.

    This sampler divides sequences into overlapping windows and distributes them across
    processes for parallel inference, it keeps windows that need to be stitched together
    on the same rank, to allow stitching on that same rank without communication.

    Additionally, it will keep track of the windows that need to be stitched together to
    allow for stitching as soon as all windows from the same contiguous sequence are
    available. This information can be passed to the stitcher which can stitch and compute
    a metric for the sequence as soon as all windows from that sequence are available,
    allowing it to free up memory quickly.

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset. Each interval is defined by a start and end time.
        window_length (float): Length of the sliding window.
        step (Optional[float], optional): Step size between windows. If None, defaults
            to window_length. Smaller steps create more overlap between windows.
        batch_size (int): Number of windows to process in each batch.
        num_replicas (Optional[int], optional): Number of processes participating in
            distributed inference. If None, will be set using torch.distributed.
        rank (Optional[int], optional): Rank of the current process. If None, will be
            set using torch.distributed.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        step: Optional[float] = None,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.step = step or window_length
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if self.step <= 0:
            raise ValueError("Step must be greater than 0.")
        if self.step > self.window_length:
            raise ValueError("Step must be less than window length.")

        # Generate indices for this rank
        self.indices, self.sequence_index = self._generate_indices()
        self.num_samples = len(self.indices)

    def _generate_indices(self) -> List[DatasetIndex]:
        """Generate indices for this rank, balancing the workload across ranks based on
        the number of windows in each interval."""
        # first, we will compute the number of contiguous windows across all intervals
        all_intervals = []
        interval_sizes = []
        for session_name, intervals in self.sampling_intervals.items():
            for start, end in zip(intervals.start, intervals.end):
                if end - start >= self.window_length:
                    # calculate number of windows in this interval
                    num_windows = (
                        int((end - start - self.window_length + 1e-9) / self.step) + 1
                    )
                    # print('start, end, num_windows', start, end, num_windows)
                    if num_windows > 0:
                        interval_sizes.append(num_windows)
                        all_intervals.append((session_name, start, end))

        # sort intervals by size in descending order for better load balancing
        sorted_indices = torch.argsort(torch.tensor(interval_sizes), descending=True)
        all_intervals = [all_intervals[i] for i in sorted_indices]
        interval_sizes = [interval_sizes[i] for i in sorted_indices]

        # track total windows per rank for load balancing
        rank_sizes = [0] * self.num_replicas

        # assign intervals to ranks to minimize imbalance
        indices_list = []
        for session_name, start, end in all_intervals:
            # assign to rank with fewest windows
            target_rank = min(range(self.num_replicas), key=lambda r: rank_sizes[r])

            indices = []
            # generate all windows for this interval
            for t in torch.arange(
                start,
                end - self.window_length + 1e-9,
                self.step,
                dtype=torch.float64,
            ):
                t = t.item()
                indices.append(DatasetIndex(session_name, t, t + self.window_length))

            # add final window if needed
            last_start = indices[-1].start if indices else start
            if last_start + self.window_length < end:
                indices.append(
                    DatasetIndex(session_name, end - self.window_length, end)
                )

            if target_rank == self.rank:
                # only add indices to this rank
                indices_list.append(indices)

            rank_sizes[target_rank] += len(indices)

        # print('indicies', len(indices), type(indices[0]), indices[0:10])
        # shuffle indices for this rank
        indices_list = [indices_list[i] for i in torch.randperm(len(indices_list))]
        indices = [item for sublist in indices_list for item in sublist]
        sequence_index = torch.tensor(
            [i for i, sublist in enumerate(indices_list) for _ in sublist]
        )
        # print('sequence_index', sequence_index.shape, sequence_index[0:3])

        return indices, sequence_index

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number. Not strictly necessary for sequential sampler
        but included for API compatibility."""
        self.epoch = epoch
