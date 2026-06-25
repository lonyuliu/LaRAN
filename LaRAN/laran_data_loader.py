"""Dataset utilities for LaRAN network-level association evaluation."""

import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def association_collate_fn(batch):
    """Pad variable-length measurements and tracks into a batch."""
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return {}

    batch_size = len(batch)
    num_frames = len(batch[0]["measurements"])
    measurement_dim = (
        batch[0]["measurements"][0].shape[1]
        if batch[0]["measurements"]
        else 4
    )
    history_len = batch[0]["gt_history"].shape[1]
    state_dim = batch[0]["gt_history"].shape[2]

    max_points = max(
        max(frame.shape[0] for frame in sample["measurements"])
        for sample in batch
    )
    max_tracks = max(sample["gt_states"].shape[1] for sample in batch)

    padded_measurements = torch.zeros(
        batch_size, num_frames, max_points, measurement_dim
    )
    measurement_mask = torch.zeros(
        batch_size, num_frames, max_points, dtype=torch.bool
    )
    gt_states = torch.zeros(
        batch_size, num_frames, max_tracks, state_dim
    )
    gt_visibility = torch.zeros(
        batch_size, num_frames, max_tracks, dtype=torch.bool
    )
    gt_history = torch.zeros(
        batch_size, max_tracks, history_len, state_dim
    )
    gt_attention_mask = torch.zeros(
        batch_size, num_frames, max_tracks, max_points
    )

    for batch_index, sample in enumerate(batch):
        num_tracks = sample["gt_states"].shape[1]
        gt_states[batch_index, :, :num_tracks] = sample["gt_states"]
        gt_visibility[batch_index, :, :num_tracks] = sample["gt_visibility"]
        gt_history[batch_index, :num_tracks] = sample["gt_history"]

        for frame_index, frame_measurements in enumerate(sample["measurements"]):
            num_points = frame_measurements.shape[0]
            if num_points == 0:
                continue

            padded_measurements[
                batch_index, frame_index, :num_points
            ] = frame_measurements
            measurement_mask[batch_index, frame_index, :num_points] = True

            attention_mask = sample["gt_attn_mask"][frame_index]
            if attention_mask.shape[1] > 0:
                gt_attention_mask[
                    batch_index, frame_index, :num_tracks, :num_points
                ] = attention_mask

    return {
        "measurements": padded_measurements,
        "meas_mask": measurement_mask,
        "gt_states": gt_states,
        "gt_visibility": gt_visibility,
        "gt_history": gt_history,
        "gt_attn_mask": gt_attention_mask,
    }


class AssociationDataset(Dataset):
    """Read fixed-length multi-object association clips from an HDF5 file."""

    feature_indices = (0, 1, 2, 3)
    state_indices = (1, 2, 3, 4, 5, 6)

    def __init__(
        self,
        h5_path,
        clip_samples,
        normalization_params,
        tracklet_len=10,
        history_len=8,
    ):
        self.h5_path = h5_path
        self.clip_samples = clip_samples
        self.tracklet_len = tracklet_len
        self.history_len = history_len
        self.h5_file = None

        epsilon = 1e-7
        self.meas_mean = self._resolve_meas_norm_param(
            normalization_params["meas_mean"], "meas_mean"
        )
        self.meas_std = (
            self._resolve_meas_norm_param(
                normalization_params["meas_std"], "meas_std"
            )
            + epsilon
        )
        self.state_mean = torch.as_tensor(
            normalization_params["state_mean"], dtype=torch.float32
        )
        self.state_std = (
            torch.as_tensor(
                normalization_params["state_std"], dtype=torch.float32
            )
            + epsilon
        )

    def _resolve_meas_norm_param(self, value, param_name):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim != 1:
            raise ValueError(
                f"{param_name} must be one-dimensional, got {tuple(tensor.shape)}"
            )

        expected_dim = len(self.feature_indices)
        if tensor.numel() == expected_dim:
            return tensor

        max_index = max(self.feature_indices)
        if tensor.numel() > max_index:
            return tensor[list(self.feature_indices)]

        raise ValueError(
            f"{param_name} length {tensor.numel()} is incompatible with "
            f"feature indices {self.feature_indices}"
        )

    def __len__(self):
        return len(self.clip_samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def close(self):
        """Close the lazily opened HDF5 handle."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not raise during interpreter shutdown.
            pass

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        sample_info = self.clip_samples[index]
        scene_index = sample_info["scene_idx"]
        start_frame = sample_info["start_frame"]
        end_frame = start_frame + self.tracklet_len
        scene_name = f"scene_{scene_index:05d}"

        try:
            scene_group = self.h5_file[scene_name]
            max_frames = scene_group["ground_truth"].shape[2]
            read_end_frame = min(end_frame, max_frames)
            read_start_frame = min(start_frame, read_end_frame)

            ground_truth = scene_group["ground_truth"][
                :, :, read_start_frame:read_end_frame
            ]
            measurements = scene_group["measurements"][
                :, :, read_start_frame:read_end_frame
            ]
            history_start = max(0, read_start_frame - self.history_len)
            ground_truth_history = scene_group["ground_truth"][
                :, :, history_start:read_start_frame
            ]
        except KeyError:
            return None

        gt_ids = ground_truth[0]
        track_ids = np.unique(gt_ids)
        track_ids = track_ids[track_ids > -1]
        num_tracks = len(track_ids)
        if num_tracks == 0:
            return None

        gt_states_all_frames = ground_truth[
            self.state_indices, :, :
        ].transpose(1, 2, 0)
        clip_gt_states = torch.zeros(
            self.tracklet_len, num_tracks, len(self.state_indices)
        )
        clip_gt_visibility = torch.zeros(
            self.tracklet_len, num_tracks, dtype=torch.bool
        )
        clip_gt_history = torch.zeros(
            num_tracks, self.history_len, len(self.state_indices)
        )

        history_ids = ground_truth_history[0]
        history_states = ground_truth_history[
            self.state_indices, :, :
        ].transpose(1, 2, 0)

        for track_index, track_id in enumerate(track_ids):
            track_history = []
            for history_frame in range(history_ids.shape[1]):
                row_indices = np.where(
                    history_ids[:, history_frame] == track_id
                )[0]
                if row_indices.size > 0:
                    track_history.append(
                        history_states[row_indices[0], history_frame]
                    )

            if track_history:
                history_array = np.asarray(track_history)
                history_array = np.pad(
                    history_array,
                    (
                        (self.history_len - len(track_history), 0),
                        (0, 0),
                    ),
                    mode="edge",
                )
            else:
                history_array = np.zeros(
                    (self.history_len, len(self.state_indices)),
                    dtype=np.float32,
                )

            history_tensor = torch.from_numpy(history_array).float()
            clip_gt_history[track_index] = (
                history_tensor - self.state_mean
            ) / self.state_std

        track_id_to_index = {
            int(track_id): index for index, track_id in enumerate(track_ids)
        }
        all_frame_measurements = []
        gt_attention_masks = []

        for frame_offset in range(self.tracklet_len):
            if frame_offset >= gt_ids.shape[1]:
                all_frame_measurements.append(
                    torch.empty(0, len(self.feature_indices))
                )
                gt_attention_masks.append(torch.empty(num_tracks, 0))
                continue

            current_frame_ids = gt_ids[:, frame_offset]
            for track_index, track_id in enumerate(track_ids):
                row_indices = np.where(current_frame_ids == track_id)[0]
                if row_indices.size == 0:
                    continue

                state = gt_states_all_frames[
                    row_indices[0], frame_offset
                ]
                if np.isfinite(state).all():
                    clip_gt_states[frame_offset, track_index] = (
                        torch.from_numpy(state).float()
                    )
                    clip_gt_visibility[frame_offset, track_index] = True

            frame_measurements = measurements[:, :, frame_offset].T
            valid_mask = (
                np.isfinite(frame_measurements).all(axis=1)
                & np.any(frame_measurements, axis=1)
            )
            frame_measurements = frame_measurements[valid_mask]

            if frame_measurements.shape[0] == 0:
                all_frame_measurements.append(
                    torch.empty(0, len(self.feature_indices))
                )
                gt_attention_masks.append(torch.empty(num_tracks, 0))
                continue

            features = frame_measurements[:, self.feature_indices]
            feature_tensor = (
                torch.as_tensor(features, dtype=torch.float32)
                - self.meas_mean
            ) / self.meas_std
            all_frame_measurements.append(feature_tensor)

            attention_mask = torch.zeros(
                num_tracks, frame_measurements.shape[0]
            )
            for point_rank, raw_track_id in enumerate(frame_measurements[:, 7]):
                clean_label = (
                    int(raw_track_id) if raw_track_id > -1 else None
                )
                track_index = track_id_to_index.get(clean_label)
                if track_index is not None:
                    attention_mask[track_index, point_rank] = 1.0
            gt_attention_masks.append(attention_mask)

        clip_gt_states = (
            clip_gt_states - self.state_mean
        ) / self.state_std

        return {
            "measurements": all_frame_measurements,
            "gt_states": clip_gt_states,
            "gt_visibility": clip_gt_visibility,
            "gt_history": clip_gt_history,
            "gt_attn_mask": gt_attention_masks,
        }


def precompute_tracklets_for_association(
    h5_path, tracklet_len=10, seed=2025
):
    """Build deterministic sliding-window clip descriptors."""
    if tracklet_len <= 0:
        raise ValueError("tracklet_len must be positive")

    clip_samples = []
    try:
        with h5py.File(h5_path, "r") as h5_file:
            scene_keys = sorted(h5_file.keys())
            for scene_key in scene_keys:
                scene_index = int(scene_key.split("_")[-1])
                ground_truth = h5_file[scene_key]["ground_truth"][0]
                max_frame = ground_truth.shape[1]
                for start_frame in range(
                    0, max_frame - tracklet_len + 1, 2
                ):
                    clip_samples.append(
                        {
                            "scene_idx": scene_index,
                            "start_frame": start_frame,
                        }
                    )
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to scan HDF5 dataset: {h5_path}"
        ) from exc

    random.Random(seed).shuffle(clip_samples)
    return clip_samples
