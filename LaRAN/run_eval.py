"""Evaluate released LaRAN checkpoints on the three public validation tasks."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from laran_data_loader import (
    AssociationDataset,
    association_collate_fn,
    precompute_tracklets_for_association,
)
from laran_model import build_deployment_model


REPO_ROOT = Path(__file__).resolve().parents[1]

PRECOMPUTED_CACHE_PATH = "precomputed_cache.pth"

TASKS = [
    ("Task1", "Task1.h5"),
    ("Task2", "Task2.h5"),
    ("Task3", "Task3.h5"),
]

CHECKPOINTS = [
    ("LaRAN-Clean", "LaRAN-Clean.pth"),
    ("LaRAN-Noise50", "LaRAN-Noise50.pth"),
]

MODEL_PARAMS = {
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "state_dim": 6,
    "meas_dim": 4,
    "rnn_layers": 2,
    "num_interactions": 2,
    "transformer_layers": 2,
    "num_newborns": 32,
}

TRACKLET_LEN = 10
HISTORY_LEN = 8
BATCH_SIZE = 16
NUM_WORKERS = 0
SCORE_THRESHOLD = 0.5


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_normalization_params(cache_path: str | Path) -> dict:
    cache = torch.load(resolve_path(cache_path), map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or "norm_params" not in cache:
        raise ValueError(f"Cache file does not contain norm_params: {cache_path}")
    return cache["norm_params"]


def calculate_physics_prediction(
    track_histories: torch.Tensor,
    norm_params: dict,
    device: torch.device,
    active_mask_prev: torch.Tensor | None = None,
) -> torch.Tensor:
    if track_histories.shape[1] < 2:
        return track_histories[:, -1, :]

    state_mean = torch.as_tensor(norm_params["state_mean"], device=device).float()
    state_std = torch.as_tensor(norm_params["state_std"], device=device).float() + 1e-7

    current_state = track_histories[:, -1, :] * state_std + state_mean
    previous_state = track_histories[:, -2, :] * state_std + state_mean

    current_position = current_state[:, :3]
    current_velocity = current_state[:, 3:]
    previous_velocity = previous_state[:, 3:]

    acceleration = current_velocity - previous_velocity
    if active_mask_prev is not None:
        acceleration = acceleration * active_mask_prev.unsqueeze(-1).float()

    predicted_position = current_position + current_velocity + 0.5 * acceleration
    predicted_velocity = current_velocity + acceleration
    predicted_state = torch.cat([predicted_position, predicted_velocity], dim=1)
    return (predicted_state - state_mean) / state_std


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1)
    positive_count = labels.sum()
    if positive_count.item() <= 0:
        return None

    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]
    true_positive_count = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, sorted_labels.numel() + 1, device=scores.device, dtype=scores.dtype)
    precision_at_k = true_positive_count / ranks
    return ((precision_at_k * sorted_labels).sum() / positive_count).item()


def init_metric_accumulator() -> dict[str, float]:
    return {
        "f1_sum": 0.0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "tsr_sum": 0.0,
        "step_count": 0.0,
        "auprc_sum": 0.0,
        "auprc_count": 0.0,
    }


def update_metrics(
    accumulator: dict[str, float],
    logits: torch.Tensor,
    labels: torch.Tensor,
    point_valid_mask: torch.Tensor,
    track_valid_mask: torch.Tensor,
    threshold: float = SCORE_THRESHOLD,
) -> None:
    if logits.numel() == 0:
        return

    edge_valid_mask = point_valid_mask.bool() & track_valid_mask.bool().unsqueeze(-1)
    active_rows = track_valid_mask.bool()
    if not active_rows.any():
        return

    logits = logits[active_rows]
    labels = labels[active_rows]
    valid_mask = edge_valid_mask[active_rows]

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).float()
    valid_mask_float = valid_mask.float()

    true_positive = (predictions * labels * valid_mask_float).sum(dim=1)
    false_positive = (predictions * (1.0 - labels) * valid_mask_float).sum(dim=1)
    false_negative = ((1.0 - predictions) * labels * valid_mask_float).sum(dim=1)

    precision = true_positive / (true_positive + false_positive + 1e-7)
    recall = true_positive / (true_positive + false_negative + 1e-7)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)

    valid_points_per_row = valid_mask.sum(dim=1)
    exact_match = ((predictions == labels) | (~valid_mask)).all(dim=1)
    track_success = (exact_match & (valid_points_per_row > 0)).float()

    accumulator["f1_sum"] += f1.mean().item()
    accumulator["precision_sum"] += precision.mean().item()
    accumulator["recall_sum"] += recall.mean().item()
    accumulator["tsr_sum"] += track_success.mean().item()
    accumulator["step_count"] += 1.0

    for row_index in range(logits.shape[0]):
        row_mask = valid_mask[row_index]
        if not row_mask.any():
            continue

        row_labels = labels[row_index][row_mask]
        if row_labels.sum().item() <= 0:
            continue

        row_scores = probabilities[row_index][row_mask]
        row_average_precision = average_precision(row_scores, row_labels)
        if row_average_precision is not None:
            accumulator["auprc_sum"] += row_average_precision
            accumulator["auprc_count"] += 1.0


def finalize_metrics(accumulator: dict[str, float]) -> dict[str, float]:
    step_count = accumulator["step_count"]
    auprc_count = accumulator["auprc_count"]
    return {
        "F1": accumulator["f1_sum"] / step_count if step_count > 0 else 0.0,
        "Precision": accumulator["precision_sum"] / step_count if step_count > 0 else 0.0,
        "Recall": accumulator["recall_sum"] / step_count if step_count > 0 else 0.0,
        "TSR": accumulator["tsr_sum"] / step_count if step_count > 0 else 0.0,
        "AUPRC": accumulator["auprc_sum"] / auprc_count if auprc_count > 0 else 0.0,
    }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    norm_params: dict,
    device: torch.device,
    progress_label: str,
) -> dict[str, float]:
    model.eval()
    accumulator = init_metric_accumulator()

    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_label, unit="batch", leave=False):
            if not batch:
                continue

            batch = move_batch_to_device(batch, device)
            batch_size, num_frames, max_tracks, _ = batch["gt_states"].shape
            if num_frames <= 1:
                continue

            unified_history = torch.cat(
                [batch["gt_history"], batch["gt_states"].transpose(1, 2)],
                dim=2,
            )
            history_len = batch["gt_history"].shape[2]

            for frame_index in range(1, num_frames):
                active_tracks = batch["gt_visibility"][:, frame_index - 1, :]
                if not active_tracks.any():
                    continue

                track_histories = unified_history[:, :, frame_index : frame_index + history_len, :]
                flat_history = track_histories.reshape(batch_size * max_tracks, history_len, 6)
                previous_active_tracks = (
                    batch["gt_visibility"][:, frame_index - 2, :].reshape(-1)
                    if frame_index >= 2
                    else torch.zeros_like(active_tracks.reshape(-1))
                )
                predicted_states = calculate_physics_prediction(
                    flat_history,
                    norm_params,
                    device,
                    previous_active_tracks,
                ).view(batch_size, max_tracks, 6)

                measurements = batch["measurements"][:, frame_index, :, :]
                measurement_mask = batch["meas_mask"][:, frame_index, :]
                amp_context = (
                    autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    outputs = model(
                        track_history=track_histories,
                        point_features=measurements,
                        point_mask=measurement_mask,
                        track_inputs_raw=predicted_states,
                        prev_hidden_states=None,
                    )
                affinity = outputs[0] if isinstance(outputs, tuple) else outputs
                num_points = measurement_mask.shape[1]
                point_valid_mask = measurement_mask.unsqueeze(1).expand(batch_size, max_tracks, num_points)

                update_metrics(
                    accumulator,
                    affinity[:, :max_tracks, :],
                    batch["gt_attn_mask"][:, frame_index, :, :],
                    point_valid_mask,
                    active_tracks,
                )

    return finalize_metrics(accumulator)


def build_loader(task_path: str | Path, norm_params: dict) -> DataLoader:
    resolved_task_path = resolve_path(task_path)
    clip_samples = precompute_tracklets_for_association(resolved_task_path, TRACKLET_LEN)
    dataset = AssociationDataset(
        resolved_task_path,
        clip_samples,
        norm_params,
        tracklet_len=TRACKLET_LEN,
        history_len=HISTORY_LEN,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=association_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def format_table(rows: list[dict[str, str]]) -> str:
    headers = ["Task", "Checkpoint", "F1", "Precision", "Recall", "TSR", "AUPRC"]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = [
        " | ".join(row[header].ljust(widths[header]) for header in headers)
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_params = load_normalization_params(PRECOMPUTED_CACHE_PATH)
    loaders = {
        task_name: build_loader(task_path, norm_params)
        for task_name, task_path in TASKS
    }

    rows = []
    for checkpoint_name, checkpoint_path in CHECKPOINTS:
        model = build_deployment_model(
            checkpoint=resolve_path(checkpoint_path),
            map_location=device,
            eval_mode=True,
            **MODEL_PARAMS,
        ).to(device)

        for task_name, loader in loaders.items():
            metrics = evaluate_model(
                model,
                loader,
                norm_params,
                device,
                progress_label=f"{checkpoint_name} on {task_name}",
            )
            rows.append(
                {
                    "Task": task_name,
                    "Checkpoint": checkpoint_name,
                    "F1": f"{metrics['F1']:.4f}",
                    "Precision": f"{metrics['Precision']:.4f}",
                    "Recall": f"{metrics['Recall']:.4f}",
                    "TSR": f"{metrics['TSR']:.4f}",
                    "AUPRC": f"{metrics['AUPRC']:.4f}",
                }
            )

    print(format_table(rows), flush=True)


if __name__ == "__main__":
    main()
