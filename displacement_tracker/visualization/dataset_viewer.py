import matplotlib
matplotlib.use("TkAgg")

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import random


class DatasetViewer:
    def __init__(self, dataset):
        self.dataset = dataset

    def show(self, idx: int, overlay: bool = True) -> None:
        if overlay:
            self.show_overlay(idx)
        else:
            self.show_split(idx)

    def _prepare_display_data(self, idx: int):
        sample = self.dataset[idx]

        def to_numpy(t):
            return (
                t.squeeze().cpu().numpy()
                if isinstance(t, torch.Tensor)
                else np.array(t)
            )

        arr_feat = to_numpy(sample["feature"])
        arr_prewar = to_numpy(sample["prewar"])
        arr_label = to_numpy(sample["label"])

        meta = sample["meta"]
        if isinstance(meta, bytes):
            meta = meta.decode("utf-8")

        try:
            meta_dict = json.loads(str(meta))
            meta_text = "\n".join(f"{k}: {v}" for k, v in meta_dict.items())
        except json.JSONDecodeError:
            meta_text = str(meta)

        return arr_feat, arr_prewar, arr_label, meta_text

    def _plot_meta(self, ax, text):
        ax.axis("off")
        ax.text(
            0.05,
            0.95,
            text,
            fontsize=11,
            color="black",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    def _plot_overlay_on_axis(self, ax, base, mask):
        ax.imshow(base, cmap="gray", interpolation="none")
        ax.imshow(
            np.ones_like(mask),
            cmap="spring",
            alpha=mask,
            interpolation="none",
        )

        h, w = mask.shape
        for i in range(1, 3):
            ax.axhline(y=i * h // 3, color="red", linestyle="--")
            ax.axvline(x=i * w // 3, color="red", linestyle="--")

        ax.axis("off")

    def show_overlay(self, idx: int) -> None:
        arr_feat, arr_prewar, arr_label, meta_text = self._prepare_display_data(idx)

        fig, axes = plt.subplots(
            1, 3, figsize=(14, 6),
            gridspec_kw={"width_ratios": [1, 3, 3]}
        )

        self._plot_meta(axes[0], meta_text)

        axes[1].imshow(arr_prewar, cmap="gray", interpolation="none")
        axes[1].set_title("Prewar")
        axes[1].axis("off")

        self._plot_overlay_on_axis(axes[2], arr_feat, arr_label)
        axes[2].set_title("Current + Label")

        plt.tight_layout()
        plt.show()

    def show_split(self, idx: int) -> None:
        arr_feat, arr_prewar, arr_label, meta_text = self._prepare_display_data(idx)

        fig, axes = plt.subplots(
            1, 4, figsize=(16, 6),
            gridspec_kw={"width_ratios": [1, 3, 3, 3]}
        )

        self._plot_meta(axes[0], meta_text)

        axes[1].imshow(arr_prewar, cmap="gray", interpolation="none")
        axes[1].set_title("Prewar")
        axes[1].axis("off")

        axes[2].imshow(arr_feat, cmap="gray", interpolation="none")
        axes[2].set_title("Current")
        axes[2].axis("off")

        axes[3].imshow(arr_label, cmap="gray")
        axes[3].set_title("Label")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()

    def show_batch(self, indices: list[int]) -> None:
        indices = indices[:18]
        n_plots = len(indices)
        if n_plots == 0:
            return

        best_rows = 1
        best_cols = n_plots
        best_waste = float("inf")
        best_scale = 0.0

        for cols in range(1, min(n_plots, 6) + 1):
            rows = (n_plots + cols - 1) // cols
            waste = rows * cols - n_plots
            scale = min(12 / cols, 8 / rows)

            if waste < best_waste or (waste == best_waste and scale > best_scale):
                best_waste = waste
                best_rows, best_cols = rows, cols
                best_scale = scale

        figsize = (best_cols * best_scale, best_rows * best_scale)
        fig, axes = plt.subplots(best_rows, best_cols, figsize=figsize)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        axes_flat = axes.flatten()

        for i, idx in enumerate(indices):
            ax = axes_flat[i]
            arr_feat, arr_prewar, arr_label, _ = self._prepare_display_data(idx)

            combined = np.concatenate([arr_prewar, arr_feat], axis=1)
            ax.imshow(combined, cmap="gray", interpolation="none")

            txt = ax.text(
                0.5,
                0.95,
                str(idx),
                fontsize=10,
                color="lightgreen",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontweight="bold",
            )
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

            ax.axis("off")

        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from displacement_tracker.paired_image_dataset import PairedImageDataset

    ds = PairedImageDataset(
        "tif_files/historic/processed/deir_el_balah_nuseirat_gaza_city_20251014_121159_ssc7_u0002_visual_clip.h5"
    )

    print("Dataset length:", len(ds))

    viewer = DatasetViewer(ds)

    if len(ds) > 0:
        indices = random.sample(range(len(ds)), min(12, len(ds)))
        print("Showing indices:", indices)
        viewer.show_batch(indices)
    else:
        print("Dataset is empty.")
