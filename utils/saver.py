import os
from typing import Any
import torch
import torch.nn as nn
import torchvision
import json
import csv
import wandb
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Saver:
    """
    Saver handles experiment tracking, model checkpointing, and logging.

    Responsibilities
    ----------------
    * Create a timestamped experiment folder under ``output_folder``.
    * Log scalar metrics to TensorBoard, WandB, and a CSV file.
    * Save model checkpoints (``.pth``) at regular intervals.
    * Save visualization images (pred / GT) to disk.
    * Build and maintain a WandB metrics table for cross-run comparison.

    Folder layout
    -------------
    ::

        results/
        └── <YYYY-MM-DD_HH-MM-SS>_<experiment_name>/
            ├── events.out.tfevents.*      (TensorBoard)
            ├── metrics.csv                (long-format: metric, epoch, value)
            ├── ckpt/                      (checkpoints)
            ├── vis/                       (visualization images)
            └── hparams.json, cmd.json     (experiment metadata)

    Attributes
    ----------
    path : str
        Root folder of this experiment (timestamped).
    ckpt_path : str
        Folder for saved model checkpoints.
    vis_path : str
        Folder for saved visualization images.
    csv_file : str
        Path to the metrics CSV file.
    writer : SummaryWriter
        TensorBoard writer.
    wandb_enabled : bool
        Whether WandB is active (i.e. not in ``disabled`` mode).
    metrics_table : wandb.Table or None
        WandB table used for cross-run comparison.
    """

    def __init__(
        self,
        output_folder: Path,
        experiment_name: str,
        wandb_mode: str,
        wandb_project: str,
        wandb_entity: str,
        args: dict,
    ):
        """
        Parameters
        ----------
        output_folder : Path or str
            Root directory where experiment folders will be created.
        experiment_name : str
            Short name for the experiment (e.g. ``'polyp_phase1'``).
        wandb_mode : str
            One of ``'online'``, ``'offline'``, or ``'disabled'``.
        wandb_project : str
            WandB project name.
        wandb_entity : str or None
            WandB entity (user or team). ``None`` uses the current account.
        args : dict
            Full argument dictionary (logged as WandB config and saved to disk).
        """
        # ------------------------------------------------------------
        # 1) Create timestamped experiment folder
        # ------------------------------------------------------------
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.path = os.path.join(output_folder, f'{timestamp_str}_{experiment_name}')
        os.makedirs(self.path, exist_ok=True)
        print(f"[Saver] Results will be saved to: {self.path}")

        # ------------------------------------------------------------
        # 2) TensorBoard writer
        # ------------------------------------------------------------
        self.writer = SummaryWriter(str(self.path), flush_secs=30)

        # ------------------------------------------------------------
        # 3) CSV metrics log (with header for easy pandas reading)
        # ------------------------------------------------------------
        self.csv_file = os.path.join(self.path, 'metrics.csv')
        with open(self.csv_file, 'w', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow(['metric', 'epoch', 'value'])

        # ------------------------------------------------------------
        # 4) Sub-folders for checkpoints and visualizations
        # ------------------------------------------------------------
        self.ckpt_path = os.path.join(self.path, 'ckpt')
        self.vis_path  = os.path.join(self.path, 'vis')
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.vis_path,  exist_ok=True)

        # ------------------------------------------------------------
        # 5) WandB metrics table buffers
        # ------------------------------------------------------------
        self.metrics_table: wandb.Table | None = None
        self.columns: list[str] | None = None
        self.rows: list[list] = []

        # ------------------------------------------------------------
        # 6) Initialize WandB (safely — never crash the training run)
        # ------------------------------------------------------------
        self.wandb_enabled = False
        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                mode=wandb_mode,
                tags=args.get('wandb_tags', []),
                name=experiment_name,
                config=args,
                reinit=True,
            )
            self.wandb_enabled = (wandb_mode != 'disabled') and (wandb.run is not None)
            if self.wandb_enabled:
                print(f"[Saver] WandB run: {wandb.run.name} (id: {wandb.run.id})")
            else:
                print("[Saver] WandB disabled.")
        except Exception as e:
            print(f"[Saver] WandB init failed: {e}")
            print("[Saver] Continuing without WandB.")
            self.wandb_enabled = False   

    def save_model(self, net: nn.Module, name: str, step: int) -> None:
        """
        Save the model's state dictionary to the checkpoint folder.

        The file is named ``<name>_<step:05d>.pth`` where ``<step>`` is
        zero-padded to 5 digits so that alphabetical sorting matches
        numerical ordering (e.g. ``00000, 00005, 00010, ...``).

        Parameters
        ----------
        net : nn.Module
            The model whose ``state_dict`` will be saved.
        name : str
            Base name for the checkpoint (e.g., ``'net_last'`` or ``'net_best'``).
        step : int
            Training step or epoch index used as a suffix.

        Returns
        -------
        None
            The checkpoint is written to ``self.ckpt_path``.
        """
        state_dict = net.state_dict()
        path = os.path.join(self.ckpt_path, f'{name}_{step:05d}.pth')
        torch.save(state_dict, path)
        print(f"[Saver] Saved checkpoint: {path}")        
    
    def save_data(self, data: Any, name: str, subfolder: str = '') -> str:
        """
        Save generic data as a ``.pth`` file in the experiment folder.

        Uses ``torch.save`` (pickle-based) and can store any picklable
        object: tensors, dictionaries, lists, numpy arrays, ...

        Parameters
        ----------
        data : Any
            Object to save (must be picklable).
        name : str
            Base file name (without extension). ``.pth`` is appended.
        subfolder : str, default ''
            Optional sub-folder under ``self.path``. If empty, the file is
            saved directly under ``self.path``.

        Returns
        -------
        str
            Full path to the saved file.

        Examples
        --------
        >>> saver.save_data({'a': 1, 'b': 2}, 'summary')
        >>> saver.save_data(tensors, 'sample_0', subfolder='tensors')
        """
        out_dir = os.path.join(self.path, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{name}.pth')
        torch.save(data, out_path)
        print(f"[Saver] Saved data: {out_path}")
        return out_path
        
    def save_json(self, data: Any, name: str, subfolder: str = '') -> str:
        """
        Save a Python object as a JSON file in the experiment folder.

        Parameters
        ----------
        data : Any
            Object to serialize. Non-JSON-serializable objects (e.g. tensors,
            datetime, Path) are converted to strings via ``default=str``.
        name : str
            Base file name (without extension). ``.json`` is appended.
        subfolder : str, default ''
            Optional sub-folder under ``self.path``.

        Returns
        -------
        str
            Full path to the saved JSON file.

        Notes
        -----
        Uses UTF-8 encoding and ``ensure_ascii=False`` so that non-ASCII
        characters (e.g., Persian, Arabic, Chinese) are preserved as-is.
        """
        out_dir = os.path.join(self.path, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{name}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=str, indent=2, ensure_ascii=False)
        print(f"[Saver] Saved JSON: {out_path}")
        return out_path
        
    def log_histogram(self, name: str, values, iter_n: int) -> None:
        """
        Log a histogram of ``values`` to WandB at a given step.

        Parameters
        ----------
        name : str
            Name of the histogram (used as key and title).
        values : torch.Tensor or np.ndarray or list
            Values to build the histogram from.
        iter_n : int
            Training step (used as the WandB log step).

        Notes
        -----
        Uses WandB's native ``Histogram`` instead of a table-based one
        for efficiency. Large inputs (> ``MAX_POINTS``) are subsampled
        to keep logging lightweight.

        Does nothing if WandB is disabled.
        """
        if not self.wandb_enabled:
            return

        # Convert to a flat numpy array
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values).flatten()

        # Subsample very large inputs
        MAX_POINTS = 10_000
        if values.size > MAX_POINTS:
            idx = np.random.choice(values.size, MAX_POINTS, replace=False)
            values = values[idx]

        wandb.log(
            {f'{name}_histogram': wandb.Histogram(values)},
            step=iter_n,
        )        

    def close(self) -> None:
        """
        Close all open resources: TensorBoard writer, WandB run, and CSV file.

        Notes
        -----
        The CSV file is opened and closed per ``log_loss`` call, so there is
        no persistent file handle to close here. Only the TensorBoard writer
        and the WandB run need explicit shutdown.
        """
        # --- TensorBoard ---
        try:
            self.writer.close()
            print("[Saver] TensorBoard writer closed.")
        except Exception as e:
            print(f"[Saver] Failed to close TensorBoard writer: {e}")

        # --- WandB ---
        if getattr(self, 'wandb_enabled', False):
            try:
                wandb.finish()
                print("[Saver] WandB run finished.")
            except Exception as e:
                print(f"[Saver] Failed to finish WandB: {e}")

        # --- CSV ---
        # NOTE: no persistent file handle exists for CSV (opened/closed
        # inside log_loss). Nothing to close here.

    def log_loss(self, name: str, value: float, iter_n: int) -> None:
        """
        Log a scalar metric to TensorBoard, WandB, and CSV.

        Parameters
        ----------
        name : str
            Metric name (e.g., 'train_loss', 'val_dice').
        value : float
            Numeric value of the metric.
        iter_n : int
            Step number (typically the epoch index).
        """
        # TensorBoard
        self.writer.add_scalar(name, value, iter_n)

        # WandB (skipped if disabled)
        if self.wandb_enabled:
            wandb.log({name: value, "EPOCH": iter_n}, step=iter_n)

        # CSV (append one row)
        with open(self.csv_file, 'a', encoding='UTF8', newline='') as f:
            csv.writer(f).writerow([name, iter_n, value])

    def log_images(
        self,
        title: str,
        images_vector: torch.Tensor,
        step: int = 0,
        nrow: int = 10,
        normalize: bool = True,
    ) -> None:
        """
        Log a grid of images to WandB.

        Parameters
        ----------
        title : str
            Key under which the grid is logged.
        images_vector : torch.Tensor
            Batch of images with shape ``(B, C, H, W)``.
        step : int, default 0
            WandB step (typically the epoch index).
        nrow : int, default 10
            Number of images per row in the grid.
        normalize : bool, default True
            If True, min-max normalize the grid to [0, 1] before logging.

        Notes
        -----
        The ``make_grid`` output is channel-first ``(C, H, W)``; WandB
        expects channel-last, so we permute to ``(H, W, C)`` before
        wrapping in ``wandb.Image``.

        Does nothing if WandB is disabled.
        """
        if not self.wandb_enabled:
            return

        # Build a grid (channel-first: C, H, W)
        img_grid = torchvision.utils.make_grid(
            images_vector.detach().cpu(),
            normalize=normalize,
            nrow=nrow,
        )

        # wandb.Image expects channel-last: (H, W, C)
        img_np = img_grid.permute(1, 2, 0).numpy()

        # Clamp to [0, 1] for safety (after permute)
        img_np = np.clip(img_np, 0.0, 1.0)

        wandb.log({title: wandb.Image(img_np)}, step=step)
    
    def log_text(self,title:str,text:str,step:int):
        self.writer.add_text(title,text,step)

    def log_hparams(self, params_dict: dict) -> str:
        """
        Log hyperparameters to TensorBoard, WandB, and a JSON file.

        Parameters
        ----------
        params_dict : dict
            Hyperparameter dictionary (typically ``args``).
            Non-JSON-serializable values (e.g., ``torch.device``) are
            converted to strings via ``default=str``.

        Returns
        -------
        str
            Full path to the saved ``hparams.json`` file.
        """
        # 1) TensorBoard — write as pretty JSON text
        hparams_str = json.dumps(params_dict, indent=2, default=str)
        self.writer.add_text('hparams', hparams_str, 0)

        # 2) WandB — update run config (skip if disabled)
        #    Note: If wandb.init(config=args) was already called, this
        #    update is redundant but harmless.
        if self.wandb_enabled:
            try:
                wandb.config.update(params_dict, allow_val_change=True)
            except Exception as e:
                print(f"[Saver] WandB config update failed: {e}")

        # 3) JSON file
        out_path = os.path.join(self.path, 'hparams.json')
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(params_dict, fp, indent=2, default=str, ensure_ascii=False)

        print(f"[Saver] Saved hyperparameters: {out_path}")
        return out_path

    def log_cmd(self, cmd: str) -> str:
        """
        Log the command-line invocation to TensorBoard and to a JSON file.

        Parameters
        ----------
        cmd : str
            The command-line string (e.g., "python main.py --task polyp").

        Returns
        -------
        str
            Full path to the saved ``cmd.json`` file.

        Notes
        -----
        The TensorBoard tag is fixed to ``'cmd'`` so that multiple runs
        appear under the same tag and can be compared side-by-side.
        """
        # 1) TensorBoard — fixed tag for comparability across runs
        self.writer.add_text('cmd', cmd, 0)

        # 2) JSON file
        out_path = os.path.join(self.path, 'cmd.json')
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump({'cmd': cmd}, fp, indent=2, ensure_ascii=False)

        print(f"[Saver] Saved command: {out_path}")
        return out_path
     
    def wb_mask(self, img, mask, gt):
        return wandb.Image(img, masks={"ground_truth" : {"mask_data" : gt, 
                                                        "class_labels" : {0: "background", 1: "mask"}},
                                        "predictions": { "mask_data": mask+2, 
                                                        "class_labels": {2: "background", 3: "mask"}}
        })
            
    @torch.no_grad()
    def log_slices(self, slices: torch.Tensor, masks: torch.Tensor, ground_truth: torch.tensor, step: int, name: str, phase: str):

        assert len(slices.shape)==5, 'Missing dimension.'
        assert slices.shape[1] == masks.shape[1] == ground_truth.shape[1]
        assert slices.shape[-1] == masks.shape[-1] == ground_truth.shape[-1]
        num_samples = slices.shape[0]
        num_slices = slices.shape[-1]
        s = slices.squeeze(1)
        msk = masks.squeeze(1)
        gt_in = ground_truth.squeeze(1)
        for b in range(num_samples):
            wandb_mask_logs = []
            for idx in range(num_slices):
                img = s[b, :, :, idx]
                m = msk[b, :, : , idx]
                gt = gt_in[b, :, :, idx]

                wandb_mask_logs.append(self.wb_mask(img.cpu().numpy(), m.cpu().numpy(), gt.cpu().numpy()))
            wandb.log({f"Segmentation/{phase}/{name}": wandb_mask_logs, 'step': step})
    
    @torch.no_grad()
    def save_images(self, images: torch.Tensor,masks: torch.Tensor, name: str, step: int):
        '''
        Save images to disk
        '''
        num_images = images.shape[-1]
        assert num_images == masks.shape[-1]
        images = images.cpu().detach().numpy()
        masks = masks.cpu().detach().numpy()
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 2, rows * 2))

        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i < num_images:
                ax.imshow(images[...,i], cmap='gray')
                ax.imshow(masks[...,i], cmap='hot', alpha=0.3)
            ax.axis('off')
        for i in range(num_images, rows * cols):
            fig.delaxes(axes.flatten()[i])
        plt.tight_layout()

        plt.savefig(os.path.join(self.vis_path, name+'.png'))

        plt.close()
        
    @torch.no_grad()
    def save_batch(self, images: torch.Tensor, name: str, step: int, remove_batch_dim: bool = True, subfolder: str = '', cmap = None, normalize = False):
        
        os.makedirs(os.path.join(self.vis_path, subfolder), exist_ok=True)
    
        if remove_batch_dim:
            images = images.squeeze(0)
            if images.dim() == 3:
                images = images.unsqueeze(1)
        if normalize:
            images = (images - images.min()) / (images.max() - images.min())
                
        if cmap is not None:
            colormap = plt.get_cmap(cmap)
            np_imgs = images.cpu().numpy()
            color_imgs = colormap(np_imgs)
            images = torch.from_numpy(color_imgs).permute(0, 3, 1, 2)
            
        torchvision.utils.save_image(images, os.path.join(self.vis_path, subfolder, f'{name}_{step}.png'))
        
    
    def create_table(self, columns=None):
        """
        Create (or recreate) the W&B table with the specified columns.
        If called multiple times, it overwrites any existing table 
        and clears old rows.
        
        :param columns: A list of column names. Must include "Step" 
                        (or any other index column you prefer).
                        Example: ["Step", "DiceTrue", "DiceFalse"]
        """
        if columns is None:
            columns = ["Step"]
        self.columns = columns
        
        # Create a new table
        self.metrics_table = wandb.Table(columns=self.columns)
        
        # Clear any existing row buffer
        self.rows = []
        
        #wandb.log({"metrics_table": self.metrics_table})
        
    def add_columns(self, new_cols):
        """
        Example if you want to dynamically add new columns.
        """
        if self.metrics_table is None:
            raise RuntimeError("No table to expand; please call create_table(...) first.")

        # Filter out columns that already exist
        new_cols = [col for col in new_cols if col not in self.columns]
        if not new_cols:
            return  # no new columns needed

        # Expand local columns
        self.columns.extend(new_cols)

        # Create new table with updated columns
        new_table = wandb.Table(columns=self.columns)

        # Copy old rows, adding None for new columns
        updated_rows = []
        for old_row in self.rows:
            # Match old_row with old columns
            row_dict = dict(zip(self.columns[:len(old_row)], old_row))
            new_row = []
            for col in self.columns:
                new_row.append(row_dict.get(col, None))
            updated_rows.append(new_row)
            new_table.add_data(*new_row)

        # Replace the old table & row buffer
        self.metrics_table = new_table
        self.rows = updated_rows

        wandb.log({"metrics_table": self.metrics_table})
        
    def log_table_row(self, step: int, metrics_dict: dict):
        """
        Adds one row of metrics to the W&B table. 
        You must call create_table() at least once 
        before using this function.
        
        :param step: Integer representing the current step/test run index.
        :param metrics_dict: {column_name: value}, 
                             matching the columns in self.columns (excluding "Step").
        """
        # Ensure the table is created
        if self.metrics_table is None or self.columns is None:
            raise RuntimeError("You must call create_table(...) before logging metrics.")

        # Build row data in the same order as self.columns
        row_data = []
        for col in self.columns:
            if col == "Step":
                row_data.append(step)
            else:
                val = metrics_dict.get(col, None)
                if isinstance(val, torch.Tensor):
                    val = val.item()  # Convert single-value tensor to float
                elif isinstance(val, np.ndarray):
                    val = val.item()  # if it’s a 1-element array
                if isinstance(val, float):
                    val = round(val, 4)
                row_data.append(val)

        # Store row in local buffer
        self.rows.append(row_data)
        
        # Add the row to the W&B table
        self.metrics_table.add_data(*row_data)
        
        # Log/update in W&B
        wandb.log({"metrics_table": self.metrics_table, "EPOCH": step})
        
