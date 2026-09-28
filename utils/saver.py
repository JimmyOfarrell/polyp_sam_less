from pathlib import Path
from time import time
from datetime import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import json
import csv
import wandb
import numpy as np 
import matplotlib.pyplot as plt


class Saver(object):
    """
    Saver allows for saving and restore networks.
    """

    def __init__(self, output_folder: Path, experiment_name: str, wandb_mode: str, wandb_project: str, wandb_entity: str, args):
        timestamp_str = datetime.fromtimestamp(
            time()).strftime('%Y-%m-%d_%H-%M-%S')
        self.path = os.path.join(output_folder, f'{timestamp_str+"_"+experiment_name}')
        print(self.path)
        self.writer = SummaryWriter(str(self.path))
        self.csv_file = os.path.join(self.path,'metrics.csv')
        # Create checkpoint sub-directory
        self.ckpt_path = os.path.join(self.path,'ckpt')
        os.makedirs(self.ckpt_path)
        self.vis_path = os.path.join(self.path,'vis')
        os.makedirs(self.vis_path)
        #wandb
        
        self.metrics_table = None
        self.columns = None
        self.rows = []
        
        wandb.init( project=wandb_project, entity=wandb_entity,mode=wandb_mode, tags=args['wandb_tags'])
        

    def save_model(self, net, name: str, step: int):
        """
        Save model parameters in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Save
        torch.save(state_dict, str(self.ckpt_path) +
                   '/' + f'{name}_{step:05d}.pth')        
    
    def save_data(self,data,name:str, subfolder:str = ''):
        ''' Save generic data in experiment folder '''
        os.makedirs(os.path.join(self.path, subfolder), exist_ok=True)
        torch.save(data, os.path.join(self.path, subfolder, f'{name}.pth'))
        
    def save_json(self, data, name: str, subfolder: str = ''):
        ''' Save json data in experiment folder '''
        os.makedirs(os.path.join(self.path, subfolder), exist_ok=True)
        with open(os.path.join(self.path, subfolder, f'{name}.json'), 'w') as f:
            json.dump(data, f, default=str, indent=2)
        
    def log_histogram(self, name: str, values: torch.Tensor, iter_n: int):
        '''
        Log histogram to WB 
        '''
        data = [[s] for s in values]
        table = wandb.Table(data=data, columns=[f"{name}"])
        wandb.log({f'{name}_histogram': wandb.plot.histogram(table, f"{name}", title=f"{name} Histogram")})
        
               

    def close(self):
        '''
        Close Tensorboard connection
        '''
        self.writer.close()
        self.csv_file.close()

    def log_loss(self, name: str, value: float, iter_n: int):
        '''
        Log loss to TB and comet_ml and csv
        '''
        self.writer.add_scalar(name, value, iter_n)
        wandb.log({name: value, "EPOCH": iter_n})
        with open(self.csv_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, iter_n, value])


    def log_images(self, title: str, images_vector: torch.Tensor):
        '''
        Log images to WANDB
        '''

        img_grid = torchvision.utils.make_grid(
            images_vector, normalize=True, nrow=10)
        images = wandb.Image(img_grid)
        wandb.log({title: images})  

    
    def log_text(self,title:str,text:str,step:int):
        self.writer.add_text(title,text,step)

    def log_hparams(self,params_dict):
        #self.writer.add_hparams(params_dict)
        
        self.writer.add_text('hparams',str(params_dict),0)
        wandb.config.update(params_dict, allow_val_change=True)
        with open(os.path.join(self.path, 'hparams.json'), 'w') as fp:
            json.dump(params_dict, fp)

    def log_cmd(self, cmd):
        self.writer.add_text(cmd, cmd, 0)

        with open(os.path.join(self.path, 'cmd.json'), 'w') as fp:
            json.dump(cmd, fp)
            
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
        
