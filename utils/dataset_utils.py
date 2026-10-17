import torch
from torch.utils.data import DataLoader
from dataset.polyp_dataset import get_polyp_dataset, get_test_polyp_dataset

TEST_DATASETS = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB"]

def get_train_val_dataloaders(args, img_dim):
    n_workers_train = int(args.get("nW", 0))
    n_workers_val = int(args.get("nW_eval", 0))
    batch_size = int(args["Batch_size"])
    pin_memory = torch.cuda.is_available()

    ds_train, ds_val = get_polyp_dataset(args, sam_image_dim=img_dim)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=n_workers_train, drop_last=True, pin_memory=pin_memory)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=n_workers_val, drop_last=False, pin_memory=pin_memory)
    
    return train_loader, val_loader

def get_test_dataloader(args, img_dim, dataset_origin):
    n_workers_test = 0
    pin_memory: bool = torch.cuda.is_available()

    ds_test = get_test_polyp_dataset(args, sam_image_dim=img_dim, dataset_origin=dataset_origin)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=n_workers_test, drop_last=False,pin_memory=pin_memory)

    return test_loader

def get_all_test_dataloaders(args, img_dim):
    test_loaders = {name: get_test_dataloader(args, img_dim, dataset_origin=name) for name in TEST_DATASETS}

    return test_loaders