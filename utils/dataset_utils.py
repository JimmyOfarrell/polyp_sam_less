import torch
from dataset.MSLesSeg import get_mslesseg_dataset

def get_dataset(args, transform, img_dim):
        
    if args['task'] == 'mslesseg':
        trainset, testset = get_mslesseg_dataset(args, sam_image_dim=img_dim)
    ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                     num_workers=int(args['nW']), drop_last=True)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    return ds, ds_val
