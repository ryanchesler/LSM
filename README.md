# LSM
Large Scroll Model - Using LLM style pretraining to our scroll data for smarter, more generalizable models

We have many terabytes worth of scroll scans but only less than 1% of this has been segmented and labeled for ink. In order to use this unlabeled data in some way we can pretrain the model on a sort of fill-in-the-blank task. I call this masked volume modeling. Basically a 3d Unet takes in a random 128x128x128 crop from the scroll volumes, punch a bunch of holes in it and then the model tries to figure out what used to be there. In the process of doing this it begins to understand the patterns and structure of the scrolls. We can run this on any scroll volumes without the need for any labels. This does not directly solve the problem of ink detection but it gives us an extremely powerful initialization point, a base model that has very deep understanding of the scrolls.

My initial work has only covered two scrolls (1 and 2) but it can be trivially extended to the rest. This has already shown in my experiments to significantly aid in the convergence of my 3d ink detection model. Yielding a better model with less finetuning required, this will save epochs of compute for all downstream models based on it. 

This repo will host the model weights from my pretraining effort as well as code and directions to reproduce it

## Installation instructions
In order to run this repository one first needs to `pip install pytorch h5py numpy opencv-python wandb albumentations scikit-learn monai tifffile git+https://github.com/wolny/pytorch-3dunet.git`

After this the volume tifs need to be downloaded and converted to hdf5 format. In order to do this download the volume tifs following the directions here https://scrollprize.org/data_scrolls#data-server 

Now one must modify the volume_to_hdf.py script to point to the path of the volume or volumes they downloaded. Script needs to be modified here for the path https://github.com/ryanchesler/LSM/blob/main/volume_to_hdf.py#L9 and here https://github.com/ryanchesler/LSM/blob/main/volume_to_hdf.py#L18 for the id of the scroll and finally here https://github.com/ryanchesler/LSM/blob/main/volume_to_hdf.py#L17 for the target path of your newly formatted array. If you want to do this for multiple scrolls then you can modify it again and rerun it to add more scrolls to the hdf5 datasets, not changing the target path. It will just be written as a new dataset in the same array. 

Now with the scrolls prepared you can run the pretraining process. One modification needs to be made here to point to the path of your hdf5 array that was created in the previous step 
https://github.com/ryanchesler/LSM/blob/main/3d_pretraining.py#L225

You will need to login to a wandb account in order for logging to be done

here is a link to a couple of the training runs done. Unetr ended up having convergence problems so I switched to 3d unet which converged much more nicely and seemed to yield a better result. The shared checkpoint is for this 3d unet shown. Trained for many many hours on scrolls 1 and 2
https://api.wandb.ai/links/ryanchesler/efxhelqb

This model was kept intentionally simple so that it could be integrated as a backbone in other modeling efforts. In order to initialize a model with these weights one can do 

```
from pytorch3dunet.unet3d.model import get_model
model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 1,
                               "f_maps": 32, "num_groups": 4, "is_segmentation": False, "num_levels":5})


model = model.load_state_dict(torch.load("pretrained_mode.pth"))
```
## Model weights available
https://drive.google.com/file/d/1QdWAbbTRAC5oiMCYo_TADVyCM6b9Emdw/view?usp=sharing

## Results on downstream ink detection training
This model was later finetuned on ink detection data that was mapped to 3d. These are the results after only 1 epoch of training following the pretraining step.

Pretraining 1 hour
validation dice 0.4867855717078965
{‘avg_train_loss’: 0.3248909587883949, ‘avg_val_loss’: 0.32889398968148376}

Pretraining 8 hours
validation dice 0.5166384899821148
{‘avg_train_loss’: 0.31726684632301333, ‘avg_val_loss’: 0.30760372345319054}

Pretrained 24 hours
validation dice 0.535030485409383
{‘avg_train_loss’: 0.31709596128463746, ‘avg_val_loss’: 0.30348434508917577}

The final checkpoint was not benchmarked, but the trend is expected to continue and give the added benefit of instilling some scroll 2 knowledge. I dont expect this to make a yield great scroll 2 ink findings without ink labels on it but it should aid the model in picking it up more easily if some labels are found to train against. 
