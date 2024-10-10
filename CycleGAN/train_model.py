import os
import torch
import hydra
import wandb
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")
from CycleGAN import Autoencoder, PCDDataModule

# main function using Hydra to organize configuration
@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    # define Weights and Biases logger
    wandb_logger = WandbLogger(project='CycleGAN')
    
    # print GPU information
    print('CUDA available:', torch.cuda.is_available())
    print('MPS available:', torch.backends.mps.is_available())
    L.seed_everything(100, workers=True)

    # define paths
    male_path = os.path.join(hydra.utils.get_original_cwd(), 'data/train/male/')
    female_path = os.path.join(hydra.utils.get_original_cwd(), 'data/train/female/')
    test_male_path = os.path.join(hydra.utils.get_original_cwd(), 'data/val/male')
    test_female_path = os.path.join(hydra.utils.get_original_cwd(), 'data/val/female')

    data_module = PCDDataModule(male_path, female_path, 
                                test_male_path, test_female_path, 
                                n_points=2048, fraction=1.,
                                batch_size=16, num_workers=8)

    # define the model
    model = Autoencoder(g_learning_rate=1e-4, d_learning_rate=1e-4, logging=True)

    # define the pytorch lightning trainer 
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=1000,
        check_val_every_n_epoch=5,
        deterministic=True,
        num_sanity_val_steps=1,
        logger=wandb_logger,
    )
    
    # train the model
    print("Starting new training")
    trainer.fit(model, data_module)

    wandb.finish()

if __name__ == "__main__":
    main()
    print("Done!")