import torch
from torch.utils.data import DataLoader
from typing import List
import argparse
import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from lightning.pl_module import TurbulenceClosureWrap

import yaml
import copy
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import omegaconf
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from foam_io.foam_io import inject_by_files
from omegaconf import ListConfig, DictConfig

from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from setup_utils import log_hyperparameters, PROJECT_ROOT

OmegaConf.register_new_resolver("fix_path", lambda path: path.replace("/", "\\"))
OmegaConf.register_new_resolver("torch", lambda x: eval(x))
OmegaConf.register_new_resolver("scipy", lambda x: eval(x))


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    return callbacks



#default_config_path = os.path.join("config", "macedo")
#default_config_name = "final_duct_nut_16.yaml"

#default_config_path = os.path.join("config", "macedo")
#default_config_name = "final_duct_rperp.yaml"

default_config_path = os.path.join("config", "mcconkey")
default_config_name = "config_nut.yaml"




parser = argparse.ArgumentParser(description="Run the training and testing scripts with specified configuration.")
parser.add_argument('--config-path', type=str, default=default_config_path,
                    help='Path to the configuration directory')
parser.add_argument('--config-name', type=str, default=default_config_name,
                    help='Name of the configuration file')

args = parser.parse_args()

def inject(model, loader, save_dir, save_name, template_path, non_negative=False,
           condition_trace=True, ignore_nondiag_z=True):
    out = model.predict_from_loader(loader, device=model.device, non_negative=non_negative, ignore_nondiag_z=ignore_nondiag_z)
    os.makedirs(save_dir, exist_ok=True)
    numpy_file = os.path.join(save_dir, save_name)
    np.save(numpy_file, out)
    inject_by_files(numpy_file, template_path, save_dir)




@hydra.main(config_path=args.config_path, config_name=args.config_name)
def main(cfg: omegaconf.DictConfig):
    hydra_dir = Path(HydraConfig.get().run.dir)

    # -------------Load Data-------------
    datamodule = hydra.utils.instantiate(cfg.data, batch_size=cfg.training.batch_size,
                                         _recursive_=True)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()


    # --------Model, loader, optimizer---------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, training_cfg=cfg.training, irreps_in=datamodule.irreps_x,
                                    _recursive_=False)
    #model.scaler, model.label_scaler = datamodule.scaler, datamodule.label_scaler
    print(f'Model params: {model.num_params}')

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.data_scaler}>")
    model.data_scaler = copy.deepcopy(datamodule.data_scaler)
    model.label_scaler = copy.deepcopy(datamodule.label_scaler)
    #torch.save(datamodule.scaler, hydra_dir / 'scaler.pt')
    #torch.save(datamodule.label_scaler, hydra_dir / 'label_scaler.pt')
    # Instantiate the callbacks
    #callbacks: List[Callback] = build_callbacks(cfg=cfg)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=hydra_dir,
                                          save_top_k=3,
                                          save_last=True,
                                          save_weights_only=True,
                                          filename='ckpt')

    trainer = Trainer(**cfg.training.trainer, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    best_checkpoint_path = checkpoint_callback.best_model_path
    if best_checkpoint_path:
        model.load_state_dict(torch.load(best_checkpoint_path)['state_dict'])
        print(f"Loaded best model from {best_checkpoint_path} for testing.")

    trainer.test(model, test_dataloaders=test_loader)
    if 'injection' in cfg.keys():
        if type(cfg.injection) == ListConfig:
            [hydra.utils.instantiate(inj, _target_=inject, model=model, loader=test_loader) for inj in cfg.injection]
        else:
            hydra.utils.instantiate(cfg.injection, _target_=inject, model=model, loader=test_loader)







if __name__ == "__main__":
    main()