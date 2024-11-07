"""Script for running inference and evaluation.

To run inference on unconditional backbone generation:
> python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional

To run inference on inverse folding
> python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_inverse_folding

To run inference on forward folding
> python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_forward_folding

##########
# Config #
##########

Config locations:
- configs/inference_unconditional.yaml: unconditional sampling config.
- configs/inference_forward_folding.yaml: forward folding sampling config.
- configs/inference_inverse_folding.yaml: inverse folding sampling config.

Most important fields:
- inference.num_gpus: Number of GPUs to use. I typically use 2 or 4.

- inference.unconditional_ckpt_path: Checkpoint path for hallucination.
- inference.forward_folding_ckpt_path: Checkpoint path for forward folding.
- inference.inverse_folding_ckpt_path: Checkpoint path for inverse folding.

- inference.interpolant.sampling.num_timesteps: Number of steps in the flow.

- inference.folding.folding_model: `esmf` for ESMFold and `af2` for AlphaFold2.

[Only for hallucination]
- inference.samples.samples_per_length: Number of samples per length.
- inference.samples.min_length: Start of length range to sample.
- inference.samples.max_length: End of length range to sample.
- inference.samples.length_subset: Subset of lengths to sample. Will override min_length and max_length.

#######################
# Directory structure #
#######################

inference_outputs/                      # Inference run name. Same as Wandb run.
├── config.yaml                         # Inference and model config.
├── length_N                            # Directory for samples of length N.
│   ├── sample_X                        # Directory for sample X of length N.
│   │   ├── bb_traj.pdb                 # Flow matching trajectory
│   │   ├── sample.pdb                  # Final sample (final step of trajectory).
│   │   ├── self_consistency            # Directory of SC intermediate files.
│   │   │   ├── codesign_seqs           # Directory with codesign sequence
│   │   │   ├── folded                  # Directory with folded structures for ProteinMPNN and the Codesign Seq.
│   │   │   ├── esmf                    # Directory of ESMFold outputs.
│   │   │   ├── parsed_pdbs.jsonl       # ProteinMPNN compatible data file.
│   │   │   ├── sample.pdb              # Copy of sample_x/sample.pdb to use in ProteinMPNN
│   │   │   └── seqs                    # Directory of ProteinMPNN sequences.
│   │   │       └── sample.fa           # FASTA file of ProteinMPNN sequences.
│   │   ├── top_sample.csv              # CSV of the SC metrics for the best sequences and ESMFold structure.
│   │   ├── sc_results.csv              # All SC results from ProteinMPNN/ESMFold.
│   │   ├── pmpnn_results.csv           # Results from running ProteinMPNN on the structure.
│   │   └── x0_traj.pdb                 # Model x0 trajectory.


"""

import os
import time
import numpy as np
import hydra
import torch
import pandas as pd
import glob
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from data.datasets import BaseDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
import torch.distributed as dist
from torch.utils.data import DataLoader


torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """

        # Read in checkpoint.
        ckpt_path = cfg.inference.unconditional_ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        self._original_cfg = cfg.copy()

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._data_cfg = self._original_cfg.data
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._task = self._data_cfg.task
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
        )
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

    def run_extraction(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')
        if self._data_cfg.dataset == 'ec':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                BaseDataset, self._cfg.ec_dataset, self._task)
            self._dataset_cfg = self._cfg.ec_dataset
        elif self._data_cfg.dataset == 'go':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                BaseDataset, self._cfg.go_dataset, self._task)
            self._dataset_cfg = self._cfg.go_dataset
        elif self._data_cfg.dataset == 'fold':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                BaseDataset, self._cfg.fold_dataset, self._task)
            self._dataset_cfg = self._cfg.fold_dataset

        self._datamodule = ProteinData(
            data_cfg=self._data_cfg,
            dataset_cfg=self._dataset_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        dataloader = DataLoader(self._train_dataset, batch_size=1, num_workers=32)
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)



@hydra.main(version_base=None, config_path="../configs", config_name="inference_unconditional")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_extraction()

    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
