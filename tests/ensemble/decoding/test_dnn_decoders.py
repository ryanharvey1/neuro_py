import random
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import neuro_py as npy


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    print(f"Random seed {seed} has been set.")
    return seed


def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled.")
    else:
        print("GPU is enabled.")

    return device


def seed_worker():
    """DataLoader will reseed workers following randomness in multi-process data
    loading algorithm.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TestDecodePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.DEVICE = set_device()
        cls.SEED = set_seed(seed=0, seed_torch=True)
        cls.N_TRIALS = 50
        cls.TRIAL_LENGTH = 10
        cls.N_NEURONS = 100
        cls.N_BV_OUT = 2
        cls.BEHAVIORAL_VARS = [
            "pv_x",
            "pv_y",
            "pv_speed",
            "pv_dir",
            "pv_dir_cos",
            "pv_dir_sin",
        ]
        cls.PREDICT_BV = [
            cls.BEHAVIORAL_VARS.index("pv_x"),
            cls.BEHAVIORAL_VARS.index("pv_y"),
        ]
        cls.RESULTS_PATH = None

    def generate_random_data(self):
        nsv_trials = [
            pd.DataFrame(np.random.rand(self.TRIAL_LENGTH, self.N_NEURONS))
            for _ in range(self.N_TRIALS)
        ]
        bv_trials = np.random.rand(self.N_TRIALS, self.TRIAL_LENGTH, self.N_BV_OUT)
        return nsv_trials, bv_trials

    def create_partitions(self, nsv_trials, bv_trials):
        return [(nsv_trials, bv_trials) * 3]

    def create_base_hyperparams(self, decoder_type):
        return {
            "batch_size": 1 if decoder_type == "M2MLSTM" else 32,
            "num_workers": 5,
            "model": decoder_type,
            "behaviors": self.PREDICT_BV,
            "bins_before": 0,
            "bins_current": 1,
            "bins_after": 0,
            "accelerator": self.DEVICE,
            "seed": self.SEED,
        }

    def run_decoder_test(self, decoder_type, model_args):
        nsv_trials, bv_trials = self.generate_random_data()
        partitions = self.create_partitions(nsv_trials, bv_trials)

        hyperparams = self.create_base_hyperparams(decoder_type)

        hyperparams["model_args"] = model_args

        bv_preds_folds, bv_models_folds, norm_params_folds, metrics_folds = (
            npy.ensemble.decoding.pipeline.train_model(
                partitions,
                hyperparams,
                resultspath=self.RESULTS_PATH,
                stop_partition=None,
            )
        )

        self.validate_outputs(
            bv_preds_folds, bv_models_folds, norm_params_folds, metrics_folds
        )
        return bv_models_folds

    def validate_outputs(
        self, bv_preds_folds, bv_models_folds, norm_params_folds, metrics_folds
    ):
        self.assertIsInstance(bv_preds_folds, list)
        self.assertIsInstance(bv_models_folds, list)
        self.assertIsInstance(norm_params_folds, list)
        self.assertIsInstance(metrics_folds, dict)

        self.assertEqual(len(bv_preds_folds), 1)
        self.assertEqual(len(bv_models_folds), 1)
        self.assertEqual(len(norm_params_folds), 1)

        self.assertEqual(
            bv_preds_folds[0].shape,
            (self.N_TRIALS * self.TRIAL_LENGTH, len(self.PREDICT_BV)),
        )
        self.assertEqual(len(norm_params_folds[0]), 4)

    def test_mlp_decoder(self):
        model_args = {
            "in_dim": None,
            "out_dim": len(self.PREDICT_BV),
            "hidden_dims": [512, 512, 0.2, 512],
            "args": {
                "clf": False,
                "activations": nn.CELU,
                "criterion": F.mse_loss,
                "epochs": 1,
                "lr": 3e-2,
                "base_lr": 1e-2,
                "max_grad_norm": 1.0,
                "iters_to_accumulate": 1,
                "weight_decay": 1e-2,
                "num_training_batches": None,
                "scheduler_step_size_multiplier": 2,
            },
        }
        self.run_decoder_test("MLP", model_args)

    def test_ndt_decoder(self):
        model_args = {
            "in_dim": self.N_NEURONS,
            "out_dim": self.N_BV_OUT,
            "hidden_dims": [64, 1, 1, 0.0, 0.0],
            "max_context_len": self.TRIAL_LENGTH,
            "args": {
                "clf": False,
                "activations": nn.CELU,
                "criterion": F.mse_loss,
                "epochs": 1,
                "lr": 3e-2,
                "base_lr": 1e-2,
                "max_grad_norm": 1.0,
                "iters_to_accumulate": 1,
                "weight_decay": 1e-2,
                "num_training_batches": None,
                "scheduler_step_size_multiplier": 1,
            },
        }
        bv_models_folds = self.run_decoder_test("NDT", model_args)
        self.assertEqual(
            bv_models_folds[0]
            .transformer_encoder.layers[0]
            .self_attn.out_proj.in_features,
            self.N_NEURONS,
        )
        self.assertEqual(
            bv_models_folds[0]
            .transformer_encoder.layers[0]
            .self_attn.out_proj.out_features,
            self.N_NEURONS,
        )
        self.assertEqual(bv_models_folds[0].decoder[-1].out_features, 2)

    def test_m2mlstm_decoder(self):
        model_args = {
            "in_dim": self.N_NEURONS,
            "out_dim": self.N_BV_OUT,
            "hidden_dims": [64, 1, 0.0],
            "args": {
                "clf": False,
                "activations": nn.CELU,
                "criterion": F.mse_loss,
                "epochs": 1,
                "lr": 3e-2,
                "base_lr": 1e-2,
                "max_grad_norm": 1.0,
                "iters_to_accumulate": 1,
                "weight_decay": 1e-2,
                "num_training_batches": None,
                "scheduler_step_size_multiplier": 1,
            },
        }
        self.run_decoder_test("M2MLSTM", model_args)


if __name__ == "__main__":
    unittest.main()
