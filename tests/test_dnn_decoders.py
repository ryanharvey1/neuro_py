import unittest
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import neuro_py as npy

from torch import nn


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')
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
    def test_train_model_pipeline(self):
        """Test the decoders.pipeline.train_model function."""
        DEVICE = set_device()
        SEED = set_seed(seed=0, seed_torch=True)
        
        BEHAVIORAL_VARS = [
            'pv_x', 'pv_y', 'pv_speed', 'pv_dir', 'pv_dir_cos', 'pv_dir_sin',
        ]

        bins_before = 0
        bins_current = 1
        bins_after = 0
        
        predict_bv = [BEHAVIORAL_VARS.index('pv_x'), BEHAVIORAL_VARS.index('pv_y')]
        decoder_type = ['MLP', 'MLPOld', 'LSTM', 'M2MLSTM', 'NDT'][0]
        
        N_TRIALS = 50
        TRIAL_LENGTH = 10
        N_NEURONS = 100
        partitions = [
            (
                [pd.DataFrame(np.random.rand(TRIAL_LENGTH, N_NEURONS)) for _ in range(N_TRIALS)],  # nsv_train
                np.random.rand(N_TRIALS, TRIAL_LENGTH, len(predict_bv)),  # bv_train
                [pd.DataFrame(np.random.rand(TRIAL_LENGTH, N_NEURONS)) for _ in range(N_TRIALS)],  # nsv_val
                np.random.rand(N_TRIALS, TRIAL_LENGTH, len(predict_bv)),  # bv_val
                [pd.DataFrame(np.random.rand(TRIAL_LENGTH, N_NEURONS)) for _ in range(N_TRIALS)],  # nsv_test
                np.random.rand(N_TRIALS, TRIAL_LENGTH, len(predict_bv)),  # bv_test
            )  # nsv_train, bv_train, nsv_val, bv_val, nsv_test, bv_test
        ]
        
        hyperparams = dict(
            batch_size=512 * 8,
            num_workers=5,
            model=decoder_type,
            model_args=dict(
                in_dim=None,
                out_dim=len(predict_bv),
                hidden_dims=[512, 512, .2, 512],
                args=dict(
                    clf=False,
                    activations=nn.CELU,
                    criterion=F.mse_loss,
                    epochs=10,
                    lr=3e-2,
                    base_lr=1e-2,
                    max_grad_norm=1.,
                    iters_to_accumulate=1,
                    weight_decay=1e-2,
                    num_training_batches=None,
                    scheduler_step_size_multiplier=2,
                )
            ),
            behaviors=predict_bv,
            bins_before=bins_before,
            bins_current=bins_current,
            bins_after=bins_after,
            device=DEVICE,
            seed=SEED
        )
        
        RESULTS_PATH = None
        
        bv_preds_folds, bv_models_folds, norm_params_folds, metrics_folds = \
            npy.ensemble.decode.pipeline.train_model(
                partitions, hyperparams, resultspath=RESULTS_PATH,
                stop_partition=None,
            )
        
        # Example assertions to validate outputs (adjust based on actual implementation)
        self.assertIsInstance(bv_preds_folds, list)
        self.assertIsInstance(bv_models_folds, list)
        self.assertIsInstance(norm_params_folds, list)
        self.assertIsInstance(metrics_folds, dict)

        # check the length of the outputs
        self.assertEqual(len(bv_preds_folds), 1)
        self.assertEqual(len(bv_models_folds), 1)
        self.assertEqual(len(norm_params_folds), 1)

        # check the shape of the outputs
        self.assertEqual(bv_preds_folds[0].shape, (N_TRIALS * TRIAL_LENGTH, len(predict_bv)))
        self.assertEqual(bv_models_folds[0].main[0].in_features, N_NEURONS)
        self.assertEqual(bv_models_folds[0].main[-1].out_features, len(predict_bv))
        self.assertEqual(len(norm_params_folds[0]), 4)


if __name__ == '__main__':
    unittest.main()
