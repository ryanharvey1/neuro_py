import copy
import os
import random

import numpy as np
import bottleneck as bn
import lightning.pytorch as pl
import sklearn
import torch
import zlib

from .mlp import MLP  # noqa
from .lstm import LSTM  # noqa
from .m2mlstm import M2MLSTM, NSVDataset  # noqa
from .transformer import NDT  # noqa


def seed_worker(worker_id):
    """
    DataLoader will reseed workers following randomness in
    multi-process data loading algorithm.

    Args:
        worker_id: integer
            ID of subprocess to seed. 0 means that
            the data will be loaded in the main process
            Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_spikes_with_history(neural_data, bins_before, bins_after, bins_current=1):
    """
    Create the covariate matrix of neural activity.

    Parameters
    ----------
    neural_data : numpy.ndarray
        A matrix of size "number of time bins" x "number of neurons",
        representing the number of spikes in each time bin for each neuron.
    bins_before : int
        How many bins of neural data prior to the output are used for decoding.
    bins_after : int
        How many bins of neural data after the output are used for decoding.
    bins_current : int, optional
        Whether to use the concurrent time bin of neural data for decoding.
        Default is 1.

    Returns
    -------
    numpy.ndarray
        A matrix of size "number of total time bins" x "number of surrounding
        time bins used for prediction" x "number of neurons".
        For every time bin, there are the firing rates of all neurons from the
        specified number of time bins before (and after).
    """
    num_examples = neural_data.shape[0]  # Number of total time bins we have neural data for
    num_neurons = neural_data.shape[1]   # Number of neurons
    surrounding_bins = bins_before + bins_after + bins_current  # Number of surrounding time bins used for prediction
    X = np.zeros([num_examples, surrounding_bins, num_neurons])  # Initialize covariate matrix with zeros

    # Loop through each time bin, and collect the spikes occurring in surrounding time bins
    # Note: The first "bins_before" and last "bins_after" rows of X will remain filled with zeros,
    # since they don't get filled below due to insufficient preceding or succeeding bins.
    for i in range(num_examples - bins_before - bins_after):
        start_idx = i
        end_idx = start_idx + surrounding_bins
        X[i + bins_before] = neural_data[start_idx:end_idx]

    return X

def _get_trial_spikes_with_no_overlap_history(X, bins_before, bins_after, bins_current):
    nonoverlap_trial_covariates = []
    if X.ndim == 2:
        X_cov = get_spikes_with_history(
            X, bins_before, bins_after, bins_current)
        nonoverlap_trial_covariates.append(X_cov)
    else:
        for X_trial in X:
            X_cov = get_spikes_with_history(
                np.asarray(X_trial), bins_before, bins_after, bins_current
            )
            nonoverlap_trial_covariates.append(X_cov)
    return nonoverlap_trial_covariates

def format_trial_segs_nsv(
        nsv_train_normed, nsv_rest_normed, bv_train, bv_rest, predict_bv,
        bins_before=0, bins_current=1, bins_after=0, 
    ):# -> tuple[NDArray, list, ndarray[Any, dtype], list, Any | nda...:
    is_2D = nsv_train_normed[0].ndim == 1
    # Format for RNNs: covariate matrix including spike history from previous bins
    X_train = np.concatenate(_get_trial_spikes_with_no_overlap_history(
        nsv_train_normed, bins_before, bins_after, bins_current))
    X_rest = []
    for nsv_feats in nsv_rest_normed:
        X_feats = np.concatenate(_get_trial_spikes_with_no_overlap_history(
            nsv_feats, bins_before, bins_after, bins_current))
        X_rest.append(X_feats)

    # each "neuron / time" is a single feature
    X_flat_train = X_train.reshape(
        X_train.shape[0], (X_train.shape[1] * X_train.shape[2]))
    X_flat_rest = []
    for X_feat in X_rest:
        X_flat_feat = X_feat.reshape(
            X_feat.shape[0], (X_feat.shape[1] * X_feat.shape[2]))
        X_flat_rest.append(X_flat_feat)

    bv_train = bv_train if not is_2D else [bv_train]
    y_train = np.concatenate(bv_train)
    y_train = y_train[:, predict_bv]
    y_rest = []
    for bv_y in bv_rest:
        bv_y = bv_y if not is_2D else [bv_y]
        y = np.concatenate(bv_y)
        y = y[:, predict_bv]
        y_rest.append(y)

    return X_train, X_rest, X_flat_train, X_flat_rest, y_train, y_rest

def zscore_trial_segs(train, rest_feats=None, normparams=None):
    is_2D = train[0].ndim == 1
    concat_train = train if is_2D else np.concatenate(train)
    train_mean = normparams['X_train_mean'] if normparams is not None else bn.nanmean(concat_train, axis=0)
    train_std = normparams['X_train_std'] if normparams is not None else bn.nanstd(concat_train, axis=0)

    train_notnan_cols = train_std != 0
    train_nan_cols = ~train_notnan_cols
    if is_2D:
        normed_train = np.divide(train-train_mean, train_std, where=train_notnan_cols)
        normed_train.loc[:, train_nan_cols] = 0        
    else:
        normed_train = np.empty_like(train)
        for i, nsvstseg in enumerate(train):
            zscored = np.divide(nsvstseg-train_mean, train_std, where=train_notnan_cols)
            zscored.loc[:, train_nan_cols] = 0
            normed_train[i] = zscored

    normed_rest_feats = []
    if rest_feats is not None:
        for feats in rest_feats:
            if is_2D:
                normed_feats = np.divide(feats-train_mean, train_std, where=train_notnan_cols)
                normed_feats.loc[:, train_nan_cols] = 0
                normed_rest_feats.append(normed_feats)
            else:
                normed_feats = np.empty_like(feats)
                for i, trialSegROI in enumerate(feats):
                    zscored = np.divide(feats[i]-train_mean, train_std, where=train_notnan_cols)
                    zscored.loc[:, train_nan_cols] = 0
                    normed_feats[i] = zscored
                normed_rest_feats.append(normed_feats)

    return normed_train, normed_rest_feats, dict(
        X_train_mean=train_mean, X_train_std=train_std,
        X_train_notnan_mask=train_notnan_cols,
    )

def normalize_format_trial_segs(nsv_train, nsv_rest, bv_train, bv_rest, predict_bv=[4,5], bins_before=0, bins_current=1, bins_after=0, normparams=None):
    nsv_train_normed, nsv_rest_normed, norm_params = zscore_trial_segs(nsv_train, nsv_rest, normparams)

    (X_train, X_rest, X_flat_train, X_flat_rest, y_train, y_rest
    ) = format_trial_segs_nsv(
        nsv_train_normed, nsv_rest_normed, bv_train, bv_rest, predict_bv,
        bins_before, bins_current, bins_after)

    #Zero-center outputs
    y_train_mean = normparams['y_train_mean'] if normparams is not None else np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean
    y_centered_rest = []
    for y in y_rest:
        y_centered_rest.append(y - y_train_mean)

    norm_params['y_train_mean'] = y_train_mean

    return X_train, X_flat_train, y_train, tuple(zip(X_rest, X_flat_rest, y_centered_rest)), norm_params

def minibatchify(
        Xtrain, ytrain, Xval, yval, Xtest, ytest, seed=0,
        batch_size=128, num_workers=5, modeltype='MLP'
    ):
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtrain).type(torch.float32),
        torch.from_numpy(ytrain).type(torch.float32))
    val = torch.utils.data.TensorDataset(
        torch.from_numpy(Xval).type(torch.float32), 
        torch.from_numpy(yval).type(torch.float32))
    test = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtest).type(torch.float32),
        torch.from_numpy(ytest).type(torch.float32))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
        drop_last=(modeltype=='LSTM'), worker_init_fn=seed_worker,
        generator=g_seed)

    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        drop_last=(modeltype=='LSTM'), worker_init_fn=seed_worker,
        generator=g_seed)
    
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        drop_last=(modeltype=='LSTM'), worker_init_fn=seed_worker,
        generator=g_seed)

    return train_loader, val_loader, test_loader

def normalize_labels(y_train, y_val, y_test):
    # map labels to integers in [0, n_classes)
    uniq_labels = np.unique(np.concatenate((y_train, y_val, y_test)))
    n_classes = len(uniq_labels)
    uniq_labels_idx_map = dict(zip(uniq_labels, range(n_classes)))
    y_train = np.vectorize(lambda v: uniq_labels_idx_map[v])(y_train)
    y_val = np.vectorize(lambda v: uniq_labels_idx_map[v])(y_val)
    y_test = np.vectorize(lambda v: uniq_labels_idx_map[v])(y_test)
    return (y_train, y_val, y_test), n_classes

def create_model(hyperparams):
    decoder = eval(f"{hyperparams['model']}")
    model = decoder(**hyperparams['model_args'])
    if 'LSTM' in hyperparams['model']:
        model.init_hidden(hyperparams['batch_size'])
        model.hidden_state = model.hidden_state.to(hyperparams['device'])
        model.cell_state = model.cell_state.to(hyperparams['device'])
    return decoder, model

def preprocess_data(hyperparams, ohe, nsv_train, nsv_val, nsv_test, bv_train, bv_val, bv_test, foldnormparams=None):
    bins_before = hyperparams['bins_before']
    bins_current = hyperparams['bins_current']
    bins_after = hyperparams['bins_after']
    if hyperparams['model'] != 'M2MLSTM':
        (
            X_cov_train, X_flat_train, y_train,
            ((X_cov_val, X_flat_val, y_val), (X_cov_test, X_flat_test, y_test)),
            fold_norm_params
        ) = normalize_format_trial_segs(
            nsv_train, (nsv_val, nsv_test),
            bv_train, (bv_val, bv_test), predict_bv=hyperparams['behaviors'],
            bins_before=bins_before, bins_current=bins_current,
            bins_after=bins_after, normparams=foldnormparams)
        X_train = X_cov_train if hyperparams['model'] == 'LSTM' else X_flat_train
        X_val = X_cov_val if hyperparams['model'] == 'LSTM' else X_flat_val
        X_test = X_cov_test if hyperparams['model'] == 'LSTM' else X_flat_test

        if hyperparams['model_args']['args']['clf']:
            (y_train, y_val, y_test), n_classes = normalize_labels(y_train, y_val, y_test)
            y_train = ohe.fit_transform(y_train).toarray()
            y_val = ohe.transform(y_val).toarray()
            y_test = ohe.transform(y_test).toarray()
            hyperparams['model_args']['out_dim'] = n_classes
            fold_norm_params['ohe'] = ohe

        train_loader, val_loader, test_loader = minibatchify(
            X_train, y_train, X_val, y_val, X_test, y_test,
            seed=hyperparams['seed'], batch_size=hyperparams['batch_size'],
            num_workers=hyperparams['num_workers'], modeltype=hyperparams['model'])
        hyperparams['model_args']['in_dim'] = X_train.shape[-1]
    else:
        y_train = [np.stack(y)[:, hyperparams['behaviors']] for y in bv_train]
        nbins_per_tseg = [len(y) for y in y_train]  # number of time bins in each trial
        tseg_bounds_train = np.cumsum([0] + nbins_per_tseg)
        y_val = [np.stack(y)[:, hyperparams['behaviors']] for y in bv_val]
        nbins_per_tseg = [len(y) for y in y_val]
        tseg_bounds_val = np.cumsum([0] + nbins_per_tseg)
        y_test = [np.stack(y)[:, hyperparams['behaviors']] for y in bv_test]
        nbins_per_tseg = [len(y) for y in y_test]
        tseg_bounds_test = np.cumsum([0] + nbins_per_tseg)

        (
            _, X_flat_train, y_train,
            ((_, X_flat_val, y_val), (_, X_flat_test, y_test)),
            fold_norm_params
        ) = normalize_format_trial_segs(
            nsv_train, (nsv_val, nsv_test),
            bv_train, (bv_val, bv_test, bv_test), predict_bv=hyperparams['behaviors'],
            bins_before=bins_before, bins_current=bins_current,
            bins_after=bins_after, normparams=foldnormparams)
        X_train = X_flat_train
        X_val = X_flat_val
        X_test = X_flat_test

        if hyperparams['model_args']['args']['clf']:
            (y_train, y_val, y_test), n_classes = normalize_labels(y_train, y_val, y_test)
            y_train = ohe.fit_transform(y_train).toarray()
            y_val = ohe.transform(y_val).toarray()
            y_test = ohe.transform(y_test).toarray()
            hyperparams['model_args']['out_dim'] = n_classes
            fold_norm_params['ohe'] = ohe

        X_train_tsegs, y_train_tsegs = [], []
        X_val_tsegs, y_val_tsegs = [], []
        X_test_tsegs, y_test_tsegs = [], []
        for i in range(1, len(tseg_bounds_train)):
            X_train_tsegs.append(X_train[tseg_bounds_train[i-1]:tseg_bounds_train[i]])
            y_train_tsegs.append(y_train[tseg_bounds_train[i-1]:tseg_bounds_train[i]])
        for i in range(1, len(tseg_bounds_val)):
            X_val_tsegs.append(X_val[tseg_bounds_val[i-1]:tseg_bounds_val[i]])
            y_val_tsegs.append(y_val[tseg_bounds_val[i-1]:tseg_bounds_val[i]])
        for i in range(1, len(tseg_bounds_test)):
            X_test_tsegs.append(X_test[tseg_bounds_test[i-1]:tseg_bounds_test[i]])
            y_test_tsegs.append(y_test[tseg_bounds_test[i-1]:tseg_bounds_test[i]])

        X_train, y_train = X_train_tsegs, y_train_tsegs
        X_val, y_val = X_val_tsegs, y_val_tsegs
        X_test, y_test = X_test_tsegs, y_test_tsegs

        train_dataset = NSVDataset(X_train, y_train)
        val_dataset = NSVDataset(X_val, y_val)
        test_dataset = NSVDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, num_workers=hyperparams['num_workers'],
            batch_size=1
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, num_workers=hyperparams['num_workers'],
            batch_size=1
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=False, num_workers=hyperparams['num_workers'],
            batch_size=1
        )
        hyperparams['model_args']['in_dim'] = X_train[0].shape[-1]

    return (X_train, y_train, X_val, y_val, X_test, y_test), (train_loader, val_loader, test_loader), fold_norm_params

def evaluate_model(hyperparams, ohe, predictor, X_test, y_test):
    if hyperparams['model'] == 'M2MLSTM':
        out_dim = hyperparams['model_args']['out_dim']
        bv_preds_fold = [predictor(torch.from_numpy(X.reshape(1, *X.shape)).type(torch.float32)) for X in X_test]
        bv_preds_fold = np.vstack([bv.squeeze().detach().cpu().numpy().reshape(-1, out_dim) for bv in bv_preds_fold])
    else:
        bv_preds_fold = predictor(torch.from_numpy(X_test).type(torch.float32))
        bv_preds_fold = bv_preds_fold.detach().cpu().numpy()

    bv_preds_fold = copy.deepcopy(bv_preds_fold)

    logits = bv_preds_fold
    labels = np.vstack(y_test)
    if hyperparams['model_args']['args']['clf']:
        logits = ohe.inverse_transform(logits)
        labels = ohe.inverse_transform(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, logits)
        metrics = dict(accuracy=accuracy)
        bv_preds_fold = logits
    else:
        coeff_determination = sklearn.metrics.r2_score(labels, logits, multioutput='variance_weighted')
        rmse = sklearn.metrics.root_mean_squared_error(labels, logits)
        metrics = dict(coeff_determination=coeff_determination, rmse=rmse)
    return metrics, bv_preds_fold

def shuffle_nsv_intrialsegs(nsv_trialsegs):
    nsv_shuffled_intrialsegs = []
    for nsv_tseg in nsv_trialsegs:
        # shuffle the data
        nsv_shuffled_intrialsegs.append(
            nsv_tseg.sample(frac=1).reset_index(drop=True)
        )
    return np.asarray(nsv_shuffled_intrialsegs, dtype=object)

def train_model(partitions, hyperparams, resultspath=None, stop_partition=None):
    """Generic function to train a DNN model on the given data partitions.

    Parameters
    ----------
    partitions : array-like
        K-fold partitions of the data with the following format:
        [(nsv_train, bv_train, nsv_val, bv_val, nsv_test, bv_test), ...]
        Each element of the list is a tuple of numpy arrays containing the with
        pairs of neural state vectors and behavioral variables for the training,
        validation, and test sets. Each array has the shape
        (ntrials, nbins, nfeats) where nfeats is the number of neurons for the
        neural state vectors and number of behavioral features to be predicted
        for the behavioral variables.
    hyperparams : dict
        Dictionary containing the hyperparameters for the model training. The
        dictionary should contain the following keys:
        - `model`: str, the type of the model to be trained. Multi-layer
            Perceptron (MLP), Long Short-Term Memory (LSTM), many-to-many LSTM
            (M2MLSTM), Transformer (NDT).
        - `model_args`: dict, the arguments to be passed to the model
            constructor. The arguments should be in the format expected by the
            model constructor.
            - `in_dim`: The number of input features.
            - `out_dim`: The number of output features.
            - `hidden_dim`: The number of hidden units each hidden layer of the
                model. Can also take float values to specify the dropout rate.
                For LSTM and M2MLSTM, it should be a tuple of the hidden size,
                the number of layers, and the dropout rate.
            - `args`:
                - `clf`: If True, the model is a classifier; otherwise, it is a
                    regressor.
                - `activations`: The activation functions for each layer.
                - `criterion`: The loss function to optimize.
                - `epochs`: The number of complete passes through the training
                    dataset.
                - `lr`: Controls how much to change the model in response to the
                    estimated error each time the model weights are updated. A
                    smaller value ensures stable convergence but may slow down
                    training, while a larger value speeds up training but risks
                    overshooting.
                - `base_lr`: The initial learning rate for the learning rate
                    scheduler.
                - `max_grad_norm`: The maximum norm of the gradients.
                - `iters_to_accumulate`: The number of iterations to accumulate
                    gradients.
                - `weight_decay`: The L2 regularization strength.
                - `num_training_batches`: The number of training batches. If
                    None, the number of batches is calculated based on the batch
                    size and the length of the training data.
                - `scheduler_step_size_multiplier`: The multiplier for the
                    learning rate scheduler step size. Higher values lead to
                    faster learning rate decay.
        - `bins_before`: int, the number of bins before the current bin to
            include in the input data.
        - `bins_current`: int, the number of bins in the current time bin to
            include in the input data.
        - `bins_after`: int, the number of bins after the current bin to include
            in the input data.
        - `behaviors`: list, the indices of the columns of behavioral features
            to be predicted. Selected behavioral variable must have homogenous
            data types across all features (continuous for regression and
            categorical for classification)
        - `batch_size`: int, the number of training examples utilized in one
            iteration. Larger batch sizes offer stable gradient estimates but
            require more memory, while smaller batches introduce noise that can
            help escape local minima.
        - `num_workers`: int, The number of parallel processes to use for data
            loading. Increasing the number of workers can speed up data loading
            but may lead to memory issues. Too many workers can also slow down
            the training process due to contention for resources.
        - `device`: str, the device to use for training. Should be 'cuda' or
            'cpu'.
        - `seed`: int, the random seed for reproducibility.
    resultspath : str or None, optional
        Path to the directory where the trained models and logs will be saved.
    stop_partition : int, optional
        Index of the partition to stop training at. Only useful for debugging,
        by default None

    Returns
    -------
    tuple
        Tuple containing the predicted behavioral variables for each fold,
        the trained models for each fold, the normalization parameters for each
        fold, and the evaluation metrics for each fold.
    """
    ohe = sklearn.preprocessing.OneHotEncoder()
    bv_preds_folds = []
    bv_models_folds = []
    norm_params_folds = []
    metrics_folds = dict() # dict with keys 'accuracy', 'coeff_determination', 'rmse' and values of length number of folds
    for i, (nsv_train, bv_train, nsv_val, bv_val, nsv_test, bv_test) in enumerate(partitions):
        # shuffle nsv bins in between tsegs to generate baseline distribution for vector dev plots
        preprocessed_data = preprocess_data(hyperparams, ohe, nsv_train, nsv_val, nsv_test, bv_train, bv_val, bv_test)
        (X_train, y_train, X_val, y_val, X_test, y_test), (train_loader, val_loader, test_loader), fold_norm_params = preprocessed_data
        hyperparams['model_args']['args']['num_training_batches'] = len(train_loader)

        decoder, model = create_model(hyperparams)

        hyperparams_cp = copy.deepcopy(hyperparams)
        del hyperparams_cp['model_args']['args']['epochs']
        del hyperparams_cp['model_args']['args']['num_training_batches']
        model_cache_name = zlib.crc32(str(hyperparams_cp).encode('utf-8'))
        best_ckpt_path = None
        if resultspath is not None:
            model_cache_path = os.path.join(resultspath, 'models', str(model_cache_name))
            best_ckpt_name_file = os.path.join(model_cache_path, f'{i}-best_model.txt')
            if os.path.exists(best_ckpt_name_file):
                with open(best_ckpt_name_file, 'r') as f:
                    best_ckpt_path = f.read()

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks = [lr_monitor]
        if resultspath is not None:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                save_top_k=1,
                monitor='val_loss',
                dirpath=model_cache_path,
                filename=f'{i}' + '-{epoch:02d}-{val_loss:.2f}',
            )
            callbacks.append(checkpoint_callback)
        logger = pl.loggers.TensorBoardLogger(
            save_dir='logs',
            name=f'{model_cache_name}-{i}',
        )
        pl.seed_everything(hyperparams['seed'], workers=True)
        trainer = pl.Trainer(
            accelerator=hyperparams['device'], devices=1,
            max_epochs=hyperparams['model_args']['args']['epochs'],
            gradient_clip_val=hyperparams['model_args']['args']['max_grad_norm'],
            accumulate_grad_batches=hyperparams['model_args']['args']['iters_to_accumulate'],
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=False,
            log_every_n_steps=5,
            reload_dataloaders_every_n_epochs=1
        )
        trainer.fit(
            model, train_loader, val_loader,
            ckpt_path=best_ckpt_path
        )
        if resultspath is not None:
            with open(best_ckpt_name_file, 'w') as f:
                f.write(checkpoint_callback.best_model_path)
        model.eval()
        trainer.test(model, test_loader)
        predictor = model if hyperparams['model'] != 'LSTM' else model.predict

        metrics, bv_preds_fold = evaluate_model(hyperparams, ohe, predictor, X_test, y_test)
        bv_preds_folds.append(bv_preds_fold)
        bv_models_folds.append(model)
        norm_params_folds.append(copy.deepcopy(fold_norm_params))
        if hyperparams['model_args']['args']['clf']:
            print('Accuracy:', metrics['accuracy'])
            if 'accuracy' not in metrics_folds:
                metrics_folds['accuracy'] = []
            metrics_folds['accuracy'].append(metrics['accuracy'])
        else:
            coeff_determination = metrics['coeff_determination']
            rmse = metrics['rmse']
            print('Variance weighed avg. coefficient of determination:',
                coeff_determination)
            print('RMSE:', rmse)

            if 'coeff_determination' not in metrics_folds:
                metrics_folds['coeff_determination'] = []
                metrics_folds['rmse'] = []
            metrics_folds['coeff_determination'].append(coeff_determination)
            metrics_folds['rmse'].append(rmse)

        if stop_partition is not None and i == stop_partition:
            break
    return bv_preds_folds, bv_models_folds, norm_params_folds, metrics_folds

def predict_models_folds(partitions, hyperparams, bv_models_folds, foldnormparams):
    ohe = sklearn.preprocessing.OneHotEncoder()
    bv_preds_folds = []
    metrics_folds = dict()
    for i, (nsv_train, bv_train, nsv_val, bv_val, nsv_test, bv_test) in enumerate(partitions):
        preprocessed_data = preprocess_data(hyperparams, ohe, nsv_train, nsv_val, nsv_test, bv_train, bv_val, bv_test, foldnormparams[i])
        (X_train, y_train, X_val, y_val, X_test, y_test), (train_loader, val_loader, test_loader), fold_norm_params = preprocessed_data
        model = bv_models_folds[i]

        model.eval()
        predictor = model if hyperparams['model'] != 'LSTM' else model.predict
        metrics, bv_preds_fold = evaluate_model(hyperparams, ohe, predictor, X_test, y_test)
        bv_preds_folds.append(bv_preds_fold)
        if hyperparams['model_args']['args']['clf']:
            if 'accuracy' not in metrics_folds:
                metrics_folds['accuracy'] = []
            metrics_folds['accuracy'].append(metrics['accuracy'])
        else:
            coeff_determination = metrics['coeff_determination']
            rmse = metrics['rmse']
            if 'coeff_determination' not in metrics_folds:
                metrics_folds['coeff_determination'] = []
                metrics_folds['rmse'] = []
            metrics_folds['coeff_determination'].append(coeff_determination)
            metrics_folds['rmse'].append(rmse)

    return bv_preds_folds, metrics_folds
