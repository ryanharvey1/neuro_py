from neuro_py.ensemble.replay import PairwiseBias
import numpy as np

def test_bayesian_replay():
    pass
    
def test_bias_matrix():
    # set random seed for reproducibility
    np.random.seed(0)

    # test significant intervals

    transformer = PairwiseBias()

    n_spikes = 1_000
    n_neurons = 30
    spk_1 = np.sort(np.random.rand(n_spikes))
    task_spikes = []
    task_neurons = []
    for i_neurons in range(n_neurons):
        task_spikes.append(spk_1 + i_neurons * 0.01)
        task_neurons.append(np.ones(n_spikes) * i_neurons)

    task_spikes = np.concatenate(task_spikes)
    task_neurons = np.concatenate(task_neurons)

    # sort the task_spikes
    idx = np.argsort(task_spikes)
    task_spikes = task_spikes[idx]
    task_neurons = task_neurons[idx]

    n_spikes = 200
    spk_1 = np.sort(np.random.rand(n_spikes))
    post_spikes = []
    post_neurons = []
    for i_neurons in range(n_neurons):
        post_spikes.append(spk_1 + i_neurons * 0.01)
        post_neurons.append(np.ones(n_spikes) * i_neurons)

    post_spikes = np.concatenate(post_spikes)
    post_neurons = np.concatenate(post_neurons)

    # sort the post_spikes
    idx = np.argsort(post_spikes)
    post_spikes = post_spikes[idx]
    post_neurons = post_neurons[idx]

    post_intervals = np.array([[0, 0.5], [0.5, 1], [1, 1.5]])

    z_score, p_value, cosine_val = transformer.fit_transform(
        task_spikes, task_neurons, post_spikes, post_neurons, post_intervals
    )
    print(f"Z-score: {z_score}, P-value: {p_value}")

    assert z_score.size == post_intervals.shape[0]
    assert p_value.size == post_intervals.shape[0]
    assert z_score.size == p_value.size
    assert (p_value < 0.05).all()


    # test non significant intervals

    n_spikes = 200
    spk_1 = np.sort(np.random.rand(n_spikes))
    post_spikes = []
    post_neurons = []
    for i_neurons in range(n_neurons):
        post_spikes.append(spk_1 + np.sort(np.random.rand(n_spikes)))
        post_neurons.append(np.ones(n_spikes) * i_neurons)

    post_spikes = np.concatenate(post_spikes)
    post_neurons = np.concatenate(post_neurons)

    z_score, p_value, cosine_val = transformer.transform(
        post_spikes, post_neurons, post_intervals
    )
    print(f"Z-score: {z_score}, P-value: {p_value}")

    assert z_score.size == post_intervals.shape[0]
    assert p_value.size == post_intervals.shape[0]
    assert z_score.size == p_value.size
    assert (p_value > 0.05).all()