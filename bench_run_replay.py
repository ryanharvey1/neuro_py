import time
import numpy as np
from neuro_py.ensemble.replay import find_replay_score


def make_test_matrix(nx=10, ny=10, nt=20):
    mat = np.zeros((nx, ny, nt), dtype=float)
    for t in range(nt):
        x = int(round(t * (nx - 1) / (nt - 1)))
        y = int(round(t * (ny - 1) / (nt - 1)))
        mat[x, y, t] = 1.0
    return mat


if __name__ == '__main__':
    # quick demo: small matrix
    nx, ny, nt = 12, 12, 30
    mat = make_test_matrix(nx, ny, nt)
    print(f"Matrix shape: {mat.shape}")

    # single run to show baseline (includes Numba compile)
    t0 = time.perf_counter()
    r, st, sp = find_replay_score(mat, threshold=1, circular=False)
    t1 = time.perf_counter()
    print(f"first_call: r={r}, st={st}, sp={sp}, elapsed={t1-t0:.4f}s")


def sweep_candidate_k(nx=100, ny=100, nt=15, threshold=1, ks=None, runs=5):
    """
    Sweep candidate_k values and report timing and score.

    Parameters
    ----------
    nx, ny, nt : ints
        Matrix dimensions to synthesize.
    threshold : int
        Disk radius.
    ks : list
        List of candidate_k values to test. None means use default.
    runs : int
        Number of timed runs (after warmup) per candidate_k.
    """
    if ks is None:
        ks = [None, 50, 100, 200, 500]

    mat = make_test_matrix(nx, ny, nt)
    M = nx * ny
    print(f"\nSweep matrix: ({nx}, {ny}, {nt}), M={M}, threshold={threshold}")
    print("candidate_k,first_elapsed,median_elapsed,mean_elapsed,score,st,sp")

    # keep a single compile pass by running a no-op first call
    _ = find_replay_score(mat, threshold=threshold, circular=False)

    for k in ks:
        # force full search when k==M by passing candidate_k=M
        candidate_k = k
        t0 = time.perf_counter()
        r, st, sp = find_replay_score(mat, threshold=threshold, circular=False, candidate_k=candidate_k)
        t1 = time.perf_counter()
        first_elapsed = t1 - t0

        # warmup
        for _ in range(2):
            _ = find_replay_score(mat, threshold=threshold, circular=False, candidate_k=candidate_k)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = find_replay_score(mat, threshold=threshold, circular=False, candidate_k=candidate_k)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_elapsed = float(np.median(times))
        mean_elapsed = float(np.mean(times))
        print(f"{str(k)},{first_elapsed:.6f},{median_elapsed:.6f},{mean_elapsed:.6f},{r},{st},{sp}")


if __name__ == '__main__':
    # run sweep for a representative 100x100x15 matrix
    sweep_candidate_k(nx=100, ny=100, nt=15, threshold=1, ks=[None, 100, 200, 500, 1000], runs=3)
