import numpy as np


def get_random_sample(sample_size: int, sequence_length: int, rng: np.random.Generator):
    p1 = rng.uniform(size=(sample_size, 1))
    k = rng.uniform(size=(sample_size, sequence_length)) > p1
    alpha = rng.gamma(1, size=(sample_size, 1))
    x = np.where(
        k,
        rng.beta(alpha, 1, size=(sample_size, sequence_length)),
        rng.beta(1, alpha, size=(sample_size, sequence_length)),
    )
    sort_index = np.argsort(-x, axis=1)
    k = np.take_along_axis(k, sort_index, axis=1)
    x = np.take_along_axis(x, sort_index, axis=1)
    return (k, x)


def validate_k(k: np.ndarray):
    assert isinstance(k, np.ndarray)
    assert k.dtype == bool


def compute_cross_entropy(k: np.ndarray, x: np.ndarray, axis: int = -1):
    validate_k(k)
    with np.errstate(divide="ignore"):
        return -np.sum(np.where(k, np.log(x), np.log(1 - x)), axis=axis)


def compute_average_precision(k: np.ndarray):
    validate_k(k)
    n_true_positive = np.cumsum(k, axis=-1)
    n_true = n_true_positive[..., -1]
    n_positive = 1 + np.arange(len(k))
    recall = n_true_positive / n_true
    precision = n_true_positive / n_positive
    return recall[..., 0] * precision[..., 0] + 0.5 * np.sum(
        (precision[..., 1:] + precision[..., :-1])
        * (recall[..., 1:] - recall[..., :-1])
    )


def compute_optimal_thresholds(k: np.ndarray):
    validate_k(k)
    cumsum0 = np.insert(np.cumsum(~k), 0, 0)
    cumsum1 = np.insert(np.cumsum(k), 0, 0)
    # print(f"k: {k.astype(np.int32)}")
    # print(f'cumsum0: {cumsum0}')
    # print(f'cumsum1: {cumsum1}')
    thresholds = k.astype(np.float64)
    for i in range(1, len(thresholds)):
        for j in range(i, -1, -1):
            m0 = cumsum0[i + 1] - cumsum0[j]
            m1 = cumsum1[i + 1] - cumsum1[j]
            x = m1 / (m1 + m0)
            # print(f"thresholds: {thresholds} i: {i} j: {j} m0: {m0} m1: {m1} x: {x}")
            if x < thresholds[j - 1]:
                break
        thresholds[j:] = x
    return thresholds


def generate_binary_sequences(length: int):
    x = np.arange(1 << length)
    y = []
    for i in range(length):
        y.append(x % 2)
        x = x // 2
    return np.stack(y, axis=1).astype(np.bool)


def compute_pareto_frontier(x: np.ndarray, y: np.ndarray):
    assert len(x) == len(y)
    assert np.all(np.diff(x) >= 0)
    frontier = []
    for i in range(len(x)):
        while len(frontier) > 1:
            if (y[frontier[-1]] - y[frontier[-2]]) * (x[i] - x[frontier[-2]]) >= (
                y[i] - y[frontier[-2]]
            ) * (x[frontier[-1]] - x[frontier[-2]]):
                frontier.pop()
            else:
                break
        frontier.append(i)
    return np.array(frontier)


def test_get_random_sample():
    sample_size = 4
    sequence_length = 8
    rng = np.random.default_rng(0)
    k, x = get_random_sample(sample_size, sequence_length, rng)
    assert k.shape == (sample_size, sequence_length), k.shape
    assert x.shape == (sample_size, sequence_length), x.shape
    assert np.all(np.diff(x, axis=1) < 0)


def test_compute_cross_entropy():
    sample_size = 4
    sequence_length = 8
    rng = np.random.default_rng(0)
    k, x = get_random_sample(sample_size, sequence_length, rng)
    l = compute_cross_entropy(k, x)
    assert l.shape == (sample_size,), l.shape
    assert np.all(l > 0)

    l = compute_cross_entropy(
        np.array([0, 1], dtype=bool), np.array([0, 1], dtype=np.float64)
    )

    assert np.isfinite(l), l


def test_compute_optimal_thresholds():
    test_cases = [
        ([0], [0]),
        ([1], [1]),
        ([0, 0], [0, 0]),
        ([1, 1], [1, 1]),
        ([1, 0], [1, 0]),
        ([0, 1], [1 / 2, 1 / 2]),
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, 1], [1 / 3, 1 / 3, 1 / 3]),
        ([0, 1, 0], [1 / 2, 1 / 2, 0]),
        ([0, 1, 1], [2 / 3, 2 / 3, 2 / 3]),
        ([1, 0, 0], [1, 0, 0]),
        ([1, 0, 1], [1, 1 / 2, 1 / 2]),
        ([1, 1, 0], [1, 1, 0]),
        ([1, 1, 1], [1, 1, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0, 1], [1 / 4, 1 / 4, 1 / 4, 1 / 4]),
        ([0, 0, 1, 0], [1 / 3, 1 / 3, 1 / 3, 0]),
        ([0, 0, 1, 1], [1 / 2, 1 / 2, 1 / 2, 1 / 2]),
        ([0, 1, 0, 0], [1 / 2, 1 / 2, 0, 0]),
        ([0, 1, 0, 1], [1 / 2, 1 / 2, 1 / 2, 1 / 2]),
        ([0, 1, 1, 0], [2 / 3, 2 / 3, 2 / 3, 0]),
        ([0, 1, 1, 1], [3 / 4, 3 / 4, 3 / 4, 3 / 4]),
        ([1, 0, 0, 0], [1, 0, 0, 0]),
        ([1, 0, 0, 1], [1, 1 / 3, 1 / 3, 1 / 3]),
        ([1, 0, 1, 0], [1, 1 / 2, 1 / 2, 0]),
        ([1, 0, 1, 1], [1, 2 / 3, 2 / 3, 2 / 3]),
        ([1, 1, 0, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 1], [1, 1, 1 / 2, 1 / 2]),
        ([1, 1, 1, 0], [1, 1, 1, 0]),
        ([1, 1, 1, 1], [1, 1, 1, 1]),
    ]
    for k, x_expected in test_cases:
        x_actual = compute_optimal_thresholds(np.array(k) > 0)
        assert np.all(x_actual == x_expected), (
            f"k: {k} x_actual: {x_actual} x_expected: {x_expected}"
        )


def test_compute_optimal_thresholds_random_sample():
    sample_size = 1 << 18
    sequence_length = 8
    rng = np.random.default_rng(0)
    min_diff = np.inf
    for j in range(1 << 20):
        k, x = get_random_sample(sample_size, sequence_length, rng)
        x_opt = np.stack([compute_optimal_thresholds(k_i) for k_i in k], 0)
        l = compute_cross_entropy(k, x)
        l_opt = compute_cross_entropy(k, x_opt)
        diff = l - l_opt
        min_diff_i = np.amin(diff)
        if min_diff_i < min_diff:
            min_diff = min_diff_i
            print(f"{min_diff_i:.6f}")
        (fail_index,) = np.where(l_opt > l)
        for i in fail_index:
            print("k: ", k[i])
            print("x: ", x[i])
            print("x_opt: ", x_opt[i])
            print("l: ", l[i])
            print("l_opt: ", l_opt[i])
            print()


def test_compute_average_precision():
    test_cases = [
        ([1], 1.0),
        ([1, 0], 1.0),
        ([0, 1], 0.25),
        ([1, 1], 1.0),
    ]
    for k, ap_expected in test_cases:
        ap_actual = compute_average_precision(np.array(k, dtype=bool))
        assert ap_actual == ap_expected, (
            f"k: {k} ap_actual: {ap_actual} ap_expected: {ap_expected}"
        )


def test_generate_binary_sequences():
    s_actual = [generate_binary_sequences(i) for i in range(1, 4)]
    s_expected = [
        [[0], [1]],
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
    ]
    for s_ai, s_ei in zip(s_actual, s_expected):
        assert np.all(s_ai == np.array(s_ei))


def main():
    test_compute_optimal_thresholds()
    test_get_random_sample()
    test_compute_cross_entropy()
    # test_compute_optimal_thresholds_random_sample()
    test_compute_average_precision()
    test_generate_binary_sequences()


if __name__ == "__main__":
    main()
