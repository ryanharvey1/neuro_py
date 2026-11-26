import numpy as np
import pytest

from neuro_py.process.utils import circular_shift


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (3, 3), (5, 1), (1, 5), (1, 1)])
def test_column_shifts_match_independent_rolls(shape):
    matrix = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    original = matrix.copy()
    shifts = np.arange(shape[1]) * 5 - 6
    expected = np.column_stack(
        [np.roll(matrix[:, col], shift) for col, shift in enumerate(shifts)]
    )
    actual = circular_shift(matrix, shifts)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        circular_shift(matrix, np.zeros(shape[1], int)), matrix
    )
    np.testing.assert_array_equal(matrix, original)
    assert actual.dtype == matrix.dtype


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (1, 5), (5, 1)])
def test_row_shifts_match_independent_rolls(shape):
    matrix = np.arange(np.prod(shape)).reshape(shape)
    shifts = np.arange(shape[0]) * 7 - 2
    expected = np.vstack(
        [np.roll(matrix[row], shift) for row, shift in enumerate(shifts)]
    )
    np.testing.assert_array_equal(circular_shift(matrix, shifts), expected)


@pytest.mark.parametrize("unsigned", [False, True])
def test_extreme_shifts_and_noncontiguous_inputs(unsigned):
    matrix = np.arange(48).reshape(6, 8)[::2, ::2]
    shifts = (
        np.array([0, 1, 2**64 - 1, 2**63], dtype=np.uint64)
        if unsigned
        else np.array([0, -1, np.iinfo(np.int64).min, np.iinfo(np.int64).max])
    )
    expected = np.column_stack(
        [
            np.roll(matrix[:, col], int(shift) % matrix.shape[0])
            for col, shift in enumerate(shifts)
        ]
    )
    np.testing.assert_array_equal(circular_shift(matrix, shifts), expected)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
)
@pytest.mark.parametrize("columns", [False, True])
def test_shift_dtypes_work_when_axis_length_exceeds_dtype_range(dtype, columns):
    matrix = np.arange(900).reshape((300, 3) if columns else (3, 300))
    limits = np.iinfo(dtype)
    shifts = np.array([limits.min, 1, limits.max], dtype=dtype)
    # Python integer arithmetic provides an independent, overflow-free oracle.
    expected = np.empty_like(matrix)
    for index, shift in enumerate(shifts):
        indices = [(sample - int(shift)) % 300 for sample in range(300)]
        if columns:
            expected[:, index] = matrix[indices, index]
        else:
            expected[index] = matrix[index, indices]
    np.testing.assert_array_equal(circular_shift(matrix, shifts), expected)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_matrices_preserve_shape_and_dtype(shape):
    matrix = np.empty(shape, dtype=np.float32)
    actual = circular_shift(matrix, np.zeros(shape[1], dtype=int))
    assert actual.shape == shape and actual.dtype == matrix.dtype


@pytest.mark.parametrize(
    "matrix,shifts",
    [
        (np.ones(3), np.array([1])),
        (np.ones((3, 4)), np.ones((1, 4), dtype=int)),
        (np.ones((3, 4)), np.array([0.5, 1, 2, 3])),
        (np.ones((3, 4)), np.zeros(2, dtype=int)),
    ],
)
def test_invalid_dimensions_shifts_and_sizes_are_rejected(matrix, shifts):
    with pytest.raises(ValueError):
        circular_shift(matrix, shifts)
