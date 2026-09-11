# Breaking change: posterior-weighted 2D regression

Replace `neuro_py.ensemble.weighted_corr_2d` with
`neuro_py.ensemble.weighted_regression_2d`. The direct Numba entry point is
`weighted_regression_2d_jit`. The old two-dimensional names are removed;
the one-dimensional `weighted_correlation` function is unchanged.

```python
from neuro_py.ensemble import weighted_regression_2d

r2, x_fit, y_fit, vx, vy, mean_x, mean_y = weighted_regression_2d(
    posterior, x_coords=x_bin_centers, y_coords=y_bin_centers,
    time_coords=elapsed_seconds,
)
```

The first result is now the variance-weighted multivariate coefficient of
determination, R² = 1 - SSE/SST, in [0,1], rather than a signed correlation
or the square root of R². All seven return positions are retained. Direction
is in the signed slopes. The final two values are spatial means, not intercepts.
Coordinates must use the same physical units for rotation invariance.

The previous signed axis-average statistic could return zero for exactly
balanced opposite axis correlations and assigned different magnitudes to
equally coherent motion at different angles. Near cancellation only changed
its sign; it did not attenuate its absolute magnitude. The regression uses
total explained spatial variance and avoids both issues.

Recompute observed scores and shuffle distributions from posteriors. There
is no general conversion from the historical signed statistic to R².
Existing numerical cutoffs must be recalibrated. Use an upper-tail test.
Changing from square-root R² to R² preserves within-event shuffle ranks,
but changes raw thresholds and standardized effect sizes.

Float32 and float64 posterior arrays retain their trajectory-array dtype;
moments and supplied coordinates use float64 precision. Integer weights are
promoted to float64. NaNs contribute zero mass. Invalid shapes, nonfinite
coordinates, and negative/infinite weights raise ValueError. Degenerate
spatial or temporal variance gives NaN R².

The tutorial has moved to `tutorials/weighted_regression_2d.ipynb`.
Consumers must migrate imports, output column names, and score labels.
This API correction does not change shuffle generation or calibrate its null.
