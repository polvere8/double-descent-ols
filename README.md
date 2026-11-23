# Sample-wise Double Descent in Minimum-Norm Least Squares

A numerical study of **double descent** in the simplest model where it appears: ordinary least squares (OLS). The number of features $d$ is fixed while the number of training points $n$ grows, using the **minimum-norm solution** when the system is underdetermined ($n < d$).

The test error behaves counter-intuitively: adding data *hurts* as $n$ approaches $d$, the error spikes at the interpolation threshold $n = d$, and then falls again.

![Monte Carlo: test error spikes at n = d](figures/monte_carlo_double_descent.png)

## Key findings

From 1,000 Monte Carlo runs per configuration:

- **Position of the peak.** The peak is always at $n = d$ (checked for $d = 30, 40, 60$).
- **Height of the peak.** It scales with the noise variance $\sigma^2$. Without label noise ($\sigma = 0$) there is no peak at all.
- **Feature distribution.** Uniform and Gaussian features give the same phenomenon, with different magnitudes.
- **Mean vs median.** The mean peak is far higher than the median one: a few nearly singular datasets dominate the average.
- **Mechanism.** At $n \approx d$, $X$ is almost square and its smallest singular value $s_{\min}(X) \to 0$. To interpolate noisy labels the solution divides by $s_{\min}$, so the noise is amplified. The condition number $\kappa(X)^2$ peaks at exactly the same $n$ as the test error.

Neither noise alone nor ill-conditioning alone produces the peak; it comes from their interaction.

| Peak height grows with noise | Conditioning peaks at the same point |
|---|---|
| ![noise levels](figures/noise_levels.png) | ![condition number](figures/condition_number.png) |

## What's in the notebook

[`double_descent.ipynb`](double_descent.ipynb):

1. **OLS via the normal equations.** Why we solve with a factorization (LU or Cholesky) instead of inverting.
2. **Minimum-norm solution.** A Lagrangian derivation of $w_{\min} = X^\top (XX^\top)^{-1} y = X^+ y$ for $n < d$.
3. **Conditioning.** Why $\kappa(XX^\top) = \kappa(X^\top X) = \kappa(X)^2$, computed from the SVD of $X$ without forming either product.
4. **Experiments.** A single-run experiment, a 1,000-seed Monte Carlo study, a sweep over noise levels $\sigma \in \{0, 10^{-3}, 0.1, 1, 10, 100\}$, and Gaussian vs uniform features.
5. **Conclusions.**

## How to run

```bash
pip install -r requirements.txt
jupyter notebook double_descent.ipynb
```

The full notebook runs in about one minute on a laptop. All experiments are seeded, so the results are reproducible.
