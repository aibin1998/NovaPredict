# Direct hurdle-MVN decision decoder: pi, mean, covariance only

This experiment maps the Stage-1 hurdle-MVN distribution directly to a point
spectrum. It does not use `pi * mu` as a prediction baseline or learn a
multiplicative correction.

The decoder receives only:

- fragmentation probability `pi`;
- conditional nonzero mean `mu`;
- conditional covariance `Sigma`.

The covariance is passed through its exact low-rank parameterization
`Sigma = diag(s^2) + U U^T`, using `s` and `U` directly. At every cleavage,
`pi` (60), `mu` (60), `s` (60), and rank-4 `U` (240) form 420 scalar inputs.
No expected-intensity, entropy, or derived variance feature is supplied.

The direct prediction is

\[
\widehat{\mathbf Y}
=\operatorname{MaxNorm}\!\left[
\operatorname{Softplus}\{f_\theta(\pi,\mu,s,U)\}\odot\mathbf M
\right].
\]

Stage 1 starts from the shared Stage-1 checkpoint and is jointly fine-tuned so
that the low-weight distribution objective is effective. The loss exactly
follows the original two-stage weighting:

\[
\mathcal L
=\mathcal L_{SA}(\widehat{\mathbf Y},\mathbf Y)
+0.01\left(0.1\mathcal L_{MVN}+0.5\mathcal L_{BCE}\right).
\]

The decoder does not receive the amino-acid sequence or Stage-1 hidden state,
but SA and distribution gradients update Stage 1 during this joint fine-tuning
phase. Checkpoints are selected by validation SA loss.

## Commands

```bash
python -m unittest discover -s tests -v
./launch_gpu4.sh
```
