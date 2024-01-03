# SimulationBasedInference.jl

`SimulationBasedInference.jl` aims to bring together a variety of different methods for *simulation-based inference*, i.e. statistical inference with simulator-like models, in the Julia programming language.

Please note that this package is currently under construction and is not yet ready for general use!

## Introduction
Simulator-type models are ubiquitous in science and engineering.

Most (all?) models require some kind of input, e.g boundary conditions (forcings), physical properties or constants, etc.

Often, these parameters are not fully known *a priori*... but usually we know something!

Bayesian inference provides a natural framework for constraining this uncertainty using observed data:

$$
p(\theta | \mathbf{y}) = \frac{p(\mathbf{y}|\theta)p(\theta)}{p(\mathbf{y})}
$$

The **posterior distribution** $p(\theta | \mathbf{y})$ represents our **best estimate** (with uncertainty) of the unknown parameters $\theta$ after observing $\mathbf{y}$.

## Simulation-based inference

Simulation-bsaed inference (SBI) [1] refers to the problem of performing **statistical inference** (Bayesian or otherwise) of unknown parameters $\theta$ where the forward model $\mathcal{M}$:

$$
y = \mathcal{M}(\theta) + \epsilon
$$

is a dynamical model or physics-based *simulator*.

In the numerical modeling literature, this is often referred to as *data assimilation*.

There are two fundamental challenges with this problem:
1. The model $\mathcal{M}$ is almost always *non-linear* and, in the case of dynamical models, *intractable* (i.e. we cannot write down the solution a priori).
2. Evaluating the forward map $\mathcal{M}(\theta)$ is usually non-trivial, i.e. **computationally expensive** or at least inconvenient.

Thus, classical statistical methods that rely on either analytical or numerical methods to derive the posterior distribution are generally difficult (or impossible) to apply.

### Methods

Although the term "simulation-based inference" is relatively new, the basic problem of statistical inference and uncertainty quantification with dynamical "simulator" models is not. In the field of geoscientific modeling and weather forecasting, the problem is often referred to as *data assimilation*.

The following is a list of methods which are planned to be included in this package:

#### Ensemble/particle algorithms

  - [x] Importance sampling (a.k.a particle batch smoothing)
  - [ ] Particle filtering/sequential Monte Carlo (PF/SMC)
  - [x] Ensemble Kalman sampling and inversion (EKS/EKI) via [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
  - [x] Ensemble multiple data assimilation (ES-MDA)
  - [ ] Adaptive multiple importance sampling (AMIS)
  - [ ] Particle flow filters [4] (PFF)

#### Density estimation
  - [ ] Sequential neural likelihood/posterior estimation (SNLE/SNPE) via [sbi](https://sbi-dev.github.io/sbi/)

#### Hybrid ensemble + emulator
  - [ ] Calibrate, emulate, sample w/ Gaussian Processes [2] (CES-GP)

#### Markov Chain Monte Carlo
  - [ ] Affine Invariant MCMC [5] (a.k.a "emcee") via [AffineInvariantMCMC.jl](https://github.com/madsjulia/AffineInvariantMCMC.jl)

## References
[1] Cranmer, Kyle, Johann Brehmer, and Gilles Louppe. "The frontier of simulation-based inference." Proceedings of the National Academy of Sciences 117.48 (2020): 30055-30062.

[2] Cleary, Emmet, et al. "Calibrate, emulate, sample." Journal of Computational Physics 424 (2021): 109716.

[3] Cornuet, Jean‐Marie, et al. "Adaptive multiple importance sampling." Scandinavian Journal of Statistics 39.4 (2012): 798-812.

[4] Hu, Chih‐Chi, and Peter Jan Van Leeuwen. "A particle flow filter for high‐dimensional system applications." Quarterly Journal of the Royal Meteorological Society 147.737 (2021): 2352-2374.

[5] Goodman, Jonathan, and Jonathan Weare. "Ensemble samplers with affine invariance." Communications in applied mathematics and computational science 5.1 (2010): 65-80.
