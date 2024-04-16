# SimulationBasedInference.jl

[![][docs-dev-img]][docs-dev-url]

[docs-dev-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-dev-url]: https://bgroenks96.github.io/SimulationBasedInference.jl/dev/

`SimulationBasedInference.jl` aims to bring together a variety of different methods for *simulation-based inference*, i.e. statistical inference with simulator-like models, in the Julia programming language.

Please note that this package is still very much under construction and things may break or change without prior notice.

If you would like to use this package in your work, please let us know by creating an issue on GitHub or sending an email to [brian.groenke@awi.de](mailto:brian.groenke@awi.de).

## Introduction
Simulator-type models are ubiquitous in science and engineering.

Most (all?) models require some kind of input, e.g boundary conditions (forcings), physical properties or constants, etc.

Often, these parameters are not fully known *a priori*... but usually we know something!

Bayesian inference provides a natural framework for constraining this uncertainty using observed data:

$$
p(\boldsymbol{\theta} | \mathbf{y}) = \frac{p(\mathbf{y}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathbf{y})}
$$

The **posterior distribution** $p(\boldsymbol{\boldsymbol{\theta}} | \mathbf{y})$ represents our best estimate (with uncertainty) of the unknown parameters $\boldsymbol{\theta}$ after observing $\mathbf{y}$.

## Simulation-based inference

Simulation-based inference (SBI) [1] refers to the problem of performing **statistical inference** (Bayesian or otherwise) of unknown parameters $\boldsymbol{\theta}$ where the forward model $\mathcal{M}$:

$$
y = \mathcal{M}(\boldsymbol{\theta}) + \epsilon
$$

is a dynamical model or physics-based *simulator* mapping from parameters to noisy ($\epsilon$) observations.

There are two fundamental challenges with this problem:
1. The forward model $\mathcal{M}$ is very often **nonlinear** and, in the case of dynamical models, **intractable** (i.e. we cannot write down the solution in analytical form).
2. Evaluating the forward map $\mathcal{M}(\boldsymbol{\theta})$ is usually non-trivial, i.e. **computationally expensive** or at least inconvenient.

Thus, classical statistical methods that rely on either analytical or numerical methods to derive the posterior distribution are generally difficult (or impossible) to apply.

### Methods

Although the term "simulation-based inference" is relatively new, the basic problem of statistical inference and uncertainty quantification with dynamical "simulator" models is not. In the field of geoscientific modeling in general and weather forecasting in particular, the problem is often referred to as *data assimilation* [2].

The following is a list of methods which are planned to be included in this package:

#### Ensemble/particle algorithms

  - [x] Importance sampling, a.k.a particle batch smoothing [3] (PBS) and generalized likelihood uncertainty estimation [4] (GLUE)
  - [ ] Particle Filtering/Sequential Monte Carlo [5,6] (PF/SMC)
  - [x] Ensemble Kalman sampling and inversion (EKS/EKI) via [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl) [7]
  - [x] Ensemble smoother with multiple data assimilation [8] (ES-MDA)
  - [ ] Adaptive multiple importance sampling [9] (AMIS) 
  - [ ] Particle flow filters [10] (PFF)

#### Density estimation
  - [x] Sequential neural likelihood/posterior estimation (SNLE/SNPE) via [sbi](https://sbi-dev.github.io/sbi/)

#### Hybrid ensemble + emulator
  - [x] Calibrate, emulate, sample w/ Gaussian Processes [11] (CES-GP)

#### Markov Chain Monte Carlo
  - [x] Affine Invariant MCMC [12] (a.k.a "emcee") via [AffineInvariantMCMC.jl](https://github.com/madsjulia/AffineInvariantMCMC.jl)

#### Package integration
  - [x] [Turing](https://github.com/TuringLang/Turing.jl)
  - [x] [DynamicHMC](https://github.com/tpapp/DynamicHMC.jl)
  - [ ] [Gen](https://github.com/probcomp/Gen.jl)

## References
[1] Cranmer, Kyle, Johann Brehmer, and Gilles Louppe. "The frontier of simulation-based inference." Proceedings of the National Academy of Sciences 117.48 (2020): 30055-30062.

[2] Evensen, Geir, et al. "Data Assimilation Fundamentals.", Springer (2022): https://doi.org/10.1007/978-3-030-96709-3

[3] Margulis, Steven, et al. "A Particle Batch Smoother Approach to Snow Water Equivalent Estimation." J. Hydrometeor. (2015): https://doi.org/10.1175/JHM-D-14-0177.1

[4] Beven, Keith, and Andrew Binley. "GLUE: 20 years on", Hydrol. Process. (2014): https://doi.org/10.1002/hyp.10082

[5] Peter Jan van Leeuwen. "Particle Filtering in Geophysical Systems". Mon. Wea. Rev. (2009): https://doi.org/10.1175/2009MWR2835.1

[6] Chopin, Nicolas, and Omiros Papaspiliopoulos. "An Introduction to Sequential Monte Carlo." Springer (2020): https://doi.org/10.1007/978-3-030-47845-2

[7] Dunbar, Oliver R. A. et al. "EnsembleKalmanProcesses.jl: Derivative-free ensemble-based model calibration." Journal of Open Source Software (2022): https://doi.org/10.21105/joss.04869

[8] Emerick, Alexandre A., and Albert C. Reynolds. "Ensemble smoother with multiple data assimilation." Computers & Geosciences (2013): https://doi.org/10.1016/j.cageo.2012.03.011

[9] Cornuet, Jean‐Marie, et al. "Adaptive multiple importance sampling." Scandinavian Journal of Statistics 39.4 (2012): 798-812.

[10] Hu, Chih‐Chi, and Peter Jan Van Leeuwen. "A particle flow filter for high‐dimensional system applications." Quarterly Journal of the Royal Meteorological Society 147.737 (2021): 2352-2374.

[11] Cleary, Emmet, et al. "Calibrate, emulate, sample." Journal of Computational Physics 424 (2021): 109716.

[12] Goodman, Jonathan, and Jonathan Weare. "Ensemble samplers with affine invariance." Communications in applied mathematics and computational science 5.1 (2010): 65-80.
