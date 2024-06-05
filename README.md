# SimulationBasedInference.jl

[![][docs-dev-img]][docs-dev-url]

[docs-dev-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-dev-url]: https://bgroenks96.github.io/SimulationBasedInference.jl/dev/

`SimulationBasedInference.jl` aims to bring together a variety of different methods for *simulation-based inference* (SBI), i.e. statistical inference with simulator-like models, in the Julia programming language. Although SBI can be applied to both Bayesian and Frequentist inference frameworks, this package focuses on the Bayesian approach.

Please note that this package is still very much under construction and things may break or change without prior notice.

If you would like to use this package in your work, please let us know by creating an issue on GitHub or sending an email to [brian.groenke@awi.de](mailto:brian.groenke@awi.de).

## Introduction
Simulator-type models are ubiquitous in science and engineering.

Most simulator-type models require some kind of input, e.g boundary conditions (forcings), physical properties or constants, etc.

Often, these parameters are not fully known *a priori*... but usually we know something!

Bayesian inference provides a natural framework for constraining this uncertainty using observed data:

$$
p(\boldsymbol{\theta} | \mathbf{y}) = \frac{p(\mathbf{y}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathbf{y})}
$$

The **posterior distribution** $p(\boldsymbol{\boldsymbol{\theta}} | \mathbf{y})$ represents our best estimate (with uncertainty) of the unknown parameters $\boldsymbol{\theta}$ after observing $\mathbf{y}$.

**Simulation-based inference** (SBI) [1] refers to the problem of performing **statistical inference** (Bayesian or otherwise) of unknown parameters $\boldsymbol{\theta}$ where the forward model $\mathcal{M}$:

$$
y = \mathcal{M}(\boldsymbol{\theta}) + \epsilon
$$

is a dynamical model or physics-based *simulator* mapping from parameters to noisy ($\epsilon$) observations.

There are two fundamental challenges with this problem:
1. The forward model $\mathcal{M}$ is very often **nonlinear** and typically has no closed-form solution.
2. Evaluating the forward map $\mathcal{M}(\boldsymbol{\theta})$ is usually non-trivial, i.e. **computationally expensive** or at least inconvenient.

Thus, classical statistical methods that rely on either analytical or numerical methods to derive the posterior distribution are generally difficult (or impossible) to apply.

## Getting started

`SimulationBasedInference` extends the basic [SciML interface](https://docs.sciml.ai/SciMLBase/stable/interfaces/Problems/) to SBI problems.

The first basic building block is `SimulatorForwardProblem`, which wraps either an arbitrary function `f(x)`, where `x` are the input parameters, or another `SciMLProblem`.

The purpose of the `SimulatorForwardProblem` abstraction is to provide a general interface for defining observables and interacting with the simulator.

To illustrate this, we start with the [linear ODE example](examples/linearode/linearode.jl). We first define an `ODEProblem` describing the dynamical model:

```julia
ode_func(u,p,t) = -p[1]*u;
α_true = 0.2
ode_p = [α_true];
tspan = (0.0,10.0);
odeprob = ODEProblem(ode_func, [1.0], tspan, ode_p)
```
We can then define *observables* of the system which should be sampled over the course of the simulation. In this case, we will define a trivial observable that just returns the state of the ODE integrator:
```julia
tsave = tspan[1]+0.1:0.2:tspan[end];
n_obs = length(tsave);
observable = SimulatorObservable(:y, integrator -> integrator.u, tspan[1], tsave, size(odeprob.u0), samplerate=0.01);
```
We are now ready to construct a forward problem from `odeprob` and `observable`:
```julia
forward_prob = SimulatorForwardProblem(odeprob, observable)
```
which can then be solved normally using the `solve` or `init` interface from SciML:
```julia
forward_sol = solve(forward_prob, ode_solver);
y_pred = get_observable(forward_sol, :y)
```
```
╭────────────────────────────────╮
│ 50-element DimArray{Float64,1} │
├────────────────────────────────┴────────────────────────── dims ┐
  ↓ Ti Sampled{Float64} 0.1:0.2:9.9 ForwardOrdered Regular Points
└─────────────────────────────────────────────────────────────────┘
 0.1  0.991057
 0.3  0.961815
 0.5  0.924101
 0.7  0.887867
 ⋮    
 9.5  0.152753
 9.7  0.146763
 9.9  0.141009
```

For the purposes of demonstration, we construct artificial noisy observations from this forward solution:
```julia
true_obs = get_observable(forward_sol, :y)
noisy_obs = true_obs .+ 0.05*randn(rng, n_obs);
```

The next step is defining a `SimulatorInferenceProblem`. We first define priors using [Distributions.jl](https://juliastats.org/Distributions.jl/stable/):
```julia
model_prior = prior(α=Beta(2,2));
noise_scale_prior = prior(σ=Exponential(0.1));
```

then we assign a likelihood:
```julia
lik = IsotropicGaussianLikelihood(observable, noisy_obs, noise_scale_prior);
```

and finally construct the inference problem:
```julia
inference_prob = SimulatorInferenceProblem(forward_prob, ode_solver, model_prior, lik);
```

We can solve the inference problem with one of the simplest methods, *ensemble importance sampling*:
```julia
const rng = Random.MersenneTwister(1234);
enis_sol = solve(inference_prob, EnIS(), ensemble_size=128, rng=rng);
# note that for EnIS, the primary output of inference is the importance weights, and the ensemble reflects only draws from the prior.
importance_weights = get_weights(enis_sol);
prior_ens = get_transformed_ensemble(enis_sol);
```

Note that all ensemble algorithms default to using the `EnsembleThreads` parallelization strategy. We could also use `EnsembleDistributed` or `EnsembleSerial` like so:
```julia
enis_sol = solve(inference_prob, EnIS(), EnsembleDistributed(), ensemble_size=128, rng=rng);
```

To apply a different inference algorithm, such as [EKS](https://clima.github.io/EnsembleKalmanProcesses.jl/dev/ensemble_kalman_sampler/), we need only change the inference solver!
```julia
eks_sol = solve(inference_prob, EKS(), ensemble_size=128, rng=rng);
```

If the `PythonCall` package is installed, we can also use one of the sequential neural density estimator methods from [sbi](https://sbi-dev.github.io/sbi/):
```julia
# this defaults to the SNPE-C method, but this can be changed in the PySNE arguments.
snpe_sol = solve(inference_prob, PySNE(), num_simulations=1000, rng=rng);

# we can also re-use simulations from before instead of generating new ones:
simdata = SimulationBasedInference.sample_ensemble_predictive(enis_sol);
snpe_sol = solve(inference_prob, PySNE(), simdata, rng=rng);
```

## Roadmap

The following is a list of inference methods and/or probablisitc programming frameworks which are planned to be integrated into this package:

#### Ensemble/particle algorithms

  - [x] Importance sampling, a.k.a particle batch smoothing [3] (PBS) and generalized likelihood uncertainty estimation [4] (GLUE)
  - [ ] Particle Filtering and Sequential Monte Carlo [5,6] (PF/SMC)
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
  - [x] [Gen](https://github.com/probcomp/Gen.jl)

## Funding acknowledgements
This work was supported by the Helmholtz Einstein Berlin International Research School in Data Science (grant HIDSS-0001) and the German Academic Exchange Service (grant 57647579).

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
