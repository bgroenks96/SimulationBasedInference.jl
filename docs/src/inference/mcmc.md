# MCMC inference

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference
end
```

Markov Chain Monte Carlo (MCMC) algorithms are generally regarded as the gold-standard of approximate Bayesian inference. However, they are usually not suitable for simulation-based inference problems due to their inefficiency for expensive forward mdoels and high-dimensional parameter spaces. Nevertheless, it is important to benchmark common SBI algorithms against MCMC on reduced-complexity models/problems in order to assess their performance.

`SimulationBasedInference` does not currently provide direct implementations of any MCMC algorithms. However, the [MCMC](@ref) type allows MCMC samplers from other Julia packages to be used as inference algorithms for [SimulatorInferenceProblem](@ref)s.

```@docs; canonical=false
MCMC
```

## DynamicHMC

[DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) is a popular Julia package that implements [Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434), a gradient-based numerical sampling algorithm. Support for `DynamicHMC` is provided via an extension module that is automatically loaded when `DynamicHMC` is installed in the Julia project environment.

Once installed, the DynamicHMC `NUTS` algorithm can be used as an inference algorithm in the standard `solve` interface:

```julia
import DynamicHMC: NUTS

hmc_sol = solve(inference_prob, MCMC(NUTS()), num_samples=5_000, rng=rng)
```

or iteratively using `init`/`step!`:

```julia
# initialize the sampler; note that this includes the warm-up sampling phase which may take some time
hmc_solver = init(inference_prob, MCMC(NUTS()), num_samples=5_000, rng=rng)
# take one HMC step
step!(hmc_solver)
# run until num_samples is reached
hmc_sol = solve!(hmc_solver)
```

Like with other MCMC algorithms, the output of HMC stored in the `result` field is a `Chains` object as defined by [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl).


## Turing.jl

`SimulationBasedInference` is interoperable with the [Turing](https://turinglang.org) probabilistic programming language (PPL). There are two ways one can use `Turing` with `SimulationBasedInference` problem:

1. To define an arbitrarily complex joint prior for the simulator model parameters (see [prior](@ref))
2. As a provider of both MCMC and variational inference (VI) algorithms; note that VI has not yet been tested with `SimulationBasedInference`.

An example of (1) would be as follows:

```julia
using Turing
using SimulationBasedInference

@model function myprior(c)
    p ~ Beta(1,1)
    x ~ Normal(0,1)
    return ComponentArray(; p, x, c)
end

```

Note that **the prior model must return an `Array`** of parameters to be passed to the simulator. This array need not correspond to the sample space of the prior; i.e., the returned parameter vector can consist of any arbitrary set of values which may be arbitrary functions of the random variables sampled in the prior model, as well as or its inputs (see the `c` parameter in the above example).

The above `Turing` model can then be transformed into a suitable prior distribution for SBI via the `prior` method:

```julia
sbi_prior = prior(myprior(1.0))
inference_prob = SimulatorInferenceProblem(forward_prob, sbi_prior, lik)
```

where `forward_prob` and `lik` correspond to a suitable [SimulatorForwardProblem](@ref) and [SimulatorLikelihood](@ref).

To use a `Turing` sampler for inference, one can simply solve the inference problem again using the `MCMC` wrapper type, e.g.,

```julia
mh_sol = solve(inference_prob, MCMC(MH(), MCMCSerial()), num_chains=4, num_samples=20_000)
```

where `MCMCSerial` can be replaced with other sampling strategies such as `MCMCThreads` or `MCMCDistributed` as described by the [Turing documentation](https://turinglang.org/).

## Gen.jl

TODO
