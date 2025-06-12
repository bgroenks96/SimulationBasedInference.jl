# Particle sampling algorithms

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference
    using SimulationBasedInference: EnsembleInferenceAlgorithm, ensemble_solve
end
```

Particle sampling algorithms are a class of Monte Carlo methods that approximate a target distribution through a set of "particles" or points in the parameter space which are weighted or evolved according to some update rule. The stationary distribution of the particles after some number of updates then typically corresponds to the target. In the setting of Bayesian inference, the target distribution is the posterior.

The simplest form of particle method is [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling) which reweights samples from a so-called *proposal* distribution. If the proposal is taken to be the prior and the weighting function is simply the likelihood, the resulting weighted samples will approximately follow the posterior distribution. This is implemented in `SimulationBasedInference` as the "ensemble importance sampling" (EnIS) algorithm:

```@docs; canonical=false
EnIS
```

where "ensemble" here refers to the collection of particles used for the Monte Carlo approximation.

Naive importance sampling algorithms tend to be very inefficient when applied to high-dimensional or highly nonlinear systems.

For fully Gaussian problems (i.e., Gaussian prior and likelihood), Kalman-type methods are a much more efficient alternative.

`SimulationBasedInference` currently provides two ensemble Kalman-type algorithms; the first is the Ensmeble Kalman Sampler (EKS; Garbuno-Inigo et al. 2020):

```@docs; canonical=false
EKS
```

and the second is the Ensemble Smoother with Multiple Data Assimilation (ES-MDA; Emerick and Reynolds 2013):

```@docs; canonical=false
ESMDA
```

## Ensemble interface methods

All `EnsembleInferenceAlgorithm`s implement the following method interface:

```@docs
get_ensemble(state::EnsembleState)
```

```@docs
isiterative(alg::EnsembleInferenceAlgorithm)
```

```@docs
hasconverged(alg::EnsembleInferenceAlgorithm, state::EnsembleState)
```

```@docs
initialstate(
    alg::EnsembleInferenceAlgorithm,
    prior::AbstractSimulatorPrior,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obscov::AbstractMatrix;
    rng::AbstractRNG
)
```

```@docs
ensemblestep!(::EnsembleSolver{algType}) where {algType} = error("not implemented for alg of type $algType")
```

```@docs
finalize!(solver::EnsembleSolver)
```

`EnsembleState` is an abstract type which must be subtyped by each ensemble algorithm implementation.

```@docs
EnsembleState
```

`EnsembleSolver` acts as a container for all of the common components shared by all ensemble algorithms.

```@docs
EnsembleSolver
```

## Utility methods

The following utility methods are also provided 

```@docs
get_transformed_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int)
```

```@docs
get_observables(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int)
```

Internally, the algorithms use `ensemble_solve` to construct and solve an `EnsembleProblem` over the parameter ensemble.

```@docs
ensemble_solve(
    ens::AbstractMatrix,
    initial_prob::SciMLBase.AbstractSciMLProblem,
    ensalg::SciMLBase.BasicEnsembleAlgorithm,
    dealg::Union{Nothing,SciMLBase.AbstractSciMLAlgorithm},
    param_map;
    iter::Integer=1,
    prob_func,
    output_func,
    pred_func,
    solve_kwargs...
)
```
