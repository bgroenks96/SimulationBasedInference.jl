# Ensemble algorithms

```@meta
CurrentModule = SimulationBasedInference
DocTestSetup = quote
    using SimulationBasedInference
    using SimulationBasedInference: EnsembleInferenceAlgorithm, ensemble_solve
end
```

```@docs; canonical=false
EnIS
```

```@docs; canonical=false
EKS
```

```@docs; canonical=false
ESMDA
```

## Ensemble utility methods

```@docs
get_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int)
```

```@docs
get_transformed_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int)
```

```@docs
get_observables(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int)
```

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
