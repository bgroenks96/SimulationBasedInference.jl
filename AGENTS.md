# SimulationBasedInference.jl - Developer Guide

**Purpose**: Comprehensive reference for LLMs and developers working with the SimulationBasedInference.jl package.

**Quick Navigation**:
- [Architecture & Core Components](#architecture--core-components)
- [API Reference by Symbol Type](#api-reference-by-symbol-type)
- [Code Style & Conventions](#code-style--conventions)
- [Usage Patterns & Examples](#usage-patterns--examples)
- [Development Guidelines](#development-guidelines)
- [Troubleshooting](#troubleshooting)

---

## Overview

`SimulationBasedInference.jl` is a Julia package for **simulation-based inference (SBI)**, focusing on Bayesian statistical inference for dynamical models and physics-based simulators. The package provides a unified interface for performing parameter estimation and uncertainty quantification when the forward model is a complex simulator rather than an analytical expression.

**Key Concept**: Given a simulator $\mathcal{M}$ that maps parameters $\theta$ to observations $y = \mathcal{M}(\theta) + \epsilon$, SBI methods estimate the posterior distribution $p(\theta \mid y)$ without requiring explicit likelihood functions or being constrained by computational cost.

**Package Metadata**:
- **Version**: 0.2.3
- **Julia Compatibility**: ≥ 1.10
- **License**: MIT
- **Maintainer**: Brian Groenke
- **UUID**: `78927d98-f421-490e-8789-96b006983a5c`

---

## Architecture & Core Components

### High-Level Architecture (Layered Design)

```
┌─────────────────────────────────────────┐
│         Inference Algorithms            │  ← MCMC, Ensemble methods (EKS, ES-MDA)
├─────────────────────────────────────────┤
│      SimulatorInferenceProblem          │  ← High-level inference problem interface
├─────────────────────────────────────────┤
│    Forward Problem + Observables        │  ← SimulatorForwardProblem, SimulatorObservable
├─────────────────────────────────────────┤
│         Simulator Interface             │  ← Traits for ForwardMap, Dynamical, Iterative
├─────────────────────────────────────────┤
│      Priors & Likelihoods               │  ← AbstractSimulatorPrior, SimulatorLikelihood
└─────────────────────────────────────────┘
```

### File-to-Component Mapping (For Quick Reference)

| Component | File Path | Primary Types | Key Functions |
|-----------|-----------|---------------|---------------|
| Simulation Data | `src/simulation_data/` | `SimulationData`, `SimulationDataSet` | `store!`, `getinputs`, `getoutputs`, `getoutput`, `getmetadata` |
| Observables | `src/observables.jl` | `Observable`, `SimulatorObservable`, `TimeSampledObservable` | `observe!`, `getvalue`, `coordinates` |
| Simulator Interface | `src/simulator_interface.jl` | `ForwardMap`, `Iterative`, `Dynamical` | `init`, `step!`, `solve!`, `isdone` |
| Priors | `src/priors/` | `AbstractSimulatorPrior`, `NamedProductPrior` | `prior()`, `logprob()`, `forward_map()` |
| Likelihoods | `src/likelihoods/` | `SimulatorLikelihood`, `GaussianLikelihood` | `predictive_distribution()`, `loglikelihood()` |
| Forward Problems | `src/forward_problem.jl` | `SimulatorForwardProblem` | `solve`, `init`, `remake` |
| Inference Problems | `src/inference_problem.jl` | `SimulatorInferenceProblem` | `getprior`, `LogDensityProblems.logdensity` |
| Ensemble Methods | `src/ensembles/` | `EnsembleSolver`, `EKS`, `ESMDA` | `initialstate`, `ensemblestep!`, `hasconverged` |
| MCMC Methods | `src/mcmc/` | `MCMC` | Extension-specific (DynamicHMC, Turing, etc.) |

---

## API Reference by Symbol Type

### Abstract Types (Interfaces)

| Symbol | File | Purpose | Subtypes |
|--------|------|---------|----------|
| `SimulatorInferenceAlgorithm` | `SimulationBasedInference.jl` | Base type for all inference algorithms | `EnsembleInferenceAlgorithm`, `MCMC` |
| `Observable{outputType}` | `observables.jl` | Base type for observables | `SimulatorObservable` |
| `AbstractSimulatorPrior` | `priors/priors.jl` | Base type for prior distributions | `NamedProductPrior` |
| `AbstractLikelihood` | `likelihoods/likelihoods.jl` | Base type for likelihood functions | `SimulatorLikelihood`, `GaussianLikelihood` |
| `SimulationData{inputType, outputType}` | `simulation_data/simulation_data.jl` | Storage for simulation data | `SimulationData`, `SimulationDataSet` |

### Concrete Types (Structs)

| Symbol | File | Fields | Use Case |
|--------|------|--------|----------|
| `SimulatorObservable{N,outputType,funcType,coordsType}` | `observables.jl` | extraction function, output type, coordinates | Define observables for forward problems |
| `NamedProductPrior{distTypes}` | `priors/distributions.jl` | NamedTuple of distributions | Diagonal priors on parameters |
| `SimulatorLikelihood{distType,priorType,obsType,dataType}` | `likelihoods/likelihoods.jl` | name, observable, data, prior | Likelihood with decoupled observables |
| `SimulatorForwardProblem{simType,paramType,seedType,obsTypes,names}` | `forward_problem.jl` | simulator, parameters, observables, rng_seed | Forward simulation problem |
| `SimulatorInferenceProblem{modelPriorType,uType,fwdProbType,fwdSolverType,priorType}` | `inference_problem.jl` | u0, forward_prob, prior, likelihoods | Full Bayesian inference problem |
| `EnsembleSolver{algType,probType,ensalgType,stateType,argTypes,kwargTypes}` | `ensembles/ensemble_solver.jl` | sol, alg, ensalg, state, loglik | Generic ensemble solver |

### Core Functions (Interface Methods)

#### Simulator Interface
```julia
# Trait-based classification
Simulator(::Type{<:Function}) = ForwardMap()
Simulator(::Type{<:SciMLBase.AbstractDEProblem}) = Dynamical()

# Simulation lifecycle
init(simulator, args...; p, tspan, kwargs...)  # Initialize simulation
step!(simulation, args...; kwargs...)           # Advance one step
solve!(simulation)                               # Solve to completion
current_state(simulation)                        # Get current state
current_time(simulation)                         # Get current time (Dynamical only)
isdone(simulation)                               # Check if complete
```

#### Observable Interface
```julia
initialize!(obs::Observable, state)   # Initialize from simulator state
observe!(obs::Observable, state)      # Extract and store observable
getvalue(obs::Observable, T=Any)      # Retrieve observed value
coordinates(obs::Observable)          # Get coordinate tuples
size(obs::Observable)                 # Get output shape
```

#### Prior Interface
```julia
prior(args...; kwargs...)             # Constructor for prior distributions
logprob(prior, x)                     # Log probability density
forward_map(prior, ζ)                 # Map from sample space to parameter space
unconstrained_forward_map(prior, ζ)   # Map from unconstrained to parameter space
```

#### Likelihood Interface
```julia
predictive_distribution(lik::SimulatorLikelihood, args...)  # Build predictive distribution
loglikelihood(lik::SimulatorLikelihood, args...)            # Evaluate log-likelihood
sample_prediction(lik::SimulatorLikelihood, args...)        # Sample from predictive
observable(lik::SimulatorLikelihood)                        # Get associated observable
getprior(lik::SimulatorLikelihood)                          # Get auxiliary parameter prior
```

#### Problem-Solver Interface (SciML-compatible)
```julia
solve(prob::SimulatorForwardProblem, solver)           # Solve forward problem
init(prob::SimulatorInferenceProblem; storage, kwargs...)  # Initialize inference
ensemble_solve(solver::EnsembleSolver)                 # Solve with ensemble method
```

#### LogDensityProblems Interface (for MCMC)
```julia
LogDensityProblems.logdensity(inference_prob, storage=SimulationDataSet())
LogDensityProblems.dimension(ldp::LogDensityProblem)  # Parameter dimension
```

### Algorithm-Specific Methods

#### Ensemble Algorithms
| Method | Purpose | Required For |
|--------|---------|--------------|
| `initialstate(alg, prior, ens, obs, obscov; rng)` | Initialize ensemble state | All ensemble algorithms |
| `ensemblestep!(solver::EnsembleSolver)` | Perform one iteration | Iterative algorithms |
| `hasconverged(alg, state) -> Bool` | Check convergence | Iterative algorithms |
| `get_ensemble(state::EnsembleState) -> Matrix` | Extract current ensemble | All ensemble states |

#### Available Ensemble Algorithms
- **EKS** (Ensemble Kalman Smoother): Single-shot smoothing
- **ES-MDA** (Ensemble Square Root with Multiple Data Assimilation): Iterative data assimilation
- **PBS** (Particle Batch Sampler): Particle filtering approach
- **EnIS** (Ensemble Importance Sampling): Importance-weighted ensembles

---

## Code Style & Conventions

### Naming Conventions

| Pattern | Convention | Examples |
|---------|------------|----------|
| Types (abstract/struct) | PascalCase | `SimulatorForwardProblem`, `NamedProductPrior` |
| Functions | snake_case | `get_observable`, `initialstate`, `ensemblestep!` |
| Mutating functions | snake_case + `!` suffix | `store!`, `observe!`, `remake!` |
| Type parameters | PascalCase or Greek letters | `{T, N}`, `{distType, obsType}` |
| Module-level constants | SCREAMING_SNAKE_CASE | `const SBI = SimulationBasedInference` |

### Type Design Patterns

**Pattern 1: Abstract Base Types Define Interfaces**
```julia
abstract type AbstractSimulatorPrior end
abstract type Observable{outputType<:SimulatorOutput} end
abstract type SimulatorInferenceAlgorithm end
```

**Pattern 2: Parametric Structs for Type Stability**
```julia
struct SimulatorLikelihood{distType,priorType,obsType,dataType} <: AbstractLikelihood
    name::Symbol
    obs::obsType
    data::dataType
    prior::priorType
end
```

**Pattern 3: NamedTuple-Based Composition for Flexibility**
```julia
struct SimulatorForwardProblem{simType,paramType,seedType,obsTypes,names} 
    simulator::simType
    p::paramType
    observables::NamedTuple{names, obsTypes}
    rng_seed::seedType
end
```

### Documentation Style Guidelines

- **Docstrings**: Use Julia's `"""..."""` format with type annotations
- **Mathematical Notation**: LaTeX in comments: `$p(\theta \mid y)$`, `$$\mathcal{M}(\theta)$$`
- **Cross-references**: Use `@ref` syntax for internal documentation links
- **Code Examples**: Provide commented code blocks showing typical usage

### Module Organization Structure

```
SimulationBasedInference.jl/
├── src/
│   ├── SimulationBasedInference.jl    # Main module, exports, __init__()
│   ├── utils.jl                       # Utility functions (with_names, ntreduce)
│   ├── simulation_data.jl            # Data storage abstractions
│   ├── observables.jl                # Observable interface & implementations
│   ├── simulator_interface.jl        # Simulator traits (ForwardMap, Dynamical, Iterative)
│   ├── priors/
│   │   ├── priors.jl                 # Prior definitions & interface
│   │   ├── distributions.jl          # NamedProductPrior implementation
│   │   └── gaussian_approx.jl        # Gaussian approximation methods
│   ├── likelihoods/
│   │   ├── likelihoods.jl            # Likelihood base types
│   │   ├── gaussian_likelihood.jl    # Gaussian variants (Gaussian, Isotropic, Diagonal)
│   │   ├── implicit_likelihood.jl    # Implicit likelihoods
│   │   └── dirac_likelihood.jl       # Dirac/deterministic likelihoods
│   ├── forward_problem.jl            # Forward problem definition
│   ├── forward_solve.jl              # Forward solve interface
│   ├── inference_problem.jl          # Inference problem definition
│   ├── logdensity.jl                 # LogDensityProblems interface
│   ├── ensembles/
│   │   ├── ensembles.jl              # Ensemble module exports
│   │   ├── ensemble_solver.jl        # Generic ensemble solver
│   │   ├── ensemble_utils.jl         # Utility functions (obscov, etc.)
│   │   ├── eks.jl                    # EKS algorithm implementation
│   │   ├── es-mda.jl                 # ES-MDA algorithm implementation
│   │   └── importance_sampling.jl    # Importance sampling utilities
│   ├── mcmc/
│   │   ├── mcmc.jl                   # MCMC exports
│   │   └── mcmc_base.jl              # MCMC base types
│   └── PySBI/                        # Python SBI integration (optional)
│       └── PySBI.jl
├── ext/                              # Julia extension modules
│   ├── SimulationBasedInferenceTuringExt/
│   ├── SimulationBasedInferenceDynamicHMCExt/
│   ├── SimulationBasedInferenceDiffEqBaseExt/
│   ├── SimulationBasedInferenceEmceeExt/
│   └── SimulationBasedInferenceFluxExt/
├── examples/                         # Example workflows
│   ├── linearode/                    # Linear ODE parameter estimation
│   ├── lotka_volterra/               # Predator-prey model inference
│   ├── gaussian2D/                   # Simple 2D Gaussian test case
│   └── ddm/                          # Drift-diffusion model
├── test/                             # Test suite
│   ├── runtests.jl                   # Test runner
│   ├── problem_tests.jl              # Forward/inference problem tests
│   ├── prior_tests.jl                # Prior distribution tests
│   ├── likelihood_tests.jl           # Likelihood function tests
│   ├── observables_tests.jl          # Observable interface tests
│   ├── storage_tests.jl              # Simulation data storage tests
│   └── issues/                       # Regression tests for specific issues
├── docs/                             # Documentation build files
├── Project.toml                      # Package manifest & dependencies
├── Manifest.toml                     # Locked dependency versions
└── AGENTS.md                         # This file (developer guide)
```

### Extension System Pattern

The package uses Julia's extension system for optional dependencies:

**In main module (`src/SimulationBasedInference.jl`):**
```julia
function __init__()
    @require PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d" begin
        using CondaPkg
        include("PySBI/PySBI.jl")
    end
end
```

**Extension modules (in `ext/` directory):**
- `SimulationBasedInferenceTuringExt` - Turing MCMC integration
- `SimulationBasedInferenceDynamicHMCExt` - DynamicHMC support
- `SimulationBasedInferenceDiffEqBaseExt` - DiffEq problem extensions
- `SimulationBasedInferenceEmceeExt` - AffineInvariantMCMC (emcee) support
- `SimulationBasedInferenceFluxExt` - Neural network surrogates

### Re-export Pattern

The main module aggressively re-exports useful namespaces for user convenience:

```julia
@reexport using Bijectors           # Transformations & bijectors
@reexport using ComponentArrays     # Named parameter vectors
@reexport using Distributions       # Probability distributions
@reexport using Statistics          # Basic statistics
@reexport using StatsBase           # Extended statistics
@reexport using DimensionalData     # Structured data with dimensions

@reexport import CommonSolve: init, solve, solve!, step!  # Solver interface
@reexport import LogDensityProblems: logdensity            # MCMC compatibility
```

**Effect**: Users can access these packages directly from `SimulationBasedInference` without explicit `using` statements.

---

## Usage Patterns & Examples

### Complete Workflow (Step-by-Step)

#### Step 1: Define the Forward Model
```julia
# Option A: SciML problem (e.g., ODE)
ode_func(u, p, t) = -p[1] * u
α_true = 0.2
odeprob = ODEProblem(ode_func, [1.0], (0.0, 10.0), [α_true])

# Option B: Simple function
forward_map(θ) = θ^2 + sin(θ[1])
```

#### Step 2: Create Observables
```julia
# Time-sampled observable from ODE integrator
dt = 0.2
tsave = 0.2:dt:10.0
observable = SimulatorObservable(
    integrator -> integrator.u,      # Extraction function
    size(odeprob.u0),                # Output shape
    name=:y,                         # Identifier
    output=TimeSampled(first(tspan), tsave, samplerate=0.01)  # Sampling scheme
)

# Custom scalar observable
magnitude_obs = SimulatorObservable(
    u -> norm(u),                    # Extract magnitude
    (),                              # Scalar output
    name=:magnitude,
    output=TransientObservable()     # Single observation
)
```

#### Step 3: Construct Forward Problem
```julia
forward_prob = SimulatorForwardProblem(odeprob, observable)
# or with multiple observables
forward_prob = SimulatorForwardProblem(odeprob, observable, magnitude_obs)
```

#### Step 4: Generate Synthetic Data (Optional)
```julia
ode_solver = Tsit5()
data_sol = solve(forward_prob, ode_solver)
true_obs = get_observable(data_sol, :y)
noise_scale = 0.05
obs_data = true_obs .+ noise_scale * randn(length(true_obs))
```

#### Step 5: Specify Priors
```julia
# Simple diagonal prior
prior_dist = prior(α=Beta(2, 2), σ=LogNormal(-1, 1))

# With Gaussian approximation (for emulation)
approx = gaussian_approx(prior_dist, method=LaplaceMethod())
```

#### Step 6: Define Likelihoods
```julia
# Gaussian likelihood with default Exponential prior on noise scale
likelihood = GaussianLikelihood(observable, obs_data)

# Custom noise prior
custom_likelihood = GaussianLikelihood(
    observable, 
    obs_data, 
    prior=NamedProductPrior(σ=LogNormal(-1, 0.5))
)

# Isotropic (homoscedastic) vs Diagonal (heteroscedastic)
iso_lik = IsotropicGaussianLikelihood(observable, obs_data)
diag_lik = DiagonalGaussianLikelihood(observable, obs_data)
```

#### Step 7: Build Inference Problem
```julia
inference_prob = SimulatorInferenceProblem(
    forward_prob,           # Forward problem definition
    nothing,                # Forward solver (if needed)
    prior_dist,             # Prior on model parameters
    likelihood              # Likelihood(s)
)

# With metadata
inference_prob = SimulatorInferenceProblem(
    forward_prob,
    nothing,
    prior_dist,
    likelihood;
    metadata=Dict(:description => "Linear ODE inversion")
)
```

#### Step 8: Solve with MCMC
```julia
# Using DynamicHMC (requires extension loaded)
using DynamicHMC
sol = solve(inference_prob, MCMC(DynamicHMC.MetropolisHastings()))

# Analyze results
chain = sol.chain  # MCMCChain object
summarize(chain)   # Posterior statistics
```

#### Step 8 (Alternative): Solve with Ensemble Method
```julia
# Initialize ensemble from prior
n_ensemble = 100
θ_init = sample(prior_dist, n_ensemble)

# Setup ensemble solver
solver = EnsembleSolver(
    inference_prob,
    EKS(),                          # Algorithm: Ensemble Kalman Smoother
    initial_ensemble=θ_init,
    n_iterations=20,
    verbose=true
)

# Solve
sol = solve(solver)

# Extract results
final_ensemble = get_ensemble(sol.state)
```

### Common Code Patterns

#### Creating Priors
```julia
# Univariate priors
prior_α = prior(:α, Beta(2, 2))

# Multivariate (diagonal) prior
prior_dist = prior(α=Beta(2, 2), σ=LogNormal(-1, 1), β=Normal(0, 1))

# Automatic prior from moments
prior_dist = autoprior(mean_vector, stddev_vector; bounds...)

# Transformed Beta from mean/dispersion
β_prior = betadist(mean=0.5, dispersion=10)
```

#### Defining Observables
```julia
# Time-sampled state
obs_state = SimulatorObservable(
    integrator -> integrator.u,
    (n_states,),
    name=:state,
    output=TimeSampled(t0, tsave, samplerate=dt)
)

# Derived quantity (e.g., power spectrum)
obs_spectrum = SimulatorObservable(
    sol -> fft(sol.u),
    (n_freqs,),
    name=:spectrum,
    output=TransientObservable()
)

# Multi-observable setup
observables = (
    state = SimulatorObservable(...),
    magnitude = SimulatorObservable(...)
)
```

#### Working with ComponentArrays
```julia
# Access parameters by name
θ = rand(prior_dist)  # Returns ComponentVector
α_value = θ.α         # Named access
σ_value = θ.σ

# Modify parameters
θ_new = copy(θ)
θ_new.α = 0.5

# Convert to array
θ_array = Array(θ)    # Vector
θ_named = NamedTuple(θ)  # NamedTuple
```

#### Using Bijectors for Constrained Optimization
```julia
# Transform from unconstrained space
ζ_unconstrained = randn(n_params)
θ_constrained = unconstrained_forward_map(prior_dist, ζ_unconstrained)

# Inverse transform
ζ_back = inverse(bijector(prior_dist))(θ_constrained)

# Use in optimization
result = optimize(ζ -> -logprob(prior_dist, unconstrained_forward_map(prior_dist, ζ)), 
                  zeros(n_params), LBFGS())
```

---

## Development Guidelines

### Adding New Inference Algorithms

**For Ensemble Methods:**

1. **Create subtype of `EnsembleInferenceAlgorithm`:**
   ```julia
   struct MyNewAlgorithm{paramType} <: EnsembleInferenceAlgorithm
       params::paramType
   end
   ```

2. **Implement required interface methods:**
   ```julia
   function initialstate(
       alg::MyNewAlgorithm,
       prior::AbstractSimulatorPrior,
       ens::AbstractMatrix,
       obs::AbstractVector,
       obscov::AbstractMatrix;
       rng::AbstractRNG=Random.GLOBAL_RNG
   )
       # Initialize and return state type
   end
   
   function ensemblestep!(solver::EnsembleSolver{<:MyNewAlgorithm})
       # Update ensemble based on algorithm logic
   end
   
   function hasconverged(alg::MyNewAlgorithm, state)
       # Return convergence criterion
   end
   ```

3. **Add constructor in appropriate module file** (e.g., `src/ensembles/my_new_algo.jl`)

4. **Export from main module:**
   ```julia
   export MyNewAlgorithm
   include("ensembles/my_new_algo.jl")
   ```

**For MCMC Methods:**

1. Create extension module in `ext/SimulationBasedInferenceMyMCMCExt/`
2. Implement `solve(inference_prob::SimulatorInferenceProblem, mcmc::MCMC{<:MyAlgorithm})`
3. Add to `Project.toml` weak dependencies and extensions section

### Adding New Observable Types

1. **Subtype `Observable{outputType}`:**
   ```julia
   struct MyCustomObservable{outputType<:SimulatorOutput} <: Observable{outputType}
       # Fields for configuration
       extraction_func::Function
       config::NamedTuple
   end
   ```

2. **Implement required methods:**
   ```julia
   function initialize!(obs::MyCustomObservable, state)
       # Initialize from simulator state
   end
   
   function observe!(obs::MyCustomObservable, state)
       # Extract and store observable value
   end
   
   function getvalue(obs::MyCustomObservable, T=Any)
       # Return observed value
   end
   
   function coordinates(obs::MyCustomObservable)
       # Return coordinate tuples
   end
   ```

3. **Consider output storage type:** `TransientObservable`, `TimeSampledObservable`, or custom

### Adding New Likelihood Types

1. **Subtype `AbstractLikelihood`:**
   ```julia
   struct MyLikelihood{distType,obsType,dataType,priorType} <: AbstractLikelihood
       name::Symbol
       obs::obsType
       data::dataType
       prior::priorType
   end
   ```

2. **Implement `predictive_distribution()` method:**
   ```julia
   function predictive_distribution(lik::MyLikelihood, args...)
       μ = getvalue(lik.obs)
       # Compute predictive distribution based on args
       return MyDistribution(μ, args...)
   end
   ```

3. **Optionally implement custom `loglikelihood()` for efficiency:**
   ```julia
   function loglikelihood(lik::MyLikelihood, args...)
       # Optimized log-likelihood computation
   end
   ```

4. **Add constructor alias if appropriate:**
   ```julia
   MyLikelihood(obs, data; kwargs...) = MyLikelihood(..., obs, data, ...)
   ```

### Adding New Prior Types

1. **Subtype `AbstractSimulatorPrior`:**
   ```julia
   struct MyPrior{paramType} <: AbstractSimulatorPrior
       params::paramType
   end
   ```

2. **Implement interface methods:**
   ```julia
   Base.rand(rng::AbstractRNG, prior::MyPrior)
   logprob(prior::MyPrior, x)
   Bijectors.bijector(prior::MyPrior)  # For constrained optimization
   ```

3. **Add constructor in `priors/distributions.jl`:**
   ```julia
   function my_prior_constructor(args...; kwargs...)
       return MyPrior(...)
   end
   ```

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Type Instability** | Performance warnings, slow execution | Ensure all struct fields have concrete types; avoid `Any` in type parameters |
| **Dimension Mismatch** | `ArgumentError`, shape errors | Check `coordinates(obs)` matches data shape; verify observable output dimensions |
| **Convergence Problems** | MCMC chains not mixing, high R-hat | Try different priors (more informative); increase ensemble size for EKS/ES-MDA |
| **Memory Issues** | Out-of-memory errors, slow performance | Use streaming observables for long simulations; reduce ensemble size |
| **Extension Not Loaded** | `MethodError` for MCMC methods | Ensure extension package is loaded: `using DynamicHMC` (loads extension automatically) |
| **Bijector Errors** | Optimization failures, domain errors | Check prior support bounds; use `unconstrained_forward_map` correctly |

### Debugging Tips

1. **Inspect Parameters:**
   ```julia
   θ = rand(prior_dist)
   @show θ           # View ComponentVector
   @show Array(θ)    # Convert to array
   @show names(prior_dist)  # Get parameter names
   ```

2. **Validate Prior Samples:**
   ```julia
   logprob(prior_dist, θ)  # Should be finite for valid samples
   ```

3. **Check Observable Values Before Inference:**
   ```julia
   forward_sol = solve(forward_prob, solver)
   obs_pred = getvalue(observable)
   @show mean(obs_pred), std(obs_pred)  # Compare with data
   ```

4. **Test with Simple Likelihood First:**
   - Start with `GaussianLikelihood` to verify setup works
   - Gradually add complexity (hierarchical models, custom likelihoods)

5. **Use Verbose Output for Ensemble Methods:**
   ```julia
   solver = EnsembleSolver(..., verbose=true)
   # Monitor log-likelihood and prior at each iteration
   ```

6. **Check LogDensityProblems Interface:**
   ```julia
   ldp = LogDensityProblems.logdensity(inference_prob)
   @show LogDensityProblems.dimension(ldp)  # Should match parameter count
   x = rand(n_params)
   @show LogDensityProblems.logdensity(ldp, x)  # Should be finite
   ```

---

## Dependencies & Compatibility

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| Julia | ≥ 1.10 | Language runtime |
| SciMLBase | 2.x | Problem/solver interface |
| CommonSolve | 0.2 | Generic solve interface |
| Distributions | 0.25 | Probability distributions |
| StatsBase | 0.34 | Statistical functions |
| ComponentArrays | 0.15 | Named parameter vectors |
| Bijectors | 0.13-0.15 | Transformations & bijectors |
| DimensionalData | 0.26-0.30 | Structured data with dimensions |
| LogDensityProblems | 2.x | MCMC compatibility interface |
| Optim | 1.x, 2.x | Optimization routines |
| MCMCChains | 6.x, 7.x | MCMC chain handling |
| PosteriorStats | 0.2-0.4 | Posterior summarization |

### Optional Extensions (Weak Dependencies)

| Extension | Package | Purpose |
|-----------|---------|---------|
| `SimulationBasedInferenceTuringExt` | Turing 0.43 | Turing MCMC samplers |
| `SimulationBasedInferenceDynamicHMCExt` | DynamicHMC 3.x | Dynamic HMC sampling |
| `SimulationBasedInferenceEmceeExt` | AffineInvariantMCMC | emcee affine-invariant sampler |
| `SimulationBasedInferenceDiffEqBaseExt` | DiffEqBase 6.x | Differential equation problem extensions |
| `SimulationBasedInferenceFluxExt` | Flux | Neural network surrogates |
| PySBI (via PythonCall) | pyABC, pyPESTO | Python SBI tools integration |

### Extension Loading Pattern

Extensions load automatically when the dependency is loaded:

```julia
# This automatically loads SimulationBasedInferenceDynamicHMCExt
using DynamicHMC
using SimulationBasedInference

# Now MCMC methods are available
sol = solve(inference_prob, MCMC(DynamicHMC.MetropolisHastings()))
```

---

## Quick Reference: Symbol Lookup

### Need to... | Use this symbol/file
-------------|---------------------
Define a prior distribution | `prior()` in `priors/distributions.jl`
Create an observable | `SimulatorObservable` in `observables.jl`
Set up forward problem | `SimulatorForwardProblem` in `forward_problem.jl`
Set up inference problem | `SimulatorInferenceProblem` in `inference_problem.jl`
Define likelihood | `GaussianLikelihood`, `SimulatorLikelihood` in `likelihoods/`
Run MCMC | `MCMC()` container in `mcmc/mcmc_base.jl`
Run ensemble method | `EnsembleSolver` in `ensembles/ensemble_solver.jl`
Use EKS algorithm | `EKS()` in `ensembles/eks.jl`
Use ES-MDA algorithm | `ESMDA()` in `ensembles/es-mda.jl`
Check convergence | `hasconverged(alg, state)`
Get ensemble values | `get_ensemble(state::EnsembleState)`
Transform to unconstrained space | `unconstrained_forward_map(prior, ζ)`
Sample from prior | `rand(prior)` or `sample(prior, n)`
Evaluate log probability | `logprob(prior, x)`

---

## References

1. **Package Repository**: https://github.com/bgroenks96/SimulationBasedInference.jl
2. **Documentation**: https://bgroenks96.github.io/SimulationBasedInference.jl/dev/
3. **SciML Ecosystem**: https://docs.sciml.ai/
4. **LogDensityProblems.jl**: https://github.com/TuringLang/LogDensityProblems.jl
5. **ComponentArrays.jl**: https://github.com/jmmease/ComponentArrays.jl
6. **Bijectors.jl**: https://github.com/TuringLang/Bijectors.jl

---

**Document Version**: 1.0  
**Last Updated**: 2026-06-25  
**Target Audience**: LLMs, developers, researchers using SimulationBasedInference.jl
