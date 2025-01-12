module SimulationBasedInference

using Reexport

# Utility
using Dates
using FileIO
using ForwardDiff
using LinearAlgebra
using Requires

# SciML/DiffEq
using CommonSolve
using DiffEqBase

# Stats
using MCMCChains
using Optim
using Random
using UnPack

# Re-exported packages
@reexport using Bijectors
@reexport using ComponentArrays
@reexport using DimensionalData: DimensionalData, Dimension, Dim, DimArray, X, Y, Z, Ti
@reexport using DimensionalData: dims, hasdim, rebuild
@reexport using Distributions
@reexport using PosteriorStats: PosteriorStats, summarize
@reexport using SciMLBase
@reexport using StatsBase
@reexport using StatsFuns
@reexport using Statistics

# to suppress name collision warning
using SciMLBase: islinear

using LogDensityProblems

import LogDensityProblems: logdensity

export LogDensityProblems, logdensity

export SimulatorInferenceAlgorithm

"""
Base type for all simulator-based inference algorithms.
"""
abstract type SimulatorInferenceAlgorithm end

# Probabilistic model constructors

"""
    likelihood_model(prob::SimulatorInferenceProblem, forward_alg; solve_kwargs...)

Constructs a function or type which represents the forward map and/or likelihood
component of the probailistic model. Can be implemented by extensions for specific
probabilistic programming languages or modeling tools.
"""
function likelihood_model end

"""
    joint_model(prob::SimulatorInferenceProblem, forward_alg; solve_kwargs...)

Constructs a function or type which represents the full joint model (prior + likelihood).
Can be implemented by extensions for specific probabilistic programming languages or modeling tools.
"""
function joint_model end

##################################

export autoprior, from_moments
include("utils.jl")

export SimulationData, SimulationArrayStorage
export store!, getinputs, getoutputs, getmetadata
include("simulation_data.jl")

export SimulatorObservable, TimeSampledObservable, TransientObservable
export observe!, getvalue, coordinates
include("observables.jl")

export AbstractSimulatorPrior, NamedProductPrior
export GaussianApproximationMethod, EmpiricalGaussian, LaplaceMethod
export prior, logprob, forward_map, unconstrained_forward_map, gaussian_approx
include("priors/priors.jl")

export SimulatorLikelihood
include("likelihoods/likelihoods.jl")

export SimulatorForwardProblem, SimulatorForwardSolution
export get_observable, get_observables
include("forward_problem.jl")

export SimulatorInferenceProblem, SimulatorInferenceSolution
include("inference_problem.jl")

# LogDensityProblems interface
include("logdensity.jl")

export SimulatorForwardSolver
include("forward_solve.jl")

export SimulatorODEForwardSolver, ODEObservable
include("forward_solve_ode.jl")

include("emulators/Emulators.jl")
using .Emulators

# Inference algorithms; these files should
# already export all relevant types/methods
include("ensembles/ensembles.jl")
include("mcmc/mcmc.jl")

export SBI # alias for base module
const SBI = SimulationBasedInference

using PackageExtensionCompat

function __init__()
    @require PythonCall="6099a3de-0909-46bc-b1f4-468b9a2dfc0d" begin
        include("../ext/pysbi/PySBI.jl")
    end
    # Backwards comaptible extension loading
    @require_extensions
end

end
