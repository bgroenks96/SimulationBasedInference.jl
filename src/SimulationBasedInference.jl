module SimulationBasedInference

using Reexport

# Utility
using Dates
using FileIO
using LinearAlgebra
using Requires

# SciML/DiffEq
using CommonSolve
using DiffEqBase

# Stats
using Bijectors
using MCMCChains
using Random
using UnPack

# Re-exported packages
@reexport using ComponentArrays
@reexport using Distributions
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
export logprob

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

export SimpleForwardMapStorage
export store!, getinputs, getoutputs
include("storage.jl")

export SimulatorObservable
export observe!, retrieve
include("observables.jl")

export ParameterTransform
include("param_map.jl")

export AbstractPrior, PriorDistribution
export prior
include("priors.jl")

export GaussianApproximationMethod, EmpiricalGaussian
export gaussian_approx
include("gaussian_approx.jl")

export SimulatorLikelihood, JointPrior
include("likelihoods/likelihoods.jl")

export SimulatorForwardProblem, SimulatorForwardSolution
include("forward_problem.jl")

export SimulatorInferenceProblem, SimulatorInferenceSolution
include("inference_problem.jl")

export SimulatorForwardSolver
include("forward_solve.jl")

export SimulatorODEForwardSolver
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
    # Backwards comaptible extension loading
    @require_extensions
end

end
