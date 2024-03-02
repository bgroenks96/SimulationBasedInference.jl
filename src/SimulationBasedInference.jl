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
using LogDensityProblems
using MCMCChains
using Random

# Re-exported packages
@reexport using ComponentArrays
@reexport using Distributions
@reexport using SciMLBase
@reexport using StatsBase
@reexport using StatsFuns
@reexport using Statistics

import LogDensityProblems: logdensity

export SimulatorInferenceAlgorithm
export logdensity

"""
Base type for all simulator-based inference algorithms.
"""
abstract type SimulatorInferenceAlgorithm end

export autoprior, from_moments
include("utils.jl")

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
include("likelihoods.jl")

export SimulatorForwardProblem, SimulatorForwardSolution
include("forward_problem.jl")

export SimulatorInferenceProblem, SimulatorInferenceSolution
include("inference_problem.jl")

export SimulatorForwardSolver
include("forward_solve.jl")

export SimulatorODEForwardSolver
include("forward_solve_ode.jl")

include("Emulators/Emulators.jl")
using .Emulators

# Inference algorithms; these files should
# already export all relevant types/methods
include("ensembles/ensembles.jl")
include("mcmc/mcmc.jl")

export SBI # alias for base module
const SBI = SimulationBasedInference

function __init__()
    # Extension loading;
    # We use Requires.jl instead of built-in extensions for now (v1.9.3).
    # This is due to built-in extensions having the incredibly unfortunate limitation
    # of not being loadable or exportable at precompile time... so there is no way to
    # re-export types defined within them.
    @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin
        include("../ext/SimulationBasedInferenceTuringExt/SimulationBasedInferenceTuringExt.jl")
        @reexport using .SimulationBasedInferenceTuringExt
    end
end

end
