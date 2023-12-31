module SimulationBasedInference

using Reexport

# Utility
@reexport using ComponentArrays
using Dates
using FileIO
using LinearAlgebra
using Requires

# SciML/DiffEq
using CommonSolve
using DiffEqBase, SciMLBase

# Stats
using Bijectors
using LogDensityProblems
using MCMCChains
using Random

# Re-exported packages
@reexport using Distributions
@reexport using StatsBase
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

include("Ensembles/Ensembles.jl")
using .Ensembles

include("Emulators/Emulators.jl")
using .Emulators

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
    @require OrdinaryDiffEq="1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        include("../ext/SimulationBasedInferenceOrdinaryDiffEqExt/SimulationBasedInferenceOrdinaryDiffEqExt.jl")
    end
end

end
