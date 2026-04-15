module SimulationBasedInference

using Reexport

# Utility
using Dates
using ForwardDiff
using LinearAlgebra
using Requires

# SciML
using SciMLBase

# Stats
using MCMCChains
using Optim
using Random

# Re-exported namespaces
@reexport using Bijectors
@reexport using ComponentArrays
@reexport using DimensionalData: DimensionalData, Dimension, Dim, DimArray, X, Y, Z, Ti
@reexport using DimensionalData: dims, hasdim, rebuild
@reexport using Distributions
@reexport using PosteriorStats: PosteriorStats, summarize
@reexport using StatsBase
@reexport using StatsFuns
@reexport using Statistics

@reexport import CommonSolve: init, solve, solve!, step!
@reexport import LogDensityProblems: LogDensityProblems, logdensity

import SciMLBase: EnsembleAlgorithm
# to suppress name collision warnings
import SciMLBase: islinear

export init, solve, solve!, step!

export SimulatorInferenceAlgorithm

"""
Base type for all simulator-based inference algorithms.
"""
abstract type SimulatorInferenceAlgorithm end

export autoprior, from_moments
include("utils.jl")

export SimulationData, SimulationArrayStorage
export store!, getinputs, getoutputs, getmetadata
include("simulation_data.jl")

export SimulatorObservable, TimeSampledObservable, TransientObservable, TimeSampled
export observe!, getvalue, coordinates
include("observables.jl")

export AbstractSimulatorPrior, NamedProductPrior
export GaussianApproximationMethod, EmpiricalGaussian, LaplaceMethod
export prior, logprob, forward_map, unconstrained_forward_map, gaussian_approx
include("priors/priors.jl")

export SimulatorLikelihood
include("likelihoods/likelihoods.jl")

export Simulator
include("simulator_interface.jl")

export SimulatorForwardProblem, SimulatorForwardSolution
export get_observable, get_observables
include("forward_problem.jl")

export SimulatorForwardSolution
include("forward_solve.jl")

export SimulatorInferenceProblem, SimulatorInferenceSolution
include("inference_problem.jl")

# LogDensityProblems interface
export LogDensityProblems, logdensity
include("logdensity.jl")

# Inference algorithms; these files should
# already export all relevant types/methods
include("ensembles/ensembles.jl")
include("mcmc/mcmc.jl")

export SBI # alias for base module
const SBI = SimulationBasedInference

function __init__()
    @require PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d" begin
        using CondaPkg
        include("PySBI/PySBI.jl")
    end
    return nothing
end

end
