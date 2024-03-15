module Emulators

using SimulationBasedInference

using Bijectors
using LinearAlgebra
using Statistics, StatsBase

# common solve interface
import CommonSolve

export Emulator

"""
    fit!(emulator::Emulator)
    fit!(regressor, X, y)

Fits the given emulator on the stored data or the given regressor to `X` and `y`.
"""
function fit! end

"""
    predict(emulator, X)

Generate predictions from the given emulator for new inputs `X`.
"""
function predict end

export EmulatorData
include("emulator_data.jl")

export StackedRegressors
include("stacked.jl")

export Decorrelated, Centered, Standardized, NoTransform
export transform_target, inverse_transform_target
include("transforms.jl")

export GPRegressor
include("gp_regression.jl")

export Emulator, StackedEmulator
include("emulator.jl")

export EmulatedObservables
include("emulated_forward_solver.jl")

end
