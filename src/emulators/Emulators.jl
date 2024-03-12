module Emulators

using SimulationBasedInference

using LinearAlgebra
using Statistics, StatsBase

# MLJ
using MLJBase, MLJModels
import MLJModelInterface: Model, predict

# common solve interface
import CommonSolve

abstract type Emulator end

export EmulatorData
include("emulator_data.jl")

export DecorrelatedTarget, CenteredTarget, NoTransform
export transform_target, inverse_transform_target
include("transforms.jl")

export StackedMLEmulator, stacked_emulator
include("stacked_emulator.jl")

export EmulatedObservables
include("solver_types.jl")

end
