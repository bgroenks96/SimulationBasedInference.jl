module Emulators

using SimulationBasedInference

using LinearAlgebra
using MLJBase
using Statistics

import CommonSolve
import MLJModelInterface as MMI

abstract type Emulator end

abstract type EmulatorDataTransform end

export EmulatorData, DecorrelatedTarget, CenteredTarget, NoTransform
export transform_target, inverse_transform_target
include("transforms.jl")

export StackedMLEmulator
include("stacked_emulator.jl")

export EmulatedObservables
include("solver_types.jl")

end
