module Emulators

using SimulationBasedInference

using LinearAlgebra
using Statistics

import MLJModelInterface as MMI

abstract type Emulator end

abstract type EmulatorDataTransform end

export Decorrelated, NoTransform
export transform_target, inverse_transform_target
include("transforms.jl")

export StackedMLEmulator
include("stacked_emulator.jl")

end
