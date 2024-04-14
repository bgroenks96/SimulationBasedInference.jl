module PySBI

using ..PythonCall

using Bijectors
using CommonSolve
using SimulationBasedInference

import Random

const torch = pyimport("torch")

# import sbi and its submodules
const sbi = pyimport("sbi")
pyimport("sbi.inference")
pyimport("sbi.utils")

export pyprior
include("pypriors.jl")

include("common.jl")

export PySNE
include("pysne.jl")

end
