module SimulationBasedInferenceGenExt

using SimulationBasedInference

using Gen
using Random

include("gen_utils.jl")

export GenSimulatorPrior
include("gen_prior.jl")

end
