import EnsembleKalmanProcesses
import EnsembleKalmanProcesses:
    EnsembleKalmanProcess, Sampler,
    get_u_final, get_obs, get_obs_noise_cov, get_error, update_ensemble!

abstract type EnsembleInferenceAlgorithm <: SimulatorInferenceAlgorithm end

const EnsembleInferenceSolution{algType} = SimulatorInferenceSolution{algType} where {algType<:EnsembleInferenceAlgorithm}

export EnsembleSolver, get_ensemble, initialstate, ensemblestep!, finalize!, ensemble_solve
include("ensemble_solver.jl")

export obscov, get_transformed_ensemble, get_predictions, get_observables
include("ensemble_utils.jl")

export EnIS, PBS, importance_weights, get_weights
include("importance_sampling.jl")

export EKS
include("eks.jl")

export ESMDA, ensemble_kalman_analysis
include("es-mda.jl")
