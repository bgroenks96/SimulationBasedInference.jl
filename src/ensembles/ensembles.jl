using EnsembleKalmanProcesses

abstract type EnsembleInferenceAlgorithm <: SimulatorInferenceAlgorithm end

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
