"""
    obscov(::AbstractMatrix)
    obscov(::AbstractVector)
    obscov(::Number)

Builds a standard form multivariate normal covariance matrix
from the given matrix, vector (diagonal), or scalar (isotropic).
"""
obscov(Σ::AbstractMatrix) = Σ
obscov(σ::AbstractVector) = Diagonal(σ)
obscov(σ::Number) = σ*I
# from SimulatorLikelihood types
"""
    obscov(likelihoods::SimulatorLikelihood...)

Implementations should build a combined covariance matrix from the given likelihood types.
The default implementation simply throws an error.
"""
obscov(likelihoods::SimulatorLikelihood...) = error("obscov not implemented for the given likelihood types")
# currently only diagonal covariances are supported
function obscov(likelihoods::SimulatorLikelihood{<:Union{IsoNormal,DiagNormal}}...)
    cov_diags = map(likelihoods) do lik
        return diag(cov(lik, first(mean(lik.prior))))
    end
    # concatenate all covariance matrices 
    return Diagonal(reduce(vcat, cov_diags))
end

"""
    get_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=length(sol.cache))

Fetches the state of the ensemble from the given solution object. For iterative algorithms, the
optinal argument `iter` may be provided, which then retrieves the ensemble at the given iteration.
"""
function get_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=length(sol.cache))
    # retrieve ensemble from cache
    return getinputs(sol.cache, iter)
end

"""
    get_transformed_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)
    
Fetches the transformed ensemble from the given solution object. For iterative algorithms, the
optinal argument `iter` may be provided, which then retrieves the ensemble at the given iteration.
"""
function get_transformed_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=length(sol.cache))
    prob = sol.prob
    inverse_transform = inverse(bijector(prob.prior.model))
    # retrieve ensemble from cache
    ens = getinputs(sol.cache, iter)
    return reduce(hcat, map(inverse_transform, eachcol(ens)))
end

function get_predictions(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=length(sol.cache))
    outputs = getoutputs(sol.cache, iter)
    return outputs.pred
end

function get_observables(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=length(sol.cache))
    # internal function to extract names from vector of named tuples
    get_names(::Vector{<:NamedTuple{names}}) where names = names    
    
    # get outputs at iteration
    outputs = getoutputs(sol.cache, iter)
    observables = outputs.observables
    names = get_names(observables)
    named_observables = map(names) do n
        n => reduce(hcat, map(nt -> nt[n], observables))
    end
    # convert to named tuples
    return (; named_observables...)
end
