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
        # choose median of prior as standard deviation
        return diag(cov(lik, first(median(lik.prior))))
    end
    # concatenate all covariance matrices 
    return Diagonal(reduce(vcat, cov_diags))
end

function iterationindices(storage::SimulationData, alg::EnsembleInferenceAlgorithm, iter::Int)
    if isiterative(alg)
        iters = map(m -> m.iter, getmetadata(storage))
        iter = iter > 0 ? iter : maximum(iters)
        inds = findall(==(iter), iters)
    else
        inds = 1:length(storage)
    end
    return inds
end

"""
    get_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)

Fetches the state of the ensemble from the given solution object. For iterative algorithms, the
optinal argument `iter` may be provided, which then retrieves the ensemble at the given iteration.
"""
function get_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)
    # find indices matching for iteration
    inds = iterationindices(sol.storage, sol.alg, iter)
    # retrieve ensemble from storage and concatenate
    return reduce(hcat, getinputs(sol.storage, inds))
end

"""
    get_transformed_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)
    
Fetches the transformed ensemble from the given solution object. For iterative algorithms, the
optinal argument `iter` may be provided, which then retrieves the ensemble at the given iteration.
"""
function get_transformed_ensemble(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)
    # get transform
    prob = sol.prob
    inverse_transform = inverse(bijector(prob.prior.model))
    # find indices matching iteration
    inds = iterationindices(sol.storage, sol.alg, iter)
    # retrieve ensemble from storage
    ens = getinputs(sol.storage, inds)
    return reduce(hcat, map(inverse_transform, ens))
end

function get_observables(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)
    # find indices matching iteration
    inds = iterationindices(sol.storage, sol.alg, iter)
    # retrieve ensemble from storage
    out = getoutputs(sol.storage, inds)
    # reduce over named tuples, concatenating each observable
    return reduce(out) do acc, outᵢ
        (; map(k -> k => hcat(acc[k], outᵢ[k]), keys(acc))...)
    end
end
