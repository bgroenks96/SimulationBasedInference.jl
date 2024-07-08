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

"""
    get_observables(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)

Returns a `NamedTuple` of concatenated observables at iteration `iter`.
"""
function get_observables(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, iter::Int=-1)
    # find indices matching iteration
    inds = iterationindices(sol.storage, sol.alg, iter)
    # retrieve ensemble from storage
    out = getoutputs(sol.storage, inds)
    # reduce over named tuples, concatenating each observable
    return ntreduce(enscat, convert(Vector{NamedTuple}, out))
end

enscat(x::AbstractVecOrMat, y::AbstractVector) = hcat(x, y)
function enscat(acc::DimArray, x::DimArray)
    acc_dims = Tuple(dims(acc))
    x_dims = Tuple(dims(x))
    if !hasdim(acc, :ens)
        acc = DimArray(reshape(acc.data, size(acc)..., 1), (acc_dims..., Dim{:ens}(1:1)))
    end
    N = size(acc, :ens)
    x = DimArray(reshape(x.data, size(x)..., 1), (x_dims..., Dim{:ens}(N+1:N+1)))
    return cat(acc, x, dims=:ens)
end

function sample_ensemble_predictive(
    sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm},
    new_storage::SimulationData=SimulationArrayStorage();
    num_samples_per_sim::Int=1,
    pred_transform=identity,
    iterations=[],
    rng::Random.AbstractRNG=Random.default_rng(),
)
    likelihoods = sol.prob.likelihoods
    prior = sol.prob.prior
    for (x, y, meta) in sol.storage
        if !isempty(iterations) && meta.iter ∉ iterations
            continue
        end
        for i in 1:num_samples_per_sim
            x_lik = map(keys(prior.lik)) do nm
                lik_prior = prior.lik[nm]
                bij = bijector(lik_prior)
                # sample from likelihood parameter prior in constrained space
                x_lik = rand(rng, lik_prior)
                # and map to unconstrained space
                nm => bij(x_lik)
            end
            x_lik = (; x_lik...)
            y_obs = map(keys(prior.lik)) do nm
                lik_prior = prior.lik[nm]
                # get inverse bijector to map back to constrained parameter space
                bij = inverse(bijector(lik_prior))
                lik = likelihoods[nm]
                setvalue!(lik.obs, reshape(y[nm], size(lik.obs)))
                y_dist = predictive_distribution(lik, bij(x_lik[nm])...)
                nm => pred_transform(rand(rng, y_dist))
            end
            y_obs = (; y_obs...)
            x_new = vcat(x, reduce(vcat, x_lik))
            store!(new_storage, x_new, y_obs; meta...)
        end
    end
    return new_storage
end

function PosteriorStats.summarize(samples::AbstractMatrix, args...; kwargs...)
    return PosteriorStats.summarize(reshape(samples, size(samples, 1), 1, size(samples, 2)), args...; kwargs...)
end

function PosteriorStats.summarize(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}, args...; iter=-1, kwargs...)
    ens = get_transformed_ensemble(sol, iter)
    # transpose to get N x k where N is the number of ensemble members (samples)
    ens_transpose = transpose(ens)
    # get parameter names
    param_names = Symbol.(labels(sol.prob.u0.model))
    # add extra "chain" dimension (here just set to one) and pass to summarize
    return PosteriorStats.summarize(reshape(ens_transpose, size(ens_transpose, 1), 1, size(ens_transpose, 2)), args...; var_names=param_names, kwargs...)
end

function MCMCChains.Chains(sol::SimulatorInferenceSolution{<:EnsembleInferenceAlgorithm}; iter=-1)
    ens = get_transformed_ensemble(sol, iter)
    # transpose to get N x k where N is the number of ensemble members (samples)
    ens_transpose = transpose(ens)
    # get parameter names
    param_names = Symbol.(labels(sol.prob.u0.model))
    return Chains(ens_transpose, param_names)
end
