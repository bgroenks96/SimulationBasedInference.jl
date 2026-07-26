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

"""
    _iteration_indices(storage::SimulationDataSet, iter::Int)

Return the indices of the stored simulations belonging to inference iteration `iter`,
ordered by member. An `iter < 1` is interpreted as the final iteration.
"""
function _iteration_indices(storage::SimulationDataSet, iter::Int)
    it = iter < 1 ? iterations(storage) : iter
    idxs = [i for i in 1:length(storage) if get(getmetadata(storage, i), :iter, 1) == it]
    members = [get(getmetadata(storage, i), :member, j) for (j, i) in enumerate(idxs)]
    return idxs[sortperm(members)]
end

"""
    get_ensemble(sol::EnsembleInferenceSolution, iter::Int=iterations(sol.storage))

Fetches the state of the ensemble from the given solution object by reconstructing the
parameter matrix from the per-member simulation records at iteration `iter` (columns are
ensemble members). For iterative algorithms, the optional argument `iter` may be provided;
`iter < 1` selects the final iteration.
"""
function get_ensemble(sol::EnsembleInferenceSolution, iter::Int=iterations(sol.storage))
    storage = sol.storage
    idxs = _iteration_indices(storage, iter)
    return reduce(hcat, [getinputs(storage, i) for i in idxs])
end

"""
    get_transformed_ensemble(sol::EnsembleInferenceSolution, iter::Int=iterations(sol.storage))

Fetches the transformed ensemble from the given solution object. For iterative algorithms, the
optional argument `iter` may be provided, which then retrieves the ensemble at the given iteration.
"""
function get_transformed_ensemble(sol::EnsembleInferenceSolution, iter::Int=iterations(sol.storage))
    # get transform
    prob = sol.prob
    inverse_transform = inverse(bijector(prob.prior.model))
    # retrieve ensemble from storage
    ens = get_ensemble(sol, iter)
    return mapslices(inverse_transform, ens, dims=1)
end

"""
    get_observables(sol::EnsembleInferenceSolution, iter::Int=iterations(sol.storage))

Returns a `NamedTuple` of the ensemble simulated observables at iteration `iter`, assembled
by concatenating each member's observable value along the ensemble dimension.
"""
function get_observables(sol::EnsembleInferenceSolution, iter::Int=iterations(sol.storage))
    storage = sol.storage
    idxs = _iteration_indices(storage, iter)
    observables = sol.prob.forward_prob.observables
    return map(observables) do obs
        reduce(enscat, [getvalue(storage[i], obs) for i in idxs])
    end
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
    sol::EnsembleInferenceSolution,
    new_storage::SimulationDataSet=SimulationDataSet();
    num_samples_per_sim::Int=1,
    pred_transform=identity,
    iters=[],
    rng::Random.AbstractRNG=Random.default_rng(),
)
    likelihoods = sol.prob.likelihoods
    prior = sol.prob.prior
    storage = sol.storage
    for j in 1:length(storage)
        data = storage[j]
        meta = getmetadata(data)
        x = getinputs(data)
        if !isempty(iters) && get(meta, :iter, 1) ∉ iters
            continue
        end
        for _ in 1:num_samples_per_sim
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
                # the stored simulation `data` already holds the simulated observable value
                y_dist = predictive_distribution(data, lik, bij(x_lik[nm])...)
                nm => pred_transform(rand(rng, y_dist))
            end
            y_obs = (; y_obs...)
            x_new = vcat(x, reduce(vcat, x_lik))
            # record the predictive sample as a new simulation
            new_data = allocate!(new_storage; meta...)
            setinputs!(new_data, x_new)
            for (nm, val) in pairs(y_obs)
                store!(new_data, nm, val)
            end
        end
    end
    return new_storage
end

function PosteriorStats.summarize(samples::AbstractMatrix, args...; kwargs...)
    return PosteriorStats.summarize(reshape(samples, size(samples, 1), 1, size(samples, 2)), args...; kwargs...)
end

function PosteriorStats.summarize(sol::EnsembleInferenceSolution, args...; iter=-1, kwargs...)
    ens = get_transformed_ensemble(sol, iter)
    # transpose to get N x k where N is the number of ensemble members (samples)
    ens_transpose = transpose(ens)
    # get parameter names
    param_names = Symbol.(labels(sol.prob.u0.model))
    # add extra "chain" dimension (here just set to one) and pass to summarize
    return PosteriorStats.summarize(reshape(ens_transpose, size(ens_transpose, 1), 1, size(ens_transpose, 2)), args...; var_names=param_names, kwargs...)
end

function MCMCChains.Chains(sol::EnsembleInferenceSolution; iter=-1)
    ens = get_transformed_ensemble(sol, iter)
    # transpose to get N x k where N is the number of ensemble members (samples)
    ens_transpose = transpose(ens)
    # get parameter names
    param_names = Symbol.(labels(sol.prob.u0.model))
    return Chains(ens_transpose, param_names)
end
