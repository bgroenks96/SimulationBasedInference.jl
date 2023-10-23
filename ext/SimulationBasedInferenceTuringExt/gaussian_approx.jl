full_gaussian_approximation(model::Turing.Model; kwargs...) = full_gaussian_approximation(Random.GLOBAL_RNG, model; kwargs...)
function full_gaussian_approximation(rng::Random.AbstractRNG, model::Turing.Model; nsamples=1000, return_chain=false)
    priorchain = sample(rng, model, Prior(), nsamples)
    f = bijector(model)
    Θ = reduce(hcat, map(f, eachrow(Array(priorchain))))
    μ = dropdims(mean(Θ, dims=2), dims=2)
    Σ = cov(θ, dims=2)
    if return_chain
        return MvNormal(μ, Σ), priorchain
    else
        return MvNormal(μ, Σ)
    end
end

function fit_diag_gaussian_approximation(
    rng::Random.AbstractRNG,
    prior::Turing.Model;
    num_samples_per_iteration=1000,
    optimizer=Newton(),
    opts=Optim.Options(iterations=100),
    return_optim_result=false,
)
    proto = ComponentArray(rand(rng, prior))
    ax = getaxes(proto)
    f = bijector(prior)

    function sample_from_prior()
        return reduce(hcat, [reshape(ComponentArray(rand(rng, prior)),:,1) for i in 1:num_samples_per_iteration])
    end

    function makedist(θ)
        n = length(θ)
        q_μ, q_σ = θ[1:Int(n//2)], θ[Int(n//2)+1:end]
        q = MvNormal(q_μ, Diagonal(q_σ.^2))
        return q
    end

    function kld(θ)
        q = makedist(θ)
        # Z = rand(q, num_samples_per_iteration)
        # X = mapslices(inverse(f), Z, dims=1)
        # logp_q = mapslices(z -> logpdf(q, z), Z, dims=1)
        # logp_m = mapslices(x -> logprior(prior, ComponentArray(x, ax)), X, dims=1)
        # logdetJ = mapslices(x -> logabsdetjacinv(f, x), X, dims=1)
        X = sample_from_prior()
        logp_m = mapslices(x -> logprior(prior, ComponentArray(x, ax)), X, dims=1)
        logdetJ = mapslices(x -> logabsdetjac(f, x), X, dims=1)
        Z = mapslices(f, X, dims=1)
        logp_q = mapslices(z -> logpdf(q, z), Z, dims=1)
        # kl = mean(logp_q .- logp_m .- logdetJ)
        kl = mean(logp_m .- logp_q .- logdetJ)
        return kl
    end

    ϕ₀ = sample_from_prior()
    z₀ = mapslices(f, ϕ₀, dims=1)
    θ₀ = vcat(mean(z₀, dims=2)[:,1], std(z₀, dims=2)[:,1])
    res = optimize(kld, θ₀, optimizer, opts, autodiff=:forward)
    if !res.g_converged
        @warn "optimization did not converge!\n$res"
    end
    dist = makedist(res.minimizer)
    return return_optim_result ? (; dist, res) : dist
end
