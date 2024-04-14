"""
    PyPrior

Wraps an `AbstractPrior` for use in the `sbi` python package. 
"""
struct PyPrior
    prior::AbstractPrior
    log_prob::Function
    sample::Function
    mean::Py
    variance::Py
end

function PyPrior(prior::AbstractPrior)
    function pylogprob(x)
        x = PyArray(x)
        binv = inverse(bijector(prior))
        if length(size(x)) == 1
            return SBI.logprob(prior, binv(x)) + SBI.logabsdetjac(binv, x)
        elseif length(size(x)) == 2
            xs = map(binv, eachrow(x))
            lp = transpose(reduce(hcat, map(xᵢ -> [SBI.logprob(prior, xᵢ) + SBI.logabsdetjac(binv, xᵢ)], xs)))
            return Py(lp).to_numpy()
        else
            error("invalid sample shape: $(size(x))")
        end
    end

    function pysample(shape)
        b = bijector(prior)
        samples = reduce(hcat, map(b, eachcol(SBI.sample(prior, pyconvert(Dims, shape)))))
        return Py(transpose(samples)).to_numpy()
    end

    # TODO: Maybe there is a better way to do this for priors that have analytical moments...
    rng = Random.MersenneTwister(1234)
    empirical_mean = Py(mean(map(bijector(prior), rand(rng, prior, 10_000)))).to_numpy()
    empirical_var = Py(var(map(bijector(prior), rand(rng, prior, 10_000)))).to_numpy()

    return PyPrior(prior, pylogprob, pysample, empirical_mean, empirical_var)
end

# conversions for Gaussian distributions
pyprior(d::Normal) = torch.distributions.Normal(torch.Tensor([d.μ]), torch.Tensor([d.σ]))
pyprior(d::LogNormal) = torch.distributions.LogNormal(torch.Tensor([d.μ]), torch.Tensor([d.σ]))
pyprior(d::LogitNormal) = torch.distributions.LogisticNormal(torch.Tensor([d.μ]), torch.Tensor([d.σ]))
pyprior(d::MvNormal) = torch.distributions.MultivariateNormal(torch.Tensor(d.μ), torch.Tensor(Py(d.Σ).to_numpy()))

# wrapper for generic priors
pyprior(prior::AbstractPrior) = PyPrior(prior)

"""
    torchprior(prior::AbstractPrior, approx::GaussianApproximationMethod)

Currently, it seems that `torch` does not provide a generic product distribution for combining
univariate distributions into a higher dimensional multivariate distribution. As a cheap workaround,
we follow here the same strategy as with the ensemble Kalman methods, and we use the `Bijector`s
defined for each `AbstractPrior` to construct an empirical multivariate normal approximation.
"""
function pyprior(prior::AbstractPrior, approx::GaussianApproximationMethod; rng=Random.default_rng())
    normal_prior = gaussian_approx(approx, prior; rng)
    μ = mean(normal_prior)
    Σ = cov(normal_prior)
    return torch.distributions.MultivariateNormal(torch.Tensor(μ), torch.Tensor(Py(Σ).to_numpy()))
end
