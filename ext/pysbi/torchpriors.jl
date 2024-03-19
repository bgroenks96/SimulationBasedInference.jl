# conversions for Gaussian distributions
torchprior(d::Normal) = torch.distributions.Normal(torch.Tensor([d.μ]), torch.Tensor([d.σ]))
torchprior(d::LogNormal) = torch.distributions.LogNormal(torch.Tensor([d.μ]), torch.Tensor([d.σ]))
torchprior(d::LogitNormal) = torch.distributions.LogisticNormal(torch.Tensor([d.μ]), torch.Tensor([d.σ]))
torchprior(d::MvNormal) = torch.distributions.MultivariateNormal(torch.Tensor(d.μ), torch.Tensor(Py(d.Σ).to_numpy()))

"""
    torchprior(prior::AbstractPrior)

Currently, it seems that `torch` does not provide a generic product distribution for combining
univariate distributions into a higher dimensional multivariate distribution. As a cheap workaround,
we follow here the same strategy as with the ensemble Kalman methods, and we use the `Bijector`s
defined for each `AbstractPrior` to construct an empirical multivariate normal approximation.
"""
function torchprior(prior::AbstractPrior, approx::GaussianApproximationMethod=LaplaceMethod(); rng=Random.default_rng())
    normal_prior = gaussian_approx(approx, prior; rng)
    μ = mean(normal_prior)
    Σ = cov(normal_prior)
    return torch.distributions.MultivariateNormal(torch.Tensor(μ), torch.Tensor(Py(Σ).to_numpy()))
end
