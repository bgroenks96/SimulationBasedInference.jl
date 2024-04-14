# This code is adapted from:
# https://github.com/john-waczak/MLJGaussianProcesses.jl/tree/main
# Revision: 40645ec
# ...with MLJ dependent pieces removed.
# Pursuant to the following license:
# MIT License

# Copyright (c) 2022 John Waczak <john.louis.waczak@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using AbstractGPs, KernelFunctions
using ParameterHandling
using Optim
using Random
using Zygote

"""
    GPRegressor

Generic implementation of a Gaussian Process regressor for a univariate outputs.
This implementation is adapted from `MLJGaussianProcesses` to be usable without
MLJ.
"""
Base.@kwdef mutable struct GPRegressor
    μ::Function = default_linear_mean
    k::Function = default_kernel
    θ_init::NamedTuple = θ_default
    μ_init::Function = mean_function_initializer(μ, Random.GLOBAL_RNG)
    σ²::Float64 = 1e-6
    optimizer::Optim.AbstractOptimizer = LBFGS()
    optimizer_opts::NamedTuple = (;)
    fitresult::Any = missing
end

θ_default = (σf² = positive(1.0), ℓ = positive(1.0))
  
# this is what the user should supply for MLJ instead of just a kernel
function default_kernel(θ::NamedTuple)
    return  θ.σf²  * (SqExponentialKernel() ∘ ScaleTransform(1/2(θ.ℓ)^2) + WhiteKernel())
end

function fit!(gp::GPRegressor, X::AbstractMatrix, y::AbstractVector; verbosity=1, kwargs...)
    Xmatrix = ColVecs(X)

    # augment θ_init to include mean function params and σ²
    θ_mean = gp.μ_init(X, y)
    θ_init_full = (gp.θ_init..., θ_mean..., σ²=positive(gp.σ²))
    flat_θᵢ, unflatten = ParameterHandling.value_flatten(θ_init_full)

    # define loss function as minus marginal log-likelihood
    function objective(θ::NamedTuple)
        k = gp.k(θ) # build kernel function with current params
        μ = gp.μ(θ) # build mean function with current params
        f = GP(μ, k) # build AbstractGP using our μ and k
        fₓ = f(Xmatrix, θ.σ²) # construct fintite GP at points in Xmatrix
        return -logpdf(fₓ, y) # return minus marginal log-likelihood
    end

    # default option uses finite diff methods
    if verbosity > 0
        opt = Optim.optimize(
            objective ∘ unflatten,
            θ -> only(Zygote.gradient(objective ∘ unflatten, θ)),
            flat_θᵢ,
            gp.optimizer,
            Optim.Options(;merge((show_trace=true,), gp.optimizer_opts)...);
            inplace=false,
        )
    else
        opt = Optim.optimize(
            objective ∘ unflatten,
            θ -> only(Zygote.gradient(objective ∘ unflatten, θ)),
            flat_θᵢ,
            gp.optimizer,
            Optim.Options(; gp.optimizer_opts...);
            inplace=false,
        )
    end


    θ_best = unflatten(opt.minimizer)

    f = GP(gp.μ(θ_best), gp.k(θ_best))
    fₓ = f(Xmatrix, θ_best.σ²)
    p_fₓ = posterior(fₓ, y)  # <-- this is our fitresult as it let's us do everything we need

    # generate new θ for fitresult
    θ_out = [p for p ∈ pairs(θ_best) if p[1] != :σ²]

    gp.fitresult = (p_fₓ, θ_out, θ_best.σ², opt)

    return gp
end

function predict(gp::GPRegressor, X)
    @assert !ismissing(gp.fitresult)
    p_fₓ, _ = gp.fitresult
    Xdata = ColVecs(X)
    fₓ = p_fₓ(Xdata)
    return marginals(fₓ)
end

# mean functions

"""
    LinearMean{coefType,meanType} <: AbstractGPs.MeanFunction

Simple implementation of a linear mean function `β⋅x + μ` where `x`
is the feature vector, `β` are the linear coefficients, and `μ` is
a constant offset term.
"""
struct LinearMean{coefType,meanType} <: AbstractGPs.MeanFunction
    β::coefType
    μ::meanType
end

AbstractGPs.mean_vector(mf::LinearMean, vecs::ColVecs) = dropdims(reshape(mf.β,1,:)*vecs.X .+ mf.μ, dims=1)
AbstractGPs.mean_vector(mf::LinearMean, vecs::RowVecs) = dropdims(vecs.X*reshape(mf.β,:,1) .+ mf.μ, dims=2)

function default_linear_mean(θ)
    return LinearMean(θ.β, θ.μ)
end

function mean_function_initializer(::typeof(default_linear_mean), rng::AbstractRNG)
    function init(X, y)
        return (β = randn(rng, size(X,1)), μ = zero(eltype(y)))
    end
end

function default_zero_mean(θ::NamedTuple)
    return ZeroMean()
end

function mean_function_initializer(::typeof(default_zero_mean), ::AbstractRNG)
    init_zero_mean(X, y) = (;)
end

function default_const_mean(θ::NamedTuple)
    return ConstMean(θ.μ)
end

function mean_function_initializer(::typeof(default_const_mean), ::AbstractRNG)
    init_const_mean(X, y) = (μ=zero(eltype(y)),)
end
