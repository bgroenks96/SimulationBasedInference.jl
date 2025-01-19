"""
    with_names(xs)

`map`s over `xs` and returns a `NamedTuple` where the keys are `nameof(x)` for each `x`.
"""
with_names(xs) = (; map(x -> nameof(x) => x, xs)...)

"""
    ntreduce(f, xs::AbstractVector{<:NamedTuple})

Applies `reduce` over a vector of named tuples, applying the reducer function `f`
to each element and returning a named tuple of the reduced outputs. All named tuples
in the vector must have the same keys.
"""
function ntreduce(f, xs::AbstractVector{<:NamedTuple})
    foldl(xs) do acc, xᵢ
        (; map(k -> k => f(acc[k], xᵢ[k]), keys(acc))...)
    end
end

"""
    adstrip(x::ForwardDiff.Dual)
    adstrip(x::Number)

Strips the AD type from `x` if it is a `Dual` number; otherwise just returns `x`.
"""
adstrip(x::ForwardDiff.Dual) = ForwardDiff.value(x)
adstrip(x::Number) = x

"""
    betadist(mean, dispersion)

Helper method for defining a `Beta` distribution with the `mean` and `dispersion` parameterization, i.e:

`betadist(mean, dispersion) = Beta(max(mean*dispersion,1), max((1-mean)*dispersion,1))`

The `dispersion` parameter controls how tightly the distribution is quasi-centered around the mean. Higher
dispsersion values correspond to a tighter distribution and thus lower variance.
"""
function betadist(mean, dispersion)
    @assert 0 <= mean <= 1
    @assert dispersion > 0
    Beta(max(mean*dispersion,1), max((1-mean)*dispersion,1))
end

"""
    autoprior(mean, stddev; lower=-Inf, upper=Inf)

Helper function that automatically constructs a prior distribution with the given mean, standard deviation, and support.
Note that `lower < mean <= upper` must be satisfied. For unbounded variables, `Normal` is returned. For variabels
bounded from below (i.e. either lower or upper bound is finite), a transformed `LogNormal` distribution is constructed.
For double bounded variables, (i.e. both lower and upper are finite) a tranformed `Beta` distribution is constructed.
"""
function autoprior(mean, stddev; lower=-Inf, upper=Inf)
    @assert lower < mean <= upper "mean must lie within the specified bounds"
    @assert stddev > zero(stddev) "stddev must be strictly positive"
    # for double bounded, choose Beta, shifting and scaling as needed
    if isfinite(upper) && isfinite(lower)
        dist = from_moments(Beta, (mean - lower) / (upper - lower), stddev / (upper - lower))
        scaled = isone(upper - lower) ? dist : (upper-lower)*dist
        shifted = iszero(lower) ? dist : scaled + lower
        return shifted
    # for single bounded, choose LogNormal, shifting as needed
    elseif isfinite(lower)
        dist = from_moments(LogNormal, mean - lower, stddev)
        return iszero(lower) ? dist : dist + lower
    elseif isfinite(upper)
        dist = from_moments(LogNormal, upper - mean, stddev)
        return iszero(upper) ? -1*dist : -1*dist + upper
    else
        # unbounded, default to Normal
        return from_moments(Normal, mean ,stddev)
    end
end

"""
    from_moments(::Type{T}, mean, stddev) where {T<:Distribution}

Construct a distribution of type `T` using the method of moments. It is assumed that
the given `mean` lies within the untransformed support of the distribution.
"""
from_moments(::Type{T}, mean, stddev) where {T<:Distribution} = error("not implemented for distribution $T")
from_moments(::Type{Normal}, mean, stddev) = Normal(mean, stddev)
function from_moments(::Type{Beta}, mean, stddev)
    var = stddev^2
    p = (mean*(1-mean)) / var - 1
    return Beta(p*mean, p*(1-mean))
end
function from_moments(::Type{LogNormal}, mean, stddev)
    var = stddev^2
    μ = log(mean / sqrt(var / mean^2 + 1))
    σ = sqrt(log(var / mean^2 + 1))
    return LogNormal(μ, σ)
end
function from_moments(::Type{InverseGaussian}, mean, stddev)
    var = stddev^2
    μ = mean
    λ = μ^3 / var
    return InverseGaussian(μ, λ)
end
function from_moments(::Type{Gamma}, mean, stddev)
    var = stddev^2
    α = mean^2 / var
    θ = var / mean
    return Gamma(α, θ)
end

function quantile_loss(::Type{distType}, bij, qs::Pair...) where {distType<:UnivariateDistribution}
    function loss(θ)
        d = distType(bij(θ)...)
        ℓ = sum(map(qv -> (qv[2] - invlogcdf(d, log(qv[1])))^2, qs))
        return ℓ
    end
end

function from_quantiles(
    d0::distType,
    qs::Pair...;
    optimizer=Newton(),
    options=Optim.Options(),
) where {distType<:UnivariateDistribution}
    bij = param_bijector(distType)
    qloss = quantile_loss(distType, inverse(bij), qs...)
    initial_x = bij(collect(Distributions.params(d0)))
    res = optimize(qloss, initial_x, optimizer, options)
    @assert res.ls_success "Optimization of $distType failed for $(qs); $res"
    return distType(inverse(bij)(res.minimizer)...)
end

# Bijector utilities

"""
    param_bijector(::Type{T}) where {T<:UnivariateDistribution}

Returns a bijector for the parameters of distribution type `T`.
"""
param_bijector(::Type{<:Union{Normal,LogNormal,LogitNormal,Logistic}}) = Stacked(identity, Base.Fix1(broadcast, log))
param_bijector(::Type{<:Union{Beta,Gamma,InverseGamma,InverseGaussian}}) = Stacked(Base.Fix1(broadcast, log), Base.Fix1(broadcast, log))
param_bijector(::Type{Exponential}) = Base.Fix1(broadcast, log)
param_bijector(::Type{Bernoulli}) = Base.Fix1(broadcast, logit)
param_bijector(::Type{T}) where {T} = error("no parameter bijector defined for distribution type $T")

bstack(b1, b2) = Stacked(b1, b2)
function bstack(b1::Stacked, b2)
    last_in = last(b1.ranges_in[end])+1
    last_out = last(b1.ranges_out[end])+1
    ranges_in = tuple(b1.ranges_in..., last_in:last_in)
    ranges_out = tuple(b1.ranges_out..., last_out:last_out)
    length_in = b1.length_in + 1
    length_out = b1.length_out + 1
    return Stacked(tuple(b1.bs..., b2), ranges_in, ranges_out, length_in, length_out)
end
function bstack(b1, b2::Stacked)
    ranges_in = tuple(1:1, map(r -> first(r)+1:last(r)+1, b2.ranges_in)...)
    ranges_out = tuple(1:1, map(r -> first(r)+1:last(r)+1, b2.ranges_out)...)
    length_in = b2.length_in + 1
    length_out = b2.length_out + 1
    return Stacked(tuple(b1, b2.bs...), ranges_in, ranges_out, length_in, length_out)
end
function bstack(b1::Stacked, b2::Stacked)
    offs_in = last(b1.ranges_in[end])
    offs_out = last(b1.ranges_out[end])
    ranges_in = vcat(b1.ranges_in, map(r -> first(r)+offs_in:last(r)+offs_in, b2.ranges_in))
    ranges_out = vcat(b1.ranges_out, map(r -> first(r)+offs_out:last(r)+offs_out, b2.ranges_out))
    return Stacked(
        tuple(b1.bs..., b2.bs...),
        tuple(ranges_in...),
        tuple(ranges_out...),
        b1.length_in + b2.length_in,
        b1.length_out + b2.length_out,
    )
end

# logprob overrides

# This is type piracy but nice to make Distributions implement log-density interface;
# TODO: consider creating an issue on LogDensityProblems or Distributions?
"""
    logprob(d::UnivariateDistribution, x::Number)

Alias for `logpdf(d,x)` on `Distribution` types.
"""
logprob(d::UnivariateDistribution, x::Number) = logpdf(d, x)
logprob(d::UnivariateDistribution, x::AbstractVector) = sum(logpdf.(d, x))
logprob(d::MultivariateDistribution, x::AbstractVector) = logpdf(d, x)
logprob(d::MultivariateDistribution, x::AbstractMatrix) = sum(map(x -> logpdf(d, x), eachcol(x)))
