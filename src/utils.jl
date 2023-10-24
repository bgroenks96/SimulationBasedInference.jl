with_names(xs) = (; map(x -> nameof(x) => x, xs)...)

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