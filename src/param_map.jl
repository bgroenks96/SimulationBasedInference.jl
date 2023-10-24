"""
    ParameterMapping{Tfn,Tlp}

Represents a callable transform which maps between different parameter spaces. `transform` should be a
function which performs the transformation f: Θ ↦ Φ and `logabsdetJ` should compute `log|det(J(f))|` where
`J(f)` is the Jacobian of `f`, i.e. ∂f/∂θ.
"""
struct ParameterMapping{Tfn,Tlp}
    transform::Tfn
    logabsdetJ::Tlp
    ParameterMapping(transform=identity, logabsdetJ=θ -> zero(eltype(θ))) = new{typeof(transform),typeof(logabsdetJ)}(transform, logabsdetJ)
end

# note that Bijectors.jl maps constrained -> unconstrained, so we need to take the inverse here
ParameterMapping(bijector::Bijectors.Bijector) = ParameterMaping(Bijectors.inverse(bijector), x -> logabsdetjacinv(bijector, x))

# makes ParameterMapping a callable type
(map::ParameterMapping)(θ::AbstractVector) = transform(map, θ)

"""
    transform(map::ParameterMapping, θ)

Invokes the forward map `map.transform` Θ ↦ Φ.
"""
transform(map::ParameterMapping, θ) = map.transform(θ)

"""
    logprob(map::ParameterMapping, θ)

Computes the additive log-probability of the transformation.
"""
logprob(map::ParameterMapping, θ) = map.logabsdetJ(θ)
