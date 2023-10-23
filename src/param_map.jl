"""
    ParameterMapping{Tfn,Tlp}

Represents a callable transform which maps between different parameter spaces. `transform` should be a
function which performs the transformation f: Θ ↦ Φ and `logprob` should compute `log|det(J(f))|` where
`J(f)` is the Jacobian of `f`, i.e. ∂f/∂θ.
"""
struct ParameterMapping{Tfn,Tlp}
    transform::Tfn
    logprob::Tlp
    ParameterMapping(transform=identity, logprob=θ -> zero(eltype(θ))) = new{typeof(transform),typeof(logprob)}(transform, logprob)
end

# makes ParameterMapping a callable type
(map::ParameterMapping)(θ::AbstractVector) = transform(map, θ)

"""
    transform(map::ParameterMapping, θ)

Invokes the forward map `map.transform` Θ ↦ Φ.
"""
transform(map::ParameterMapping, θ) = map.transform(θ)

"""
    transform(map::ParameterMapping, θ)

Computes the additive log-probability of the transformation.
"""
logprob(map::ParameterMapping, θ) = map.logprob(θ)
