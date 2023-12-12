"""
    ParameterTransform{Tfn,Tlp}

Represents a callable transform which maps between different parameter spaces. `transform` should be a
function which performs the transformation f: Θ ↦ Φ and `logabsdetJ` should compute `log|det(J(f))|` where
`J(f)` is the Jacobian of `f`, i.e. ∂f/∂θ.
"""
struct ParameterTransform{Tfn,Tlp}
    transform::Tfn
    logabsdetJ::Tlp
    ParameterTransform(transform=identity, logabsdetJ=θ -> zero(eltype(θ))) = new{typeof(transform),typeof(logabsdetJ)}(transform, logabsdetJ)
end

# makes ParameterTransform a callable type
(map::ParameterTransform)(θ::AbstractVector) = transform(map, θ)

"""
    transform(map::ParameterTransform, θ)

Invokes the forward map `map.transform` Θ ↦ Φ.
"""
transform(map::ParameterTransform, θ) = map.transform(θ)

"""
    logdensity(map::ParameterTransform, θ)

Computes the additive log-probability of the transformation.
"""
logdensity(map::ParameterTransform, θ) = map.logabsdetJ(θ)
