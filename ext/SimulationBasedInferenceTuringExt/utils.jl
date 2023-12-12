"""
    ParameterTransform(priormodel::Turing.Model)

Constructs a `ParameterTransform` from the given Turing model. It is assumed that `priormodel`
returns the transformed parameter vector as an output.
"""
function SimulationBasedInference.ParameterTransform(priormodel::Turing.Model)
    params = rand(priormodel)
    param_names = keys(params)
    param_array = ComponentArray(params)
    # get bijector which maps from constrained to unconstrained parameter space
    priormap = bijector(priormodel)
    # get inverse function; i.e. unconstrained -> constrained
    invpriormap = inverse(priormap)
    # define the parameter mapping function
    function param_map(θ::AbstractVector{T}) where {T}
        # map from unconstrained θ-space to constrained ϕ-space and construct ComponentArray from the result.
        ϕ = ComponentArray(invpriormap(θ), getaxes(param_array))
        # here we have to do some nasty Turing manipulation to make sure this function is
        # autodiff compatible; basically we have to reconstruct `varinfo` based on the type of ϕ.
        varinfo = Turing.DynamicPPL.VarInfo(priormodel);
        context = priormodel.context;
        # constructs a new VarInfo type from the parameter values
        new_vi = Turing.DynamicPPL.unflatten(varinfo, context, ϕ);
        # evaluate the Turing model and multiply by one(T) to convert the type
        p = first(Turing.DynamicPPL.evaluate!!(priormodel, new_vi, context))
        return p.*one(T)
    end
    # define logdensity function for transform
    lp(θ) = Turing.logabsdetjacinv(priormap, θ)
    return ParameterTransform(param_map, lp)
end

function extract_parameter_names(m::Turing.Model)
    # sample chain to extract param names
    chain = sample(m, Prior(), 1, progress=false, verbose=false)
    return chain.name_map.parameters
end
