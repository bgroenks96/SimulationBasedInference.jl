using Turing

@model function priormodel_Tsurf_with_amplitude(
    ts::AbstractVector,
    p0::ComponentVector,
    ::Type{T}=Float64;
    T₀_mean=-9.0,
    Tair_amp=28.0,
    σ_Tsurf=0.5,
    period=ustrip(u"s", 1.0u"yr"),
    offset_mean=zeros(length(ts)-1),
) where {T}
    p = similar(p0, T)
    p .= zero(p) .+ p0
    N = length(ts) - 1
    T₀ ~ Normal(T₀_mean, σ_Tsurf)
    phase_shift ~ Normal(0, π/4)
    # period_offset ~ Normal(0, 1.0)
    offset ~ MvNormal(offset_mean, 1.0)
    amp_scale ~ arraydist(LogitNormal.(logit.(ones(N).*0.5), 0.2))
    Tsurf = T₀ .+ offset.*σ_Tsurf
    p.top["Tsurf.value"] .= Tsurf
    p.top["Tsurf.initialvalue"] .= T₀
    p.top["amp.value"] .= Tair_amp.*amp_scale
    p.top["amp.initialvalue"] .= Tair_amp*amp_scale[1]
    p.top["phase_shift"] .= -5π/4 + phase_shift
    p.top["period"] .= period
    return p
end

function set_up_Tsurf_inference_problem(
    forward_prob::SimulatorForwardProblem,
    forward_solver,
    t_knots::AbstractVector,
    Ts_target,
    obs_depths;
    noise_scale=0.05,
    model_kwargs...,
)
    tile = Tile(forward_prob.prob.f)
    p0 = ustrip.(vec(CryoGrid.parameters(tile)))
    # use simple prior for exact Kneier '18 stratigraphy,
    # otherwise use "robust" method that includes soil parameters.
    m_prior = priormodel_Tsurf_with_amplitude(t_knots, p0; model_kwargs...)
    prior = SBI.prior(m_prior)
    # set up inference problem
    Ts_pred_observable = forward_prob.observables.Ts_pred
    Ts_likelihood = IsotropicGaussianLikelihood(
        Ts_pred_observable,
        Ts_target,
        SBI.prior(σ=Exponential(noise_scale)),
    )
    inference_prob = SimulatorInferenceProblem(
        forward_prob,
        forward_solver,
        prior,
        Ts_likelihood;
        metadata=Dict("t_knots" => t_knots, "obs_depths" => obs_depths),
    );
    return inference_prob
end
