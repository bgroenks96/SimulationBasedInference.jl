using Turing

@model function priormodel_Tsurf_with_amplitude(
    ts::AbstractVector,
    p::ComponentVector,
    ::Type{T}=Float64;
    amp_scale=20.0,
) where {T}
    N = length(ts) - 1
    μ₀ ~ Normal(-10.0, 2.0)
    T₀_offset ~ Normal(0.0, 1.0)
    σ_Tsurf ~ Exponential(2.0)
    σ_logamp ~ Exponential(0.1)
    offset ~ MvNormal(zeros(N), 1.0)
    logamp ~ MvNormal(zeros(N), 1.0)
    T₀ = μ₀ .+ T₀_offset.*σ_Tsurf
    Tsurf = T₀ .+ offset.*σ_Tsurf
    p.top["Tsurf.value"] .= Tsurf
    p.top["Tsurf.initialvalue"] .= T₀
    p.top["amp.value"] .= amp_scale.*exp.(logamp.*σ_logamp)
    p.top["amp.initialvalue"] .= amp_scale
    return p
end

function set_up_Tsurf_inference_problem(
    forward_prob::SimulatorForwardProblem,
    forward_solver,
    t_knots::AbstractVector,
    Ts_target,
    obs_depths;
    noise_scale=0.1,
)
    tile = Tile(forward_prob.prob.f)
    p0 = ustrip.(vec(CryoGrid.parameters(tile)))
    # use simple prior for exact Kneier '18 stratigraphy,
    # otherwise use "robust" method that includes soil parameters.
    m_prior = priormodel_Tsurf_with_amplitude(t_knots, p0)
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
