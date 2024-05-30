using Dates
using Gen
using LinearAlgebra, StatsFuns

"""
Represents the precipitation state at time `t`.
"""
struct PrecipState{T}
    precip::T
    t::Date
end

struct AnnualStats{T}
    min::T
    mid::T
    max::T
    peak_month::Int
    function AnnualStats(min::T, mid::T, max::T, peak_month::Int) where {T}
        @assert min < mid < max
        return new{T}(min, mid, max, peak_month)
    end
end

StatsFuns.logistic(b::Gen.DistWithArgs{T}) where T <: Real =
    Gen.DistWithArgs(Gen.TransformedDistribution{T, T}(b.base, 0, logistic, logit, x -> (1.0 / (x-x^2),)), b.arglist)

@dist reparameterized_normal(μ, σ) = μ + normal(0,1)*σ

@gen function logitnormal(μ, σ)
    z ~ reparameterized_normal(μ, σ)
    return logistic(z)
end

@gen function positive_cosine_transform(level_min, level_mid, level_max, phase_shift)
    dist = SimulationBasedInference.from_moments(LogNormal, level_mid, (level_max - level_min)/2)
    log_level ~ reparameterized_normal(log(level_mid), dist.σ)
    amp_scale ~ logitnormal(0.0, 1.0)
    level = exp(log_level)
    amp = amp_scale*abs(level)
    return ComponentVector(; level, amp, phase_shift)
end

@gen function logistic_cosine_transform(level_min, level_mid, level_max, phase_shift)
    level ~ logitnormal(logit(level_mid), 0.5)
    amp_scale ~ logitnormal(logit((level_max - level_min)/2), 1.0)
    amp = amp_scale*min(1 - level, level)
    return ComponentVector(; level, amp, phase_shift)
end

function apply_cosine_transform(n::Int, params, t)
    return params.level + params.amp*cos(2π/n*(t - params.phase_shift))
end

@gen function richardson_params(pwd::AnnualStats, pdd::AnnualStats, precip::AnnualStats)
    logit_p_sigma ~ normal(0.0, 0.5)
    log_precip_sigma ~ normal(-1.0, 0.5)
    pwd = @trace(logistic_cosine_transform(pwd.min, pwd.mid, pwd.max, pwd.peak_month), :pwd)
    pdd = @trace(logistic_cosine_transform(pdd.min, pdd.mid, pdd.max, pdd.peak_month), :pdd)
    precip_mean = @trace(positive_cosine_transform(precip.min, precip.mid, precip.max, precip.peak_month), :precip_mean)
    log_gamma_shape ~ normal(0.0, 0.5)
    gamma_shape = exp(log_gamma_shape)
    return ComponentVector(; pwd, pdd, precip_mean, gamma_shape)
end

@gen function richardson_kernel(i::Int, state::PrecipState, params)
    t = state.t + Day(1)
    m = month(t)
    pwd = apply_cosine_transform(12, params.pwd, m)
    pdd = apply_cosine_transform(12, params.pdd, m)
    precip_mean = apply_cosine_transform(12, params.precip_mean, m)
    gamma_shape = params.gamma_shape
    gamma_scale = precip_mean / gamma_shape
    @assert gamma_shape > 0 "$params"
    @assert gamma_scale > 0 "$params"
    p = ifelse(state.precip > 0, pwd, pdd)
    dry ~ bernoulli(p)
    if dry
        return PrecipState(zero(state.precip), t)
    else
        precip = @trace(gamma(gamma_shape, gamma_scale), :precip)
        return PrecipState(precip, t)
    end
end

@gen (static) function richardson_precip(params, initialstate::PrecipState, nsteps::Int)
    outputs = @trace(Unfold(richardson_kernel)(nsteps, initialstate, params), :state)
    return outputs
end

@gen (static) function richardson_model(initialstate::PrecipState, nsteps::Int)
    params = @trace(richardson_params(12), :para)
    outputs = @trace(richardson_precip(params, initialstate, nsteps), :model)
    return outputs
end

function get_precip_from(trace)
    retval = Gen.get_retval(trace)
    return collect(map(state -> state.precip, retval))
end

function get_timestamps_from(trace)
    retval = Gen.get_retval(trace)
    return collect(map(state -> state.t, retval))
end

function get_para_from(trace)
    choices = Gen.get_choices(trace)
    return Gen.get_selected(choices, Gen.select(:para))
end

function richardson_precip_obs(idx::Int, precip)
    if ismissing(precip)
        choicemap()
    elseif precip > 0
        choicemap((:model => :state => idx => :dry, false), (:model => :state => idx => :precip, precip))
    else
        choicemap((:model => :state => idx => :dry, true),)
    end
end
