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

StatsFuns.logistic(b::Gen.DistWithArgs{T}) where T <: Real =
    Gen.DistWithArgs(Gen.TransformedDistribution{T, T}(b.base, 0, logistic, logit, x -> (1.0 / (x-x^2),)), b.arglist)

@dist logitnormal(μ, σ) = logistic(normal(μ, σ))

@gen function iid_logitnormal(μ::AbstractVector, σ::Real)
    x ~ mvnormal(μ, Diagonal(zero(μ) .+ σ))
    return logistic.(x)
end

@gen function richardson_params(n::Int, pwd_mean=0.5, pdd_mean=0.5, precip_mean=5.0)
    log_p_sigma ~ normal(log(0.5), 0.2)
    p_sigma = exp(log_p_sigma)
    pwd0 ~ logitnormal(logit(pwd_mean), p_sigma)
    pdd0 ~ logitnormal(logit(pdd_mean), p_sigma)
    log_precip_std ~ normal(-2.0, 0.5)
    log_precip_mean0 ~ normal(log(precip_mean), exp(log_precip_std))
    lpwd = @trace(mvnormal(logit.(pwd0*ones(n)), Diagonal(ones(n)*p_sigma)), :lpwd)
    lpdd = @trace(mvnormal(logit.(pdd0*ones(n)), Diagonal(ones(n)*p_sigma)), :lpdd)
    log_precip_mean ~ mvnormal(log_precip_mean0*ones(n), Diagonal(ones(n)*exp(log_precip_std)))
    log_gamma_shape ~ normal(log(2.0), 0.1)
    return (; lpwd, lpdd, log_precip_mean, log_gamma_shape)
end

@gen function richardson_kernel(i::Int, state::PrecipState, params)
    t = state.t + Day(1)
    m = month(t)
    pwd = logistic(params.lpwd[m])
    pdd = logistic(params.lpdd[m])
    gamma_shape = exp(params.log_gamma_shape)
    gamma_scale = exp(params.log_precip_mean[m]) / gamma_shape
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
