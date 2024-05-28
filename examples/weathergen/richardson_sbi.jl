using CSV, DataFrames, Impute
using PythonCall
using SimulationBasedInference
using SimulationBasedInference.PySBI
using TimeSeries

import Plots

include("richardson_precip.jl")

precip_data_pluvio = DataFrame(CSV.File("examples/data/ddm/NYA_pluvio_l1_precip_daily_v00_2017-2022.csv"))
precip_data_pluvio_2018 = filter(row -> year(row.time) == 2018, precip_data_pluvio)

Plots.plot(precip_data_pluvio_2018.prec)

"""
    cumulative_dry_days(prec::AbstractVector{T}, agg=mean) where {T}

Computes cumulative dry days (CDD) over precipication series `prec` with aggregation function `agg`.
"""
function cumulative_dry_days(prec::AbstractVector{T}, agg=mean) where {T}
    dry_spells = [0]
    for i in eachindex(prec)
        if prec[i] > zero(T) && dry_spells[end] > 0
            push!(dry_spells, 0)
        elseif iszero(prec[i])
            dry_spells[end] += 1
        end
    end
    return agg(dry_spells)
end

function precip_summary_stats(ts::AbstractVector, prec::AbstractVector)
    p99 = quantile(filter(>(0), prec), 0.99)
    p01 = quantile(filter(>(0), prec), 0.01)
    pmean = mean(filter(>(0), prec))
    ptot = sum(prec)
    wd = sum(prec .> 0.0)
    cdd = cumulative_dry_days(prec)
    # prec_tarr = TimeArray(ts, prec)
    # ptot_monthly = collapse(prec_tarr, month, first, sum)
    # return vcat([wd,dd,p99,pmean,ptot], values(ptot_monthly))
    return ComponentVector(; p99, p01, pmean, ptot, wd, cdd)
end

data = Impute.fill(precip_data_pluvio_2018, value=0.0)
stats_obs = precip_summary_stats(data.time[2:end], data.prec[2:end])
t0 = data.time[1]
pr0 = data.prec[1]
initialstate = PrecipState(pr0, t0)
genprior = prior(richardson_params, (12,))
θ₀ = rand(genprior)
Nt = 364
precip_sim = SimulatorObservable(:prec, sol -> get_precip_from(sol.u), (364,))
stats_sim = SimulatorObservable(:stats, sol -> precip_summary_stats(get_timestamps_from(sol.u), get_precip_from(sol.u)), (length(stats_obs),))
forward_prob = SimulatorForwardProblem(richardson_precip, (initialstate, Nt), θ₀, precip_sim, stats_sim)
implicit_lik = ImplicitLikelihood(forward_prob.observables.stats, stats_obs)
inference_prob = SimulatorInferenceProblem(forward_prob, genprior, implicit_lik)
inference_sol = solve(inference_prob, PySNE(), num_simulations=10_000)
posterior_samples = Float64.(sample(inference_sol.result, 1000))
precip_samples = map(eachcol(posterior_samples)) do x
    get_precip_from(solve(forward_prob, p=x).sol)
end
precip_samples = reduce(hcat, precip_samples)
mean(map(x -> precip_summary_stats(data.time[2:end], x), eachcol(precip_samples)))
stats_obs
Plots.plot(precip_samples[:,10])
Plots.plot(precip_data_pluvio_2018.prec[2:end])
