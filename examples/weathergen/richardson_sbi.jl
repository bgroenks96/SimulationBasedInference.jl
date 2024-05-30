using CSV, DataFrames, Impute
using PythonCall
using SimulationBasedInference
using SimulationBasedInference.PySBI
using TimeSeries

import Plots, StatsPlots
import Random

const rng = Random.MersenneTwister(1234)

include("richardson_precip.jl")

selectyear(df, sel_year) = filter(row -> year(row.time) == sel_year, df)

precip_data_pluvio = DataFrame(CSV.File("examples/data/ddm/NYA_pluvio_l1_precip_daily_v00_2017-2022.csv"))
precip_data_pluvio_sub = selectyear(precip_data_pluvio, 2020)
data = Impute.fill(precip_data_pluvio_sub, value=0.0)

Plots.plot(precip_data_pluvio_sub.prec)

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
    nzprec = filter(>(0), prec)
    p99 = length(nzprec) > 0 ? quantile(nzprec, 0.99) : 0.0
    p01 = length(nzprec) > 0 ? quantile(filter(>(0), prec), 0.01) : 0.0
    pmean = mean(nzprec)
    ptot = sum(prec)
    wd = sum(prec .> 0.0)
    cdd = cumulative_dry_days(prec)
    # prec_tarr = TimeArray(ts, prec)
    # ptot_monthly = collapse(prec_tarr, month, first, sum)
    stats = ComponentVector(; p99, p01, pmean, ptot, wd, cdd)
    return stats
    # return vcat(stats, values(ptot_monthly))
end

stats_obs = precip_summary_stats(data.time[2:end], data.prec[2:end])
t0 = data.time[1]
pr0 = data.prec[1]
initialstate = PrecipState(pr0, t0)
richardson_prior_args = (
    pwd = AnnualStats(0.2, 0.4, 0.6, 1),
    pdd = AnnualStats(0.6, 0.7, 0.8, 2),
    precip = AnnualStats(1.0, 4.0, 7.0, 10),
)
genprior = prior(richardson_params, values(richardson_prior_args))
θ₀ = genprior(rand(genprior))
Nt = 364
precip_sim = SimulatorObservable(:prec, sol -> get_precip_from(sol.u), (364,))
stats_sim = SimulatorObservable(:stats, sol -> precip_summary_stats(get_timestamps_from(sol.u), get_precip_from(sol.u)), (length(stats_obs),))
forward_prob = SimulatorForwardProblem(richardson_precip, (initialstate, Nt), θ₀, precip_sim, stats_sim)
implicit_lik = ImplicitLikelihood(forward_prob.observables.stats, stats_obs)

# prior predictive
prior_samples = rand(rng, genprior, 1000);
prior_pred = map(prior_samples) do x
    forward_sol = solve(forward_prob, p=genprior(x))
    prec = get_precip_from(forward_sol.sol)
    ts = get_timestamps_from(forward_sol.sol)
    precip_summary_stats(ts, prec)
end
@show mean(reduce(hcat, prior_pred), dims=2)[:,1]
@show maximum(reduce(hcat, prior_pred), dims=2)[:,1]
@show minimum(reduce(hcat, prior_pred), dims=2)[:,1]

# StatsPlots.histogram(filter(>(0), data.prec))

# posterior inference
inference_prob = SimulatorInferenceProblem(forward_prob, genprior, implicit_lik)
inference_sol = solve(inference_prob, PySNE(), num_simulations=10_000)

# draw posterior samples and generate predictions
posterior_samples = Float64.(sample(inference_sol.result, 1000))
posterior_mean = ComponentArray(mean(posterior_samples, dims=2)[:,1], genprior.axes)
posterior_std = ComponentArray(std(posterior_samples, dims=2)[:,1], genprior.axes)
precip_samples = map(eachcol(posterior_samples)) do x
    reduce(hcat, [get_precip_from(solve(forward_prob, p=genprior(x)).sol) for i in 1:10])
end
precip_samples = reduce(hcat, precip_samples)

@show pred_mean = mean(reduce(hcat, map(x -> precip_summary_stats(data.time[2:end], x), eachcol(precip_samples))), dims=2);
@show pred_std = std(reduce(hcat, map(x -> precip_summary_stats(data.time[2:end], x), eachcol(precip_samples))), dims=2);
@show pred_mean[:,1] .- stats_obs;

Plots.plot(precip_samples[:,3])
Plots.plot!(Impute.fill(selectyear(precip_data_pluvio, 2019), value=0.0).prec[2:end])

precdf = Impute.fill(precip_data_pluvio, value=0.0)
tarr = TimeArray(precdf.time, precdf.prec)
monthly_precip = collapse(tarr, Month(1), first, sum)
Plots.plot(monthly_precip)
