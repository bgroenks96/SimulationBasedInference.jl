# python interop
using PythonCall

# inference/statistics packages
using ArviZ
using DynamicHMC
using SimulationBasedInference
using SimulationBasedInference.PySBI

# plotting
import CairoMakie as Makie

# RNG
import Random

# RNG with fixed random seed for reproducibility
const rng = Random.MersenneTwister(1234)

const datadir = joinpath("examples", "data", "ddm")

include("datenum.jl")
include("ddm.jl")
include("data.jl")

# Arbitrarily select "true" parameters and run forward model
p_true = ComponentVector(a=2.5, b=0.65)
N_obs = 100
σ_true = 10.0
σ_prior = prior(σ=LogNormal(log(σ_true), 1.0))
data = generate_synthetic_dataset(N_obs, σ_true, p_true; datadir)

# σ_prior = prior(:σ, LogNormal(log(20.0), 1.0))
# data = load_ny_alesund_dataset(Date(2020,9,1), Date(2021,9,1); datadir)

# plot the data
let fig = Makie.Figure(size=(1200,600)),
    xticks = (1:30:length(data.ts), Dates.format.(data.ts[1:30:end], "YYYY-mm")),
    ax1 = Makie.Axis(fig[1,1], xticks=xticks, ylabel="Water equivalent / mm"),
    ax2 = Makie.Axis(fig[2,1], xticks=xticks, ylabel="Temperature / °C");
    Makie.hidexdecorations!(ax1, ticklabels=true, grid=false, ticks=false)
    ax2.xticklabelrotation = π/4
    if haskey(data, :y_true)
        lines = Makie.lines!(ax1, 1:length(data.ts), data.y_true[:,1], linewidth=2.0, color=:gray)
        lines_pr = Makie.lines!(ax1, 1:length(data.precip), cumsum(data.precip)./10, color=:blue)
        points = Makie.scatter!(ax1, data.idx, data.y_obs, color=:black)
        Makie.axislegend(ax1, [lines_pr, lines, points], ["Cumulative precip. x 0.1", "Ground truth", "Pseudo-obs"], position=:lt)
    else
        lines = Makie.lines!(ax1, 1:length(data.precip), cumsum(data.precip), color=:blue)
        points = Makie.scatter!(ax1, data.idx, data.y_obs, color=:black)
        Makie.axislegend(ax1, [points, lines], ["Observed SWE", "Cumulative precip."], position=:lt)
    end
    Makie.lines!(ax2, 1:length(data.ts), data.Tair, linewidth=2.0, color=:lightblue)
    fig
end

# Define observables with time coordinates
y_obs_pred = SimulatorObservable(:y_obs, state -> state.u[data.idx,1], (Ti(data.ts[data.idx]),))
y_pred = SimulatorObservable(:y, state -> state.u[:,1], (Ti(data.ts),))

# Define simple prior
prior_dist = prior(
    a = LogNormal(0,2),
    b = LogNormal(0,2),
)

# Construct forward problem;
# first we need a function of *only* the parameters
function ddm_forward(ts, precip, Tair)
    ddm(θ) = DDM(ts, precip, Tair, θ...)
    return ddm
end

forward_prob = SimulatorForwardProblem(
    ddm_forward(data.ts, data.precip, data.Tair),
    ComponentVector(mean(prior_dist)), # initial parameters, can be anything
    y_obs_pred,
    y_pred,
)

# Define isotropic normal likelihood
lik = IsotropicGaussianLikelihood(y_obs_pred, data.y_obs, σ_prior)

# Construct inference problem
inference_prob = SimulatorInferenceProblem(forward_prob, prior_dist, lik)

function summarize_ensemble(inference_sol, observable=:y)
    # prior/posterior stats
    prior_ens = get_transformed_ensemble(inference_sol, 1)
    posterior_ens = get_transformed_ensemble(inference_sol)
    posterior_mean = median(posterior_ens, dims=2)[:,1]
    posterior_std = std(posterior_ens, dims=2)[:,1]
    # predictions
    obsv = get_observables(inference_sol)
    pred_ens = obsv[observable]
    pred_mean = median(pred_ens, dims=2)
    pred_std = std(pred_ens, dims=2)
    return (; prior_ens, posterior_ens, posterior_mean, posterior_std, pred_ens, pred_mean, pred_std)
end

function summarize_markov_chain(inference_sol, observable=:y)
    stats, qs = describe(inference_sol.result)
    posterior_ens = transpose(Array(inference_sol.result))
    posterior_mean = qs[:,Symbol("50.0%")]
    posterior_std = stats[:,:std]
    pred_ens = reduce(hcat, map(out -> out[observable], getoutputs(inference_sol)))
    pred_mean = median(pred_ens, dims=2)[:,1]
    pred_std = std(pred_ens, dims=2)[:,1]
    return (; posterior_ens, posterior_mean, posterior_std, pred_ens, pred_mean, pred_std)
end

function summarize_snpe(inference_sol, observable=:y; nsamples=10_000)
    posterior_ens = sample(inference_sol.result, nsamples)
    posterior_mean = mean(posterior_ens, dims=2)[:,1]
    posterior_std = std(posterior_ens, dims=2)[:,1]
    pred_ens = map(eachcol(posterior_ens)) do p
        ϕ = zero(inference_sol.prob.u0) + p
        sol = solve(forward_prob, p=ϕ.model)
        retrieve(sol.prob.observables[observable])
    end
    pred_ens = reduce(hcat, pred_ens)
    pred_mean = mean(pred_ens, dims=2)
    pred_std = std(pred_ens, dims=2)
    return (; posterior_ens, posterior_mean, posterior_std, pred_ens, pred_mean, pred_std)
end

function plot_density!(ax, res, idx::Integer, color; offset=0.0)
    samples = res.posterior_ens[idx,:]
    d = Makie.density!(ax, samples, color=(color,0.5), offset=offset)
    vl = Makie.vlines!(ax, [res.posterior_mean[idx]], color=(color,0.75), linewidth=2.0)
    return (; d, vl)
end

function plot_predictions!(ax, res, ts, color; hdi_prob=0.90)
    # compute highest density interval (HDI) using ArviZ
    pred_hdi = mapslices(x -> hdi(x, prob=hdi_prob), res.pred_ens, dims=2)[:,1]
    band = Makie.band!(ax, 1:length(ts), map(intrv -> intrv.lower, pred_hdi), map(intrv -> intrv.upper, pred_hdi), color=(color, 0.5))
    lines = Makie.lines!(ax, 1:length(ts), res.pred_mean[:,1], linewidth=4.0, color=(color,0.75))
    return (; band, lines)
end

# Solve with EnIS
enis_sol = solve(inference_prob, EnIS(), EnsembleThreads(), n_ens=512, verbose=false, rng=rng);
enis_res = summarize_ensemble(enis_sol)

# Solve inference problem with EKS
eks_sol = @time solve(inference_prob, EKS(), EnsembleThreads(), n_ens=512, verbose=false, rng=rng);
eks_res = summarize_ensemble(eks_sol)

# Solve with ESMDA
esmda_sol = @time solve(inference_prob, ESMDA(maxiters=10), EnsembleThreads(), n_ens=512, verbose=false, rng=rng);
esmda_res = summarize_ensemble(esmda_sol)

hmc_sol = @time solve(inference_prob, MCMC(NUTS(), nsamples=2000));
hmc_res = summarize_markov_chain(hmc_sol)
hmc_sol.result

snpe_sol = @time solve(inference_prob, PySNPE(), num_simulations=5000);
snpe_res = summarize_snpe(snpe_sol)

plotsdir = mkpath(joinpath(@__DIR__, "plots"))

# densities
let fig = Makie.Figure(size=(800,600), dpi=300.0),
    ax = Makie.Axis(fig[1,1], xlabel="Degree-day melt factor");
    d1, _ = plot_density!(ax, eks_res, 1, :blue, offset=3.0)
    d2, _ = plot_density!(ax, esmda_res, 1, :orange, offset=2.0)
    d3, _ = plot_density!(ax, snpe_res, 1, :green, offset=1.0)
    d4, _ = plot_density!(ax, hmc_res, 1, :gray, offset=0.0)
    if haskey(data, :y_true)
        vt = Makie.vlines!([p_true[1]], color=:black, linestyle=:dash)
        Makie.axislegend(ax, [d1,d2,d3,d4,vt], ["EKS", "ES-MDA", "SNPE", "HMC", "True value"])
    else
        Makie.axislegend(ax, [d1,d2,d3,d4], ["EKS", "ES-MDA", "SNPE", "HMC"])
    end
    # Makie.xlims!(ax, 1.0, 4.0)
    Makie.save(joinpath(plotsdir, "ddm_factor_posterior_comparison.png"), fig)
    fig
end

let fig = Makie.Figure(size=(800,600)),
    ax = Makie.Axis(fig[1,1], xlabel="Accumulation scale factor");
    d1, _ = plot_density!(ax, eks_res, 2, :blue, offset=15.0)
    d2, _ = plot_density!(ax, esmda_res, 2, :orange, offset=10.0)
    d3, _ = plot_density!(ax, snpe_res, 2, :green, offset=5.0)
    d4, _ = plot_density!(ax, hmc_res, 2, :gray)
    if haskey(data, :y_true)
        vt = Makie.vlines!([p_true[2]], color=:black, linestyle=:dash)
        Makie.axislegend(ax, [d1,d2,d3,d4,vt], ["EKS", "ES-MDA", "SNPE", "HMC", "True value"])
    else
        Makie.axislegend(ax, [d1,d2,d3,d4], ["EKS", "ES-MDA", "SNPE", "HMC"])
    end
    Makie.save(joinpath(plotsdir, "acc_factor_posterior_comparison.png"), fig)
    fig
end

# predictions
let fig = Makie.Figure(size=(900,600)),
    ax = Makie.Axis(fig[1,1], xlabel="Day of year", ylabel="SWE / mm");
    if haskey(data, :y_true)
        yt = Makie.lines!(ax, 1:length(data.ts), data.y_true[:,1], linewidth=2.0, color=:black)
    end
    yo = Makie.scatter!(ax, data.idx, data.y_obs, color=:black)
    plt1 = plot_predictions!(ax, eks_res, data.ts, :blue)
    plt2 = plot_predictions!(ax, esmda_res, data.ts, :orange)
    plt3 = plot_predictions!(ax, snpe_res, data.ts, :green)
    plt4 = plot_predictions!(ax, hmc_res, data.ts, :gray)
    plots = [collect(plt1), collect(plt2), collect(plt3), collect(plt4), yo]
    names = ["EKS", "ES-MDA", "SNPE", "HMC", "Pseudo-obs"]
    if haskey(data, :y_true)
        push!(plots, yt)
        push!(names, "Ground truth")
    end
    Makie.axislegend(ax, plots, names)
    Makie.save(joinpath(plotsdir, "posterior_predictive_comparison.png"), fig)
    fig
end
