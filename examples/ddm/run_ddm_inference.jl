using Downloads: download
using MAT

# inference/statistics packages
using ArviZ
using DynamicHMC
using SimulationBasedInference

# plotting
import CairoMakie as Makie

# RNG
import Random

const download_url = "https://www.dropbox.com/scl/fi/fbxn7antmrchk39li44l6/daily_forcing.mat?rlkey=u1s2lu13f4grqnbxt4ediwlk2&dl=0"

# RNG with fixed random seed for reproducibility
const rng = Random.MersenneTwister(1234)

include("datenum.jl")
include("ddm.jl")

# Download forcing data if not present
datadir = mkpath(joinpath("examples", "data"))
filepath = joinpath(datadir, "finse_tp.mat")
if !isfile(filepath)
    @info "Downloading forcing data to $filepath"
    download(download_url, filepath)
end

# Read foricng data
data = matread(filepath)
forcing = data["f"]
ts = todatetime.(DateNumber.(forcing["t"]))[:,1]
precip = forcing["P"][:,1]
Tair = forcing["T"][:,1]

# Arbitrarily select "true" parameters and run forward model
p_true = [2.5, 0.65]
y_true = @time DDM(ts, precip, Tair, p_true...)

# add noise ϵ ~ N(0,10) to get synthetic observation data
N_obs = 100
σ_true = 5.0
idx = sort(sample(rng, 1:length(ts), N_obs, replace=false))
y_obs = max.(y_true[idx] .+ randn(rng, length(idx)).*σ_true, 0.0)

# plot the data
let fig = Makie.Figure(),
    ax = Makie.Axis(fig[1,1], xticks=(1:30:length(ts), Dates.format.(ts[1:30:end], "YYYY-mm")));
    ax.xticklabelrotation = π/4
    lines = Makie.lines!(ax, 1:length(ts), y_true[:,1], linewidth=2.0)
    points = Makie.scatter!(ax, idx, y_obs, color=:black)
    Makie.axislegend(ax, [lines, points], ["Ground truth", "Pseudo-obs"], position=:rt)
    fig
end

# Define observables
y_obs_pred = SimulatorObservable(:y_obs, state -> state.u[idx,1], ndims=length(idx))
y_pred = SimulatorObservable(:y, state -> state.u[:,1], ndims=length(ts))

# Define simple prior
prior_dist = prior(
    a = LogNormal(0,1),
    b = LogitNormal(logit(0.5), 1.0)
)

# Construct forward problem;
# first we need a function of *only* the parameters
ddm_forward(θ) = DDM(ts, precip, Tair, θ...)

forward_prob = SimulatorForwardProblem(
    ddm_forward,
    rand(rng, prior_dist), # initial parameters, can be anything
    y_obs_pred,
    y_pred,
)

# Define isotropic normal likelihood
σ_prior = prior(:σ, Exponential(10.0))
lik = SimulatorLikelihood(IsoNormal, y_obs_pred, y_obs, σ_prior)

# Construct inference problem
inference_prob = SimulatorInferenceProblem(forward_prob, nothing, prior_dist, lik)

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
    pred_ens = reduce(hcat, map(out -> out[observable], getoutputs(inference_sol.cache)))
    pred_mean = median(pred_ens, dims=2)[:,1]
    pred_std = std(pred_ens, dims=2)[:,1]
    return (; posterior_ens, posterior_mean, posterior_std, pred_ens, pred_mean, pred_std)
end

function plot_density!(ax, res, idx::Integer, color)
    d = Makie.density!(ax, res.posterior_ens[idx,:], color=(color,0.5))
    vl = Makie.vlines!(ax, [res.posterior_mean[idx]], color=(color,0.75), linewidth=2.0)
    return (; d, vl)
end

function plot_predictions!(ax, res, ts, color; hdi_prob=0.90)
    # compute highest density interval (HDI) using ArviZ
    pred_hdi = mapslices(x -> hdi(x, prob=hdi_prob), res.pred_ens, dims=2)[:,1]
    band = Makie.band!(ax, 1:length(ts), map(intrv -> intrv.lower, pred_hdi), map(intrv -> intrv.upper, pred_hdi), color=(color, 0.5))
    lines = Makie.lines!(ax, 1:length(ts), res.pred_mean[:,1], linewidth=2.0, color=(color,0.75))
    return (; band, lines)
end

# Solve with EnIS
# enis_sol = solve(inference_prob, EnIS(), EnsembleThreads(), n_ens=256, verbose=false, rng=rng);
# enis_res = summarize_ensemble(enis_sol)

# Solve inference problem with EKS
eks_sol = @time solve(inference_prob, EKS(), EnsembleThreads(), n_ens=512, verbose=false, rng=rng);
eks_res = summarize_ensemble(eks_sol)

# Solve with ESMDA
esmda_sol = @time solve(inference_prob, ESMDA(), EnsembleThreads(), n_ens=512, verbose=false, rng=rng);
esmda_res = summarize_ensemble(esmda_sol)

hmc_sol = @time solve(inference_prob, MCMC(NUTS(), nsamples=1000));
hmc_res = summarize_markov_chain(hmc_sol)

plotsdir = mkpath(joinpath(@__DIR__, "plots"))

# densities
let fig = Makie.Figure(size=(800,600), dpi=300.0),
    ax = Makie.Axis(fig[1,1], xlabel="Degree-day melt factor");
    d1, _ = plot_density!(ax, eks_res, 1, :blue)
    d2, _ = plot_density!(ax, esmda_res, 1, :orange)
    d3, _ = plot_density!(ax, hmc_res, 1, :gray)
    vt = Makie.vlines!([p_true[1]], color=:black, linestyle=:dash)
    Makie.axislegend(ax, [d1,d2,d3,vt], ["EKS", "ES-MDA", "HMC", "True value"])
    # Makie.xlims!(ax, 1.0, 4.0)
    Makie.save(joinpath(plotsdir, "ddm_factor_posterior_comparison.png"), fig)
    fig
end

let fig = Makie.Figure(size=(800,600)),
    ax = Makie.Axis(fig[1,1], xlabel="Accumulation scale factor");
    d1, _ = plot_density!(ax, eks_res, 2, :blue)
    d2, _ = plot_density!(ax, esmda_res, 2, :orange)
    d3, _ = plot_density!(ax, hmc_res, 2, :gray)
    vt = Makie.vlines!([p_true[2]], color=:black, linestyle=:dash)
    Makie.axislegend(ax, [d1,d2,d3,vt], ["EKS", "ES-MDA", "HMC", "True value"])
    Makie.save(joinpath(plotsdir, "acc_factor_posterior_comparison.png"), fig)
    fig
end

# predictions
let fig = Makie.Figure(size=(900,600)),
    ax = Makie.Axis(fig[1,1], xlabel="Day of year", ylabel="SWE / mm");
    yt = Makie.lines!(ax, 1:length(ts), y_true[:,1], linewidth=2.0, color=:black)
    yo = Makie.scatter!(ax, idx, y_obs, color=:black)
    plt1 = plot_predictions!(ax, eks_res, ts, :blue)
    plt2 = plot_predictions!(ax, esmda_res, ts, :orange)
    plt3 = plot_predictions!(ax, hmc_res, ts, :gray)
    Makie.axislegend(ax, [collect(plt1), collect(plt2), collect(plt3), yo, yt], ["EKS", "ES-MDA", "HMC", "Pseudo-obs", "Ground truth"])
    Makie.save(joinpath(plotsdir, "posterior_predictive_comparison.png"), fig)
    fig
end

using Flux
using SimulationBasedInference.Emulators

# train an emulator for the y_obs observable
X_data = hcat(getinputs(esmda_sol, 1), getinputs(esmda_sol, 5))
y_data = hcat(getoutputs(esmda_sol, :y_obs, 1), getoutputs(esmda_sol, :y_obs, 5))
emulator_data = EmulatorData(X_data, y_data .+ rand(size(y_data)...))
# input_transform = Decorrelated(emulator_data.X, frac=1.0)
# output_transform = Decorrelated(emulator_data.Y, frac=1.0)
# emulator = StackedEmulator(emulator_data, GPRegressor(); input_transform, output_transform)
nn = Flux.Chain(
    Dense(size(X_data,1), 256, Flux.relu),
    Dense(256, size(y_data,1), Flux.relu),
)
emulator = Emulator(emulator_data, nn)
fitted_emulator = Emulators.fit!(emulator, epochs=20, verbose=true)
let fig = Makie.Figure(),
    ax = Makie.Axis(fig[1,1]),
    sample_idx=rand(1:512, 5),
    test_idx=2;
    preds = Emulators.predict(fitted_emulator, getinputs(esmda_sol, test_idx))
    Makie.series!(ax, preds[:,sample_idx]')
    Makie.series!(ax, getoutputs(esmda_sol, :y_obs, test_idx)[:,sample_idx]', linestyle=:dash)
    fig
end

function predict_func(emulator, p)
    preds = Emulators.predict(emulator, p)
    return preds[:,1]
end

em_solver = EmulatedObservables(predict_func, y_obs=fitted_emulator)
emulated_inference_prob = remake(inference_prob, forward_solver=em_solver)
emulated_hmc = solve(emulated_inference_prob, MCMC(NUTS(), autodiff=:Zygote, nsamples=10000, warmup_reporter=DynamicHMC.LogProgressReport()));
emulated_hmc.result
