using Downloads: download
using MAT
using Plots, StatsPlots

using SimulationBasedInference
using Turing

import Random

const download_url = "https://www.dropbox.com/scl/fi/fbxn7antmrchk39li44l6/daily_forcing.mat?rlkey=u1s2lu13f4grqnbxt4ediwlk2&dl=0"

# RNG with fixed random seed for reproducibility
const rng = Random.MersenneTwister(1234)

include("datenum.jl")
include("ddm.jl")

# Load or download forcing data
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
σ_true = 8.0
idx = sample(rng, 1:length(ts), N_obs, replace=false)
y_obs = max.(y_true[idx] .+ randn(rng, length(idx)).*σ_true, 0.0)

# plot the data
plot(ts, y_true, linewidth=3.0, label="True SWE")
scatter!(ts[idx], y_obs, alpha=0.75, label="Observed SWE")

# Define observables
y_obs_pred = SimulatorObservable(:y_obs, state -> state.u[idx,1], ndims=length(idx))
y_pred = SimulatorObservable(:y, state -> state.u[:,1], ndims=length(ts))

# Define simple prior
prior = PriorDistribution(
    a = LogNormal(0,1),
    b = LogitNormal(logit(0.5), 1.0)
)

# Construct forward problem;
# first we need a function of *only* the parameters
ddm_forward(θ) = DDM(ts, precip, Tair, θ...)

forward_prob = SimulatorForwardProblem(
    ddm_forward,
    rand(rng, prior), # initial parameters, can be anything
    y_obs_pred,
    y_pred,
)

# Define isotropic normal likelihood
σ_prior = PriorDistribution(:σ, Exponential(15.0))
lik = SimulatorLikelihood(IsoNormal, y_obs_pred, y_obs, σ_prior)

# Construct inference problem
inference_prob = SimulatorInferenceProblem(forward_prob, nothing, prior, lik)

# Solve inference problem with EKS
eks_sol = solve(inference_prob, EKS(), EnsembleThreads(), n_ens=256, verbose=false, rng=rng);
prior_ens = get_transformed_ensemble(eks_sol, 1)
posterior_ens = get_transformed_ensemble(eks_sol)
posterior_mean = mean(posterior_ens, dims=2)[:,1]

density(posterior_ens[1,:], label="a")
vline!([posterior_mean[1]], c=:blue)
vline!([p_true[1]], c=:black, linestyle=:dash)

density(posterior_ens[2,:], label="b")
vline!([posterior_mean[2]], c=:blue)
vline!([p_true[2]], c=:black, linestyle=:dash)

eks_obsv = get_observables(eks_sol)
pred_mean = mean(eks_obsv.y, dims=2)
pred_std = std(eks_obsv.y, dims=2)
plot(ts, y_true, linewidth=2.0, alpha=0.75)
plot!(ts, pred_mean, linewidth=2.0, alpha=0.75, ribbon=pred_std)
