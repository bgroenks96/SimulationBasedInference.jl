using Distributed
using DrWatson

# python interop
using PythonCall
using SimulationBasedInference.PySBI

# plotting
import CairoMakie as Makie

# RNG
import Random

# RNG with fixed random seed for reproducibility
const rng = Random.MersenneTwister(1234)

const datadir = joinpath("examples", "data", "Tsurf")
const outdir = mkpath(joinpath("examples", "Tsurf_inv", "output"))

overwrite = false

addprocs(Int(length(Sys.cpu_info()) // 2))

@everywhere begin

# inference/statistics packages
using ArviZ
using SimulationBasedInference
using Statistics
using Turing

# diffeq
using OrdinaryDiffEq

# utility
using DataFrames, CSV
using Dates
using Impute
using Printf
using ProgressMeter
using Rasters, NCDatasets
using TimeSeries
using Unitful

include("forward_model.jl")
include("inverse_model.jl")

end

function load_borehole_dataset(filename::String)
    # load NetCDF file
    raster = Raster(filename, name=:Ts)
    # rebuild Raster with modified dimensions (e.g. dropping missing type)
    depths = float.(dims(raster, :depth))*u"cm"
    timestamps = collect(dims(raster, Ti))
    interpolated_data = Impute.impute(raster.data', Impute.Interpolate())
    return Raster(interpolated_data', (Z(depths), Ti(timestamps)), name=:Ts)
end

function plot_predicted_temperature_profiles!(ax, pred_summary, z_pred, color)
    band = Makie.band!(ax, z_pred, pred_summary[Symbol("hdi_3%")], pred_summary[Symbol("hdi_97%")], color=(color, 0.3))
    Makie.rotate!(band, π/2)
    lines = Makie.lines!(ax, z_pred, pred_summary[:mean], linewidth=4.0, color=color)
    Makie.rotate!(lines, π/2)
    return (; band, lines)
end

Ts_dataset = load_borehole_dataset(joinpath(datadir, "SaHole2006_2006-2021_Ts.nc"))
Ts_tspan = Ts_dataset[Ti(Where(t -> tspan[1] <= t <= tspan[end])), Z(Where(>=(100.0u"cm")))]
# Ts_top_annual = collapse(TimeArray(collect(dims(Ts_tspan, Ti)), Ts_tspan[Z(1)]), Year(1), first, mean)
tspan = (DateTime(2015,9,1), DateTime(2020,9,1))
Ts_profile_initial = Ts_dataset[Ti(Near(tspan[1])), Z(Where(>=(100.0u"cm")))]
Ts_profile_initial_mean = mean(Ts_dataset[Ti(Where(t -> tspan[1] <= t <= tspan[1]+Year(1))), Z(Where(>=(100.0u"cm")))], dims=Ti)[:,1]
obs_depths = ustrip.(u"m", dims(Ts_profile_target, Z))
@assert !any(ismissing.(Ts_profile_target)) "Missing data in target T profile!"

met_dataset_full = DataFrame(CSV.File("examples/data/Tsurf/Samoylov_level2.dat", missingstring="NA", comment="#"))
met_dataset_full[!,:UTC] .= map(t -> DateTime(t, "YYYY-mm-dd HH:MM:ss"), met_dataset_full.UTC)
met_dataset = filter(row -> tspan[1] <= row.UTC <= tspan[2], met_dataset_full)
Tair = TimeArray(met_dataset.UTC, Impute.interp(met_dataset.Tair_200))
Tair_annual = collapse(Tair, Year(1), first, mean)
Tair_amp = (collapse(Tair, Year(1), first, x -> quantile(x, 0.99)) .- collapse(Tair, Year(1), first, x -> quantile(x, 0.01))) ./ 2
Tair_offsets = Tair_annual[2:end] .- mean(values(Tair_annual[2:end]))
Ts_center_annual = collapse(TimeArray(met_dataset.UTC, Impute.interp(met_dataset.Ts_center_1)), Year(1), first, mean)
Makie.lines(met_dataset.Ts_center_1)
Makie.lines(Ts_dataset[Z(1),Ti(Where(t -> t > tspan[1] && t < tspan[2]))])

let fig = Makie.Figure(),
    ax = Makie.Axis(fig[1,1], yreversed=true, ylabel="Depth / m");
    l1 = Makie.lines!(ax, Ts_profile_initial, ustrip.(u"m", dims(Ts_profile_initial, Z)))
    l2 = Makie.lines!(ax, Ts_profile_target[:,1], ustrip.(u"m", dims(Ts_profile_target, Z)))
    Makie.axislegend(ax, [l1,l2], [string(year(tspan[1])), string(year(tspan[end]))], position=:rb)
    fig
end

soilprofile_samoylov = SoilProfile(
    0.0u"m" => MineralOrganic(por=0.80,sat=1.0,org=0.50),
    0.4u"m" => MineralOrganic(por=0.55,sat=1.0,org=0.25),
    3.0u"m" => MineralOrganic(por=0.50,sat=1.0,org=0.0),
    10.0u"m" => MineralOrganic(por=0.30,sat=1.0,org=0.0),
)
discretization = AutoGrid(spacing=LinearSpacing(min_thick=10.0u"cm"))
tempprofile = TemperatureProfile(map(Pair, dims(Ts_profile_initial, Z), Ts_profile_initial.*u"°C")...)
initT = initializer(:T, tempprofile)
t_knots = tspan[1]:Year(1):tspan[2]-Year(1)
forward_prob = set_up_Tsurf_forward_problem(
    soilprofile_samoylov,
    discretization,
    initT,
    tspan,
    t_knots,
    obs_depths,
    heatop=Heat.EnthalpyImplicit(),
);

target_profile = collect(skipmissing(Ts_profile_target.data[:,1]))
inference_prob = set_up_Tsurf_inference_problem(
    forward_prob,
    LiteImplicitEuler(),
    t_knots,
    target_profile,
    obs_depths;
    Tair_amp=mean(values(Tair_amp)),
)

# θ₀ = sample(inference_prob.prior)
# loglik = @time SBI.forward_eval!(inference_prob, θ₀)
# T_ub = retrieve(inference_prob.observables.T_ub)
# Ts_out = retrieve(inference_prob.observables.Ts)
# Makie.lines(T_ub)
# Makie.lines(Ts_out[2,:])

config = Dict(
    "ensemble_size" => 256
)

# eks_solver = init(inference_prob, EKS(maxiters=10), EnsembleDistributed(), ensemble_size=config["ensemble_size"])

eks_output, _ = produce_or_load(config, outdir, filename="eks_inference_data", force=overwrite) do config
    sol = solve(inference_prob, EKS(maxiters=10), EnsembleDistributed(), ensemble_size=config["ensemble_size"]);
    @strdict sol
end;
eks_sol = eks_output["sol"]
eks_summary = SBI.summarize(eks_sol)
eks_posterior = get_transformed_ensemble(eks_sol)
eks_preds = get_observables(eks_sol)
eks_preds_summary = SBI.summarize(transpose(eks_preds.Ts_pred))
eks_T_ub_summary = SBI.summarize(eks_preds.T_ub')
Makie.lines(eks_T_ub_summary[:mean])

# Makie.lines(mean(eks_preds.T_ub, dims=2)[:,1])

esmda_output, _ = produce_or_load(config, outdir, filename="esmda_inference_data", force=overwrite) do config
    sol = solve(inference_prob, ESMDA(maxiters=10), EnsembleDistributed(), ensemble_size=config["ensemble_size"]);
    @strdict sol
end;
esmda_sol = esmda_output["sol"];
esmda_posterior = get_transformed_ensemble(esmda_sol)
esmda_preds = get_observables(esmda_sol)
esdma_preds_summary = SBI.summarize(transpose(esmda_preds.Ts_pred))

snpe_config = Dict("n_samples" => 1000)
snpe_output, _ = produce_or_load(snpe_config, outdir, filename="snpe_inference_data", force=overwrite) do config
    # eks_sim_data = SBI.sample_ensemble_predictive(eks_sol)
    sol = solve(inference_prob, PySNPE(), num_simulations=1000);
    posterior = sample(sol.result, config["n_samples"])
    observables = progress_map(eachcol(posterior)) do θ
        θ = zero(inference_prob.u0) + θ
        ϕ = SBI.forward_map(inference_prob.prior, θ)
        sol = solve(forward_prob, LiteImplicitEuler(), p=ϕ.model)
        map(retrieve, sol.prob.observables)
    end
    preds = SBI.ntreduce(hcat, observables)
    @strdict sol posterior preds
end;
snpe_sol = snpe_output["sol"];
snpe_preds = snpe_output["preds"];
snpe_preds_summary = SBI.summarize(transpose(snpe_preds.Ts_pred))

# emcee_config = Dict("n_samples" => 1000, "n_chains" => 32)
# emcee_output, _ = produce_or_load(emcee_config, outdir, filename="emcee_inference_data") do config
#     sol = solve(inference_prob, MCMC(SBI.Emcee()), num_samples=config["n_samples"], num_chains=config["n_chains"]);
#     @strdict sol
# end;

param_map = SBI.forward_map(inference_prob.prior.model)
eks_posterior_Tsurf = reduce(hcat, map(θ -> param_map(θ).top[Symbol("Tsurf.value")], eachcol(eks_posterior)))
eks_posterior_amp = reduce(hcat, map(θ -> param_map(θ).top[Symbol("amp.value")], eachcol(eks_posterior)))
Makie.density(eks_posterior_amp[end,:])
Makie.lines(mean(eks_posterior_Tsurf, dims=2)[:,1])

let fig = Makie.Figure(size=(1200,600), fontsize=22);
    # Left panel
    ts = tspan[1]+Day(1):Day(1):tspan[end]
    xs = 1:length(ts)
    xticks = (1:180:length(ts), Dates.format.(ts[1:180:end], "YYYY-mm"))
    ax1 = Makie.Axis(fig[1,1:2]; xticks, xticklabelrotation=π/4, ylabel="Temperature / °C", yticks=-25:5.0:20.0)
    eks_T_ub_summary = SBI.summarize(eks_preds.T_ub')
    # esmda_T_ub_summary = SBI.summarize(esmda_preds.T_ub')
    # snpe_T_ub_summary = SBI.summarize(snpe_preds.T_ub')
    Tsurf_obs = collapse(TimeArray(select(met_dataset, :UTC, :Ts_center_1), timestamp=:UTC), Day(1), first, mean)[ts]
    Impute.interp!(values(Tsurf_obs))
    Tsurf_obs_annual = collapse(Tsurf_obs, Year(1), last, mean)
    # Makie.band!(ax1, xs, snpe_T_ub_summary[Symbol("hdi_3%")], snpe_T_ub_summary[Symbol("hdi_97%")], color=(:green,0.2))
    Makie.band!(ax1, xs, eks_T_ub_summary[Symbol("hdi_3%")], eks_T_ub_summary[Symbol("hdi_97%")], color=(:blue,0.2))
    # Makie.band!(ax1, xs, esmda_T_ub_summary[Symbol("hdi_3%")], esmda_T_ub_summary[Symbol("hdi_97%")], color=(:orange,0.2))
    # Makie.band!(ax1, xs, eks_T_ub_summary[Symbol("hdi_3%")], eks_T_ub_summary[Symbol("hdi_97%")], color=(:blue,0.2))
    # Makie.lines!(ax1, xs, snpe_T_ub_summary[:mean], color=:green, alpha=0.6, linewidth=2.0)
    Makie.lines!(ax1, xs, eks_T_ub_summary[:mean], color=:blue, alpha=0.6, linewidth=2.0)
    # Makie.lines!(ax1, xs, esmda_T_ub_summary[:mean], color=:orange, alpha=0.6, linewidth=2.0)
    Makie.lines!(ax1, xs, values(Tsurf_obs), color=:black, linestyle=:dash)
    # Makie.lines!(ax1, findall(∈(timestamp(Tsurf_obs_annual)), ts), values(Tsurf_obs_annual), color=:black)
    
    # Right panel
    ax2 = Makie.Axis(
        fig[1,3],
        ylabel="Depth / m",
        yticks=0.0:5.0:30.0,
        xlabel="Temperature / °C",
        xtickformat=x -> map(xᵢ -> @sprintf("%.1f", ifelse(isapprox(xᵢ, 0.0), 0.0, -xᵢ)), x),
    );
    z_pred = ustrip.(u"m", dims(Ts_profile_target, Z))
    l1 = Makie.scatterlines!(ax2, ustrip.(u"m", dims(Ts_profile_initial_mean, Z)), Ts_profile_initial_mean, color=:gray, linestyle=:dash, linewidth=2)
    Makie.rotate!(l1, π/2)
    # snpe_plot_res = plot_predicted_temperature_profiles!(ax2, snpe_preds_summary, z_pred, :green)
    eks_plot_res = plot_predicted_temperature_profiles!(ax2, eks_preds_summary, z_pred, :blue)
    # esmda_plot_res = plot_predicted_temperature_profiles!(ax2, esdma_preds_summary, z_pred, :orange)
    l2 = Makie.scatterlines!(ax2, z_pred, Ts_profile_target[:,1], linewidth=2, color=:black, linestyle=:dash)
    Makie.rotate!(l2, π/2)
    # Makie.axislegend(ax2,
    #     [collect(eks_plot_res), collect(esmda_plot_res), collect(snpe_plot_res), l1, l2],
    #     ["EKS", "ES-MDA", "SNPE", "Observed $(year(tspan[1]))", "Observed $(year(tspan[end]))"], position=:rb,
    # )
    Makie.xlims!(ax2, 3.5, 9.0)
    Makie.ylims!(ax2, 0.0, 30.0)
    ax2.yreversed = true
    ax2.xreversed = true
    save(joinpath(outdir, "Tsurf_with_predicted_profiles_all.pdf"), fig)
    fig
end
