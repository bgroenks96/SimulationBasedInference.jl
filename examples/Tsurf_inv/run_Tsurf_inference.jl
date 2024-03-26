# python interop
using PythonCall

# inference/statistics packages
using ArviZ
using DynamicHMC
using SimulationBasedInference
using SimulationBasedInference.PySBI
using Statistics
using Turing

# utility
using Dates
using Impute
using Rasters, NCDatasets
using Unitful

# plotting
import CairoMakie as Makie

# RNG
import Random

# RNG with fixed random seed for reproducibility
const rng = Random.MersenneTwister(1234)

const datadir = joinpath("examples", "data", "Tsurf")

include("forward_model.jl")
include("inverse_model.jl")

function load_borehole_dataset(filename::String)
    # load NetCDF file
    raster = Raster(filename, name=:Ts)[depth=Where(z -> z > 100.0)]
    # rebuild Raster with modified dimensions (e.g. dropping missing type)
    depths = float.(dims(raster, :depth))*u"cm"
    timestamps = collect(dims(raster, Ti))
    interpolated_data = Impute.impute(raster.data', Impute.Interpolate())
    return Raster(interpolated_data', (Z(depths), Ti(timestamps)), name=:Ts)
end

Ts_dataset = load_borehole_dataset(joinpath(datadir, "SaHole2006_2006-2021_Ts.nc"))
obs_depths = ustrip.(u"m", dims(Ts_dataset, Z))
tspan = (DateTime(2010,9,1), DateTime(2021,9,1))
Ts_profile_initial = Ts_dataset[Ti(Near(tspan[1]))]
Ts_profile_target = mean(Ts_dataset[Ti(Where(t -> tspan[end]-Year(1) <= t <= tspan[end]))], dims=Ti)
@assert !any(ismissing.(Ts_profile_target)) "Missing data in target T profile!"

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
);

target_profile = collect(skipmissing(Ts_profile_target.data[:,1]))
inference_prob = set_up_Tsurf_inference_problem(forward_prob, CGEuler(), t_knots, target_profile, obs_depths)

θ₀ = sample(inference_prob.prior)
p0 = SBI.forward_map(inference_prob.prior, θ₀)
loglik = SBI.forward_eval!(inference_prob, p0)

T_ub = retrieve(inference_prob.observables.T_ub)
Makie.lines(T_ub)

integrator = init(remake(forward_prob.prob, p=inference_prob.prior.model(p0.model)), CGEuler())
step!(integrator, 24*3600*365)

# sol = @time solve(forward_prob, CGEuler());
