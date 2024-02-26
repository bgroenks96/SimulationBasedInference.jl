using Downloads: download
using MAT
using Plots

const download_url = "https://www.dropbox.com/scl/fi/fbxn7antmrchk39li44l6/daily_forcing.mat?rlkey=u1s2lu13f4grqnbxt4ediwlk2&dl=0"

include("datenum.jl")
include("ddm.jl")

datadir = mkpath(joinpath("examples", "data"))
filepath = joinpath(datadir, "finse_tp.mat")
if !isfile(filepath)
    @info "Downloading forcing data to $filepath"
    download(download_url, filepath)
end

data = matread(filepath)
forcing = data["f"]
ts = todatetime.(DateNumber.(forcing["t"]))[:,1]
precip = forcing["P"][:,1]
Tair = forcing["T"][:,1]

a = [1.0,3.0,10.0]
b = [1.0,1.0,5.0]

# uncomment to debug in VSCode
#@run D=DDM(ts,P,T,a,b)

D = @time DDM(ts, precip, Tair, a, b)

plot(D)

# define the system as an ODE instead;
# there probably isn't any real benefit to this since the simple DDM model
# is autonomous w.r.t the state; i.e. only the forcings actually change the system.
using OrdinaryDiffEq
using DiffEqCallbacks

dudt, resid, u0, p, tspan = DDM_ode(ts, precip, Tair);
# define ODEProblem with "positive domain" callback
prob = ODEProblem(dudt, u0, tspan, p, callback=PositiveDomain(u0))
sol = @time solve(prob, Heun(), dtmax=24*3600.0)
D_sol = reduce(vcat, sol.(tspan[1]:24*3600:tspan[end]))
plot([D_sol D[:,1]])

# taking gradients using forward-mode AD
using ForwardDiff
using Statistics

loss(p) = mean(DDM(ts, precip, Tair, p[1], p[2]))

ForwardDiff.gradient(loss, [3.0,1.0])
