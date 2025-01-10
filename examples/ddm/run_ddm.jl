using Downloads: download
using MAT

import CairoMakie as Makie

include("datenum.jl")
include("ddm.jl")
include("data.jl")

forcings = load_finse_era5_forcings()

a = [1.0,3.0,10.0]
b = [1.0,1.0,5.0]

# uncomment to debug in VSCode
#@run D=DDM(ts,P,T,a,b)

D = @time DDM(forcings.ts, forcings.precip, forcings.Tair, a, b)

Makie.lines(D[:,1])

# taking gradients using forward-mode AD
using ForwardDiff
using Statistics

loss(p) = mean(DDM(forcings.ts, forcings.precip, forcings.Tair, p[1], p[2]))

ForwardDiff.gradient(loss, [3.0,1.0])

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
Makie.series([D_sol D[:,1]]')
