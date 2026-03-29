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
