using Downloads: download
using MAT
using Plots

const URL = "https://www.dropbox.com/scl/fi/fbxn7antmrchk39li44l6/daily_forcing.mat?rlkey=u1s2lu13f4grqnbxt4ediwlk2&dl=0"

include("../datenum.jl")
include("ddm.jl")

filepath = download(URL, joinpath("examples", "data", "finse_tp.mat"))
data = matread(filepath)
forcing = data["f"]
ts = todatetime.(DateNumber.(forcing["t"]))
P=forcing["P"]
T=forcing["T"]

a=[3.0,10.0]
b=[1.0,5.0]

#@run D=DDM(ts,P,T,a,b)
D=DDM(ts,P,T,a,b)
plot(D)

using ForwardDiff

loss(p) = mean(DDM(ts,P,T,p[1],p[2]))

ForwardDiff.gradient(loss, [3.0,1.0])
