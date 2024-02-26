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

p_true = [2.5, 0.65]
y_true = @time DDM(ts, precip, Tair, p_true...)
# add observation noise ϵ ~ N(0,10)
y_obs = y_true .+ randn(length(y_true)).*10.0
plot(y_true, linewidth=3.0, label="True SWE")
scatter!(y_obs, alpha=0.75, label="Observed SWE")



