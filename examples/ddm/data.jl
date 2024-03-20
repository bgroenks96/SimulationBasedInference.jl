using CSV, DataFrames
using Dates
using Downloads: download
using Impute
using MAT
using Statistics

const datadir = joinpath("examples", "data", "ddm")

function generate_synthetic_dataset(N_obs::Int, σ_true::Real, p_true::AbstractVector)
    # Download forcing data if not present
    download_url = "https://www.dropbox.com/scl/fi/fbxn7antmrchk39li44l6/daily_forcing.mat?rlkey=u1s2lu13f4grqnbxt4ediwlk2&dl=0"
    datadir = mkpath(datadir)
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
    y_true = DDM(ts, precip, Tair, p_true...)
    idx = sort(sample(rng, 1:length(ts), N_obs, replace=false))
    # add noise ϵ ~ N(0,10) to get synthetic observation data
    y_obs = max.(y_true[idx] .+ randn(rng, length(idx)).*σ_true, 0.0)
    return (; ts, Tair, precip, y_obs, idx, y_true)
end

function load_bayelva_air_temp_daily(years=[2019,2020,2021,2022], var=:Tair_200)
    datasets = []
    for yr in years
        df = DataFrame(CSV.File(joinpath(datadir, "BaMet2009_$(yr)_lv1_final.dat"), missingstring="NA"))
        df[!,:UTC] = map(t -> DateTime(t, "YYYY-mm-dd HH:MM:ss"), df[:,:UTC])
        push!(datasets, select(df, :UTC, var))
    end
    df_all = reduce(vcat, datasets)
    df_all_by_day = groupby(transform(df_all, :UTC => ByRow(Date) => :date), :date)
    return Impute.impute(
        combine(df_all_by_day, var => mean, renamecols=false),
        Impute.Interpolate(limit=5),
    )
end

function load_bayelva_swe_daily(years=[2019,2020,2021,2022], var=:SWE_K)
    datasets = []
    for yr in years
        df = DataFrame(CSV.File(joinpath(datadir, "BaSnow2019cs_$(yr)_lv1_final.dat"), missingstring="NA"))
        df[!,:UTC] = map(t -> DateTime(t, "YYYY-mm-dd HH:MM:ss"), df[:,:UTC])
        push!(datasets, select(df, :UTC, var => ByRow(float) => var))
    end
    df_all = reduce(vcat, datasets)
    df_all_by_day = groupby(transform(df_all, :UTC => ByRow(Date) => :date), :date)
    return combine(df_all_by_day, var => maximum, renamecols=false)
end

function load_ny_alesund_pluvio_precip_daily()
    df = DataFrame(CSV.File(joinpath(datadir, "NYA_pluvio_l1_precip_daily_v00_2017-2022.csv")))
    # replace missing values with zero
    return Impute.fill(rename(df, :time => :date), value=0.0)
end
