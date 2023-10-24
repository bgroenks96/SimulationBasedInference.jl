struct EmulatorData
    X::AbstractMatrix
    Y::AbstractMatrix
end

struct NoTransform <: EmulatorDataTransform end

struct Decorrelated <: EmulatorDataTransform
    mean::AbstractVector
    cov::AbstractMatrix
    truncated_svd::SVD
end

function Decorrelated(data::EmulatorData; sampledim=2, frac=0.99)
    X, Y = data.X, data.Y
    Y_c = mean(Y, dims=sampledim)
    Y_centered = Y .- Y_c
    C_Y = cov(Y, dims=sampledim)
    C_svd = svd(C_Y)
    rank = findfirst(>=(frac), cumsum(C_svd.S ./ sum(C_svd.S)))
    Vt_r = C_svd.Vt[1:rank,:]
    S_r = C_svd.S[1:rank]
    truncated_svd = SVD(C_svd.U, S_r, Vt_r)
    return Decorrelated(dropdims(Y_c, dims=sampledim), C_Y, truncated_svd)
end

function transform_target(d::Decorrelated, Y)
    Y_centered = Y .- d.mean
    S_r = d.truncated_svd.S
    Yt = inv(Diagonal(sqrt.(S_r)))*d.truncated_svd.Vt*Y_centered
    return Yt
end

function inverse_transform_target(d::Decorrelated, Z)
    S_r = d.truncated_svd.S
    Y_c = d.truncated_svd.V*Diagonal(sqrt.(S_r))*Z
    Y = Y_c .+ d.mean
    return Y
end
