obscov(Σ::AbstractMatrix) = Σ
obscov(σ::AbstractVector) = Diagonal(σ)
obscov(σ::Number) = σ*I
# from SimulatorLikelihood types
obscov(likelihoods::SimulatorLikelihood...) = error("obscov not implemented for the given likelihood types")
# currently only diagonal covariances are supported
function obscov(likelihoods::SimulatorLikelihood{<:Union{IsoNormal,DiagNormal}}...)
    cov_diags = map(likelihoods) do lik
        return diag(cov(lik, first(mean(lik.prior))))
    end
    # concatenate all covariance matrices 
    return Diagonal(reduce(vcat, cov_diags))
end
