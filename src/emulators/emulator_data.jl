"""
    EmulatorData

Generic container for emulator training data matrices `X` and `Y`.
`X` should have shape `N x m` where `N` is the number of samples
and `m` is the number of covariates. `Y` should have dimensions
`N x d` where `d` is the number of ouptut covariates.
"""
struct EmulatorData
    X::AbstractMatrix
    Y::AbstractMatrix
    function EmulatorData(X::AbstractMatrix, Y::AbstractMatrix)
        @assert size(X,1) == size(Y,1) "X and Y must have the same number of rows; got $(size(X,2)) != $(size(Y,2))"
        return new(X,Y)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", data::EmulatorData)
    println(io, "Emulator dataset with $(size(data.X,1)) samples and input/output dimensions: $(size(data.X,2)) → $(size(data.Y,2))")
    println(io, "X $(summarystats(data.X))")
    println(io, "Y $(summarystats(data.Y))")
end
