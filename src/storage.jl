"""
    ForwardMapStorage

Base type for forward map memoization; i.e. saving input/output pairs from
forward model evaluations.
"""
abstract type ForwardMapStorage end

"""
    store!(::ForwardMapStorage, x, y)

Stores the input/output pair x/y in the given forward map storage container.
"""
function store! end

Base.length(storage::ForwardMapStorage) = length(getinputs(storage))
Base.lastindex(storage::ForwardMapStorage) = lastindex(getinputs(storage))
Base.firstindex(storage::ForwardMapStorage) = firstindex(getinputs(storage))

struct SimpleForwardMapStorage <: ForwardMapStorage
    inputs::Vector
    outputs::Vector
end

SimpleForwardMapStorage() = SimpleForwardMapStorage([], [])

function Base.getindex(storage::SimpleForwardMapStorage, i)
    return (storage.inputs[i], storage.outputs[i])
end

getinputs(storage::SimpleForwardMapStorage) = storage.inputs
getinputs(storage::SimpleForwardMapStorage, i) = storage.inputs[i]

getoutputs(storage::SimpleForwardMapStorage) = storage.outputs
getoutputs(storage::SimpleForwardMapStorage, i) = storage.outputs[i]
getoutputs(storage::SimpleForwardMapStorage, name::Symbol) = map(i -> getoutputs(storage, name, i) , 1:length(storage.outputs))
getoutputs(storage::SimpleForwardMapStorage, name::Symbol, i) = reduce(hcat, map(obs -> obs[name], storage.outputs[i].observables))

function store!(storage::SimpleForwardMapStorage, x, y)
    push!(storage.inputs, x)
    push!(storage.outputs, y)
end
