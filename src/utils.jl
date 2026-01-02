"""
    with_names(xs)

`map`s over `xs` and returns a `NamedTuple` where the keys are `nameof(x)` for each `x`.
"""
with_names(xs) = (; map(x -> nameof(x) => x, xs)...)

"""
    ntreduce(f, xs::AbstractVector{<:NamedTuple})

Applies `reduce` over a vector of named tuples, applying the reducer function `f`
to each element and returning a named tuple of the reduced outputs. All named tuples
in the vector must have the same keys.
"""
function ntreduce(f, xs::AbstractVector{<:NamedTuple})
    foldl(xs) do acc, xᵢ
        (; map(k -> k => f(acc[k], xᵢ[k]), keys(acc))...)
    end
end

"""
    tupleinsert(tuple, i, x)

Insert value `x` into `tuple` at index `i`. This function is fully type-stable and allocation-free.
"""
function tupleinsert(tuple::Tuple{Vararg{Any, N}}, i, x) where {N}
    return map(ntuple(identity, Val{N+1}())) do j
        if j < i
            tuple[j]
        elseif j == i
            x
        else
            tuple[j-1]
        end
    end
end
