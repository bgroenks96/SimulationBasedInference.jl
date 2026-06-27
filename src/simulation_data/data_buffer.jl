"""
    DataBuffer{kind, B<:StorageBackend}

A lightweight, ordered, appendable **view** onto a variable with `name` hosted by the given
[`StorageBackend`](@ref). `kind` is either `:output` or `:scratch` (a transient working buffer).
A `DataBuffer` owns no data of its own; `store!`, `getindex`, `length`, and `empty!` all forward
to the backend.
"""
struct DataBuffer{kind, B<:StorageBackend}
    backend::B
    sim::Int
    name::Symbol
end

DataBuffer{kind}(backend::B, sim::Integer, name::Symbol) where {kind,B<:StorageBackend} =
    DataBuffer{kind,B}(backend, Int(sim), name)

# forwarded primitives for :output
store!(buffer::DataBuffer{:output}, x) = store_output!(buffer.backend, buffer.sim, buffer.name, x)
Base.getindex(buffer::DataBuffer{:output}, i::Integer) = get_output(buffer.backend, buffer.sim, buffer.name, i)
Base.length(buffer::DataBuffer{:output}) = output_length(buffer.backend, buffer.sim, buffer.name)
Base.empty!(buffer::DataBuffer{:output}) = empty_output!(buffer.backend, buffer.sim, buffer.name)

# forwarded primitives for :scratch
store!(buffer::DataBuffer{:scratch}, x) = store_scratch!(buffer.backend, buffer.sim, buffer.name, x)
Base.getindex(buffer::DataBuffer{:scratch}, i::Integer) = get_scratch(buffer.backend, buffer.sim, buffer.name, i)
Base.length(buffer::DataBuffer{:scratch}) = scratch_length(buffer.backend, buffer.sim, buffer.name)
Base.empty!(buffer::DataBuffer{:scratch}) = empty_scratch!(buffer.backend, buffer.sim, buffer.name)

# generic derived methods (kind-agnostic)
Base.push!(buffer::DataBuffer, x) = store!(buffer, x)
Base.firstindex(::DataBuffer) = 1
Base.lastindex(buffer::DataBuffer) = length(buffer)
Base.isempty(buffer::DataBuffer) = length(buffer) == 0
Base.size(buffer::DataBuffer) = (length(buffer),)
Base.getindex(buffer::DataBuffer, r::AbstractVector) = [buffer[i] for i in r]
Base.iterate(buffer::DataBuffer, state::Int=1) = state <= length(buffer) ? (buffer[state], state + 1) : nothing
Base.collect(buffer::DataBuffer) = [buffer[i] for i in 1:length(buffer)]
Base.first(buffer::DataBuffer) = buffer[1]
Base.last(buffer::DataBuffer) = buffer[length(buffer)]
