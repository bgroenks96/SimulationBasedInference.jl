"""
    DataBuffer{kind, H} where {H <:StorageHandle}

A lightweight **view** onto a variable with `name` with data accessed via the given [`StorageHandle`](@ref).
The handle owns the simulation index (`handle.index`). `kind` is either `:output` (persistent output) or `:scratch`
(a transient working buffer). A `DataBuffer` owns no data of its own; `store!`, `getindex`, `length`, and `empty!` all
forward to the I/O handle.
"""
struct DataBuffer{kind, H<:StorageHandle}
    handle::H
    name::Symbol
end

DataBuffer{kind}(handle::StorageHandle, name::Symbol) where {kind} = DataBuffer{kind, typeof(handle)}(handle, name)

# forwarded primitives for :output with handle (fast path)
store!(buffer::DataBuffer{:output}, x) = store_output!(buffer.handle, buffer.name, x)
Base.getindex(buffer::DataBuffer{:output}, i::Integer) = get_output(buffer.handle, buffer.name, i)
Base.length(buffer::DataBuffer{:output}) = output_length(buffer.handle, buffer.name)
Base.empty!(buffer::DataBuffer{:output}) = empty_output!(buffer.handle, buffer.name)

# forwarded primitives for :scratch with handle (fast path)
store!(buffer::DataBuffer{:scratch}, x) = store_scratch!(buffer.handle, buffer.name, x)
Base.getindex(buffer::DataBuffer{:scratch}, i::Integer) = get_scratch(buffer.handle, buffer.name, i)
Base.length(buffer::DataBuffer{:scratch}) = scratch_length(buffer.handle, buffer.name)
Base.empty!(buffer::DataBuffer{:scratch}) = empty_scratch!(buffer.handle, buffer.name)

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
Base.close(buffer::DataBuffer) = close(buffer.handle)
