module SimulationBasedInferenceDistributedExt

using SimulationBasedInference
using SimulationBasedInference: EnsembleForwardSolver, SimulatorForwardSolution, InMemoryStorage,
    descriptor, from_descriptor

using Distributed
using SciMLBase

import CommonSolve

"""
    RemoteSimulationData

Serializable reference to simulation `index` in a shared (e.g. on-disk) storage backend,
reopened on the worker from the backend's descriptor.
"""
struct RemoteSimulationData{descType<:NamedTuple}
    descriptor::descType
    index::Int
end

# In-memory storage is process-local, so each worker receives a single-member copy of its
# simulation rather than the (potentially large) backend shared by the whole ensemble.
function _to_remote(data::SimulationData{<:InMemoryStorage})
    backend = typeof(data.backend)(1)
    _copy_member!(backend, 1, data.backend, data.index)
    return SimulationData(backend, 1)
end
# All other backends are assumed to be shared between processes and are reopened by descriptor.
_to_remote(data::SimulationData) = RemoteSimulationData(descriptor(data.backend), data.index)

_from_remote(data::SimulationData) = data
_from_remote(data::RemoteSimulationData) = SimulationData(from_descriptor(data.descriptor.type, data.descriptor), data.index)

# Copy the results of a remote solve back into the caller's simulation data; a no-op for shared backends.
_copy_back!(dest::SimulationData{<:InMemoryStorage}, src::SimulationData{<:InMemoryStorage}) = _copy_member!(dest.backend, dest.index, src.backend, src.index)
_copy_back!(dest::SimulationData, src::SimulationData) = nothing

function _copy_member!(dest::InMemoryStorage, i::Integer, src::InMemoryStorage, j::Integer)
    dest.inputs[i] = src.inputs[j]
    dest.metadata[i] = src.metadata[j]
    dest.outputs[i] = src.outputs[j]
    return dest
end

# Worker-side functions. Each solver lives on its worker in the store of a `Future`;
# fetching the `Future` on its owner returns the stored solver itself, so state persists
# across calls without round trips of the simulator state.

function _init_member(forward_prob, data, args...; kwargs...)
    return init(forward_prob, args...; simdata=_from_remote(data), kwargs...)
end

_step_member!(solver_ref::Future) = step!(fetch(solver_ref))

function _solve_member!(solver_ref::Future)
    sol = solve!(fetch(solver_ref))
    # release any open storage handle so that the solution can be serialized
    close(sol.simdata)
    return sol
end

# Run `f(i, pid)` for each ensemble member, sequentially per worker and concurrently across workers.
function _foreach_member(f, pids::Vector{Int})
    @sync for pid in unique(pids)
        @async for i in findall(==(pid), pids)
            f(i, pid)
        end
    end
end

function CommonSolve.init(
    forward_probs::Vector{<:SimulatorForwardProblem},
    simdata::Vector{<:SimulationData},
    ensalg::EnsembleDistributed,
    args...;
    kwargs...
)
    # assign ensemble members to workers round-robin; each solver then stays on its worker
    pool = workers()
    pids = [pool[mod1(i, length(pool))] for i in eachindex(forward_probs)]
    solver_refs = Vector{Future}(undef, length(forward_probs))
    _foreach_member(pids) do i, pid
        solver_refs[i] = remotecall_wait(_init_member, pid, forward_probs[i], _to_remote(simdata[i]), args...; kwargs...)
    end
    return EnsembleForwardSolver(ensalg, simdata, solver_refs)
end

function CommonSolve.step!(enssolver::EnsembleForwardSolver{EnsembleDistributed})
    results = Vector{Any}(undef, length(enssolver.solvers))
    _foreach_member(map(ref -> ref.where, enssolver.solvers)) do i, pid
        results[i] = remotecall_fetch(_step_member!, pid, enssolver.solvers[i])
    end
    return results
end

function CommonSolve.solve!(enssolver::EnsembleForwardSolver{EnsembleDistributed})
    sols = Vector{Any}(undef, length(enssolver.solvers))
    _foreach_member(map(ref -> ref.where, enssolver.solvers)) do i, pid
        sol = remotecall_fetch(_solve_member!, pid, enssolver.solvers[i])
        dest = enssolver.simdata[i]
        _copy_back!(dest, sol.simdata)
        sols[i] = SimulatorForwardSolution(sol.prob, sol.sol, dest)
    end
    return collect(sols)
end

end
