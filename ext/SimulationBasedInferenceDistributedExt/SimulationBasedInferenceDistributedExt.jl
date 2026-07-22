module SimulationBasedInferenceDistributedExt

using SimulationBasedInference
using SimulationBasedInference: EnsembleForwardSolver

using Distributed
using SciMLBase

import CommonSolve

function _init(forward_prob, descriptor, solve_args)
    simdata = SBI.from_descriptor(descriptor.type, descriptor)
    solver = init(forward_prob, solve_args.forward_alg, solve_args.args...; simdata=simdata, solve_args.kwargs...)
    return solver
end

function CommonSolve.init(
    forward_probs::Vector{<:SimulatorForwardProblem},
    simdata::Vector{<:SimulationData},
    ensalg::EnsembleDistributed,
    args...;
    storage::SimulationDataSet = SimulationDataSet(),
    attributes = (;),
    kwargs...
)
    descriptors = map(SBI.descriptor, simdata)
    solve_args = repeat([(; args, kwargs)], length(forward_probs))
    solvers = pmap(_init, forward_probs, descriptors, solve_args)
    return EnsembleForwardSolver(ensalg, solvers)
end

function CommonSolve.step!(enssolver::EnsembleForwardSolver{EnsembleDistributed})
    results = pmap(step!, enssolver.solvers)
    return results
end

function CommonSolve.solve!(enssolver::EnsembleForwardSolver{EnsembleDistributed})
    results = pmap(solve!, enssolver.solvers)
    simdata = map(result -> result.simdata, results)
    # copy data from results back into original simulation data (no-op for disk backends)
    for (src, dest) in zip(simdata, enssolver.simdata)
        copy!(dest, src)
    end
    return results
end


end
