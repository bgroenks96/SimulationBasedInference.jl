module SimulationBasedInferenceDistributedExt

using SimulationBasedInference
using SimulationBasedInference: EnsembleForwardSolver

using Distributed
using SciMLBase

import CommonSolve

function _init(forward_prob, simdata, solve_args)
    solver = init(forward_prob, solve_args.forward_alg, solve_args.args...; simdata=simdata, solve_args.kwargs...)
    return solver
end

function CommonSolve.init(
    forward_probs::Vector{<:SimulatorForwardProblem},
    forward_alg,
    ensalg::EnsembleDistributed,
    args...;
    storage::SimulationDataSet = SimulationDataSet(),
    attrs = (;),
    kwargs...
)
    simdata = [allocate!(storage, prob.p; attrs) for prob in forward_probs]
    solve_args = repeat([(; forward_alg, args, kwargs)], length(forward_probs))
    solvers = pmap(_init, forward_probs, simdata, solve_args)
    return EnsembleForwardSolver(ensalg, solvers)
end

function CommonSolve.step!(enssolver::EnsembleForwardSolver{EnsembleDistributed})
    results = pmap(step!, enssolver.solvers)
    return results
end

function CommonSolve.solve!(enssolver::EnsembleForwardSolver{EnsembleDistributed})
    results = pmap(solve!, enssolver.solvers)
    return results
end


end
