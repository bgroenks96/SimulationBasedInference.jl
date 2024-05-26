function SimulationBasedInference.SimulatorForwardProblem(
    gen_fn::GenerativeFunction,
    gen_fn_args::Tuple,
    initial_params::ComponentVector,
    observables::SimulatorObservable...;
    retval_fn=trace -> Gen.get_retval(trace),
)
    f = function(θ)
        params = copyto!(similar(initial_params), θ)
        trace = simulate(gen_fn, (params, gen_fn_args...))
        return retval_fn(trace)
    end
    
    return SimulatorForwardProblem(f, initial_params, observables...)
end

function SimulationBasedInference.likelihood_model(
    inference_prob::SimulatorInferenceProblem{<:GenSimulatorPrior},
    forward_alg;
    solve_kwargs...
)
    # TODO
    error("not yet implemented")
end

function SimulationBasedInference.joint_model(
    inference_prob::SimulatorInferenceProblem{<:GenSimulatorPrior},
    forward_alg;
    solve_kwargs...
)
    # TODO
    error("not yet implemented")   
end
