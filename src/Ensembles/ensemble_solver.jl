mutable struct EnsembleSolver{probType,algType,ensalgType,stateType,kwargTypes}
    prob::probType
    alg::algType
    ensalg::ensalgType
    state::stateType
    prob_func::Function # problem generator
    output_func::Function # output function
    pred_func::Function # prediction function
    itercallback::Function # iteration callback
    verbose::Bool
    retcode::ReturnCode.T
    solve_kwargs::kwargTypes
end
