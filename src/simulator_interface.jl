# Simulator interface

abstract type SimulatorKind end

struct FunctionMap <: SimulatorKind end

struct IterativeSolve <: SimulatorKind end

struct DynamicalSolve{timeType} <: SimulatorKind end

"""
    SimulatorKind(simulator)
    SimulatorKind(::Type{SimulatorType})

Trait that identifies the type of forward map defind by a simulator.
Options are `FunctionMap` and `IterativeSolve`. `FunctionMap` simulators
should be defined by a simple function `f(ϕ)` or `f(ϕ, X)` that directly
return the output of the simulation, while `IterativeSolve` and `DynamicalSolve`
simulators should implement the `CommonSolve` interface: `init(::Simulator)::Simulation`,
`step!(::Simulation)`, and `solve!(::Simulation)`, where the `Simulator` and
`Simulation` types are defined by the user or another package. Simulators
characterized by a SciML problem type are automatically assumed to be `IterativeSolve`s.
"""
SimulatorKind(::Type{<:Function}) = FunctionMap()
SimulatorKind(::Type{<:AbstractSciMLProblem}) = IterativeSolve()
SimulatorKind(::Type{<:AbstractDEProblem}) = DynamicalSolve{Float64}()
SimulatorKind(simulator) = SimulatorKind(typeof(simulator))

"""
    current_state(simulation)

Return the current state of the simulation. The type of the returned state
may depend on the implementation.
"""
current_state(simulation) = simulation
current_state(integrator::SciMLBase.DEIntegrator) = integrator.u

"""
    current_iteration(simulation)

Return the current time for simulations of dynamical systems. See also
[`current_iteration`](@ref). Default implementation returns `nothing`.
"""
current_time(simulation) = nothing

"""
    current_iteration(simulation)

Return the current iteration of the simulation. See also [`current_time`](@ref).
Default implementation returns `nothing`.
"""
current_iteration(simulation) = nothing
