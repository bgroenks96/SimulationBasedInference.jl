# Simulator interface

"""
See [`Simulator(simulator)`](@ref)
"""
abstract type Simulator end

"""
Trait for simulators that define a simple forward mapping `f: Φ ↦ X` where Φ is the
parameter space and X is the simulator output.
"""
struct ForwardMap <: Simulator end

"""
Trait for simulators that solve a nonlinear system or optimization problem iteratively.
"""
struct Iterative <: Simulator end

"""
Trait for simulators that simulate a dynamical system over a finite time span.
"""
struct Dynamical <: Simulator end

"""
    Simulator(simulator)
    Simulator(::Type{SimulatorType})

Trait that characterizes the forward map defined by a simulator function or type.
Options are `ForwardMap`, `Iterative`, or `Dynamical`. `ForwardMap` simulators
are defined by a single function `f(p)` that should directly return the output of the
simulation, while `Iterative` and `Dynamical` simulators should be defined using custom types
that implement the following interface:
- `CommonSolve.init(simulator, args...; p::AbstractVecOrMat, tspan, kwargs...)::SimulationType`
- `CommonSolve.step!(::SimulationType, [dt,] args...; kwargs...)`
- `CommonSolve.solve!(::SimulationType)::SolutionType`
- `current_state(::SimulationType)` (optional)
- `isdone(::SimulationType)::Boolean` (optional)

where `SimulationType` and `SolutionType` are types defined by the user or downstream package.
`Dynamical` simulators must additionally define the following:
- `timespan(::SimulationType)::NTuple{2, timeType}`
- `current_time(::SimulationType)::timeType`

The keyword arguments `p` and `tspan` are mandatory and respectively correspond to the choice of
simulation parameters and time span. Simulators characterized by a SciML problem type are broadly
assumed to be `Iterative` or `Dynamical` depending on the problem type.
"""
Simulator(::Type{<:Function}) = ForwardMap()
Simulator(::Type{<:SciMLBase.AbstractSciMLProblem}) = Iterative()
Simulator(::Type{<:SciMLBase.AbstractDEProblem}) = Dynamical()
Simulator(simulator) = Simulator(typeof(simulator))

# Simulation interface

"""
    current_state(simulation)

Return the current state of the simulation. Default implementation returns `nothing`.
"""
current_state(simulation) = nothing
current_state(integrator::SciMLBase.DEIntegrator) = integrator.u

"""
    current_time(simulation)

Return the current time for simulations of dynamical systems. Default implementation returns `nothing`.
    
See also [`current_iteration`](@ref).
"""
current_time(simulation) = nothing
current_time(integrator::SciMLBase.DEIntegrator) = integrator.t

"""
    current_iteration(simulation)::Integer

Return the current iteration number for iterative simulations. Default implementation returns `nothing`.

See also [`current_time`](@ref).
"""
current_iteration(simulation) = nothing

"""
    timespan(simulation)

Return the finite simulation time span for simulations of dynamical systems. Default implementation
returns `nothing`.
"""
timespan(simulation) = nothing
timespan(integrator::SciMLBase.DEIntegrator) = integrator.sol.prob.tspan

"""
    isdone(simulation)

Return `true` if the simulation is finished, `false` otherwise.
"""
isdone(simulation) = false
isdone(integrator::SciMLBase.DEIntegrator) = SciMLBase.done(integrator)
