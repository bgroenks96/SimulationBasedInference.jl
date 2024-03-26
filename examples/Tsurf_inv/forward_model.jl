using CryoGrid
using CryoGridML
using Interpolations

function piecewise_para(
    name::Symbol,
    T₀,
    t_knots::AbstractVector{DateTime},
    tspan;
    interp=Constant()
)
    Δts = ustrip.(abs.(diff(convert_t.(t_knots))))
    # rescale bin widths to sum to unity
    Δts = Δts ./ sum(Δts)
    piecewise_para = PiecewiseLinear(
        name,
        T₀,
        map(Δt -> (Δt, T₀), Δts)...;
        # map(Δt -> (Param(Δt), Param(ustrip(T₀), units=u"°C")), Δts)...;
        tstart=convert_t(tspan[1]),
        tstop=convert_t(tspan[end]),
        interp,
    )
    return piecewise_para
end


function periodicbc(level, amplitude, t0)
    period = ustrip(u"s", 1.0u"yr")
    phase_shift = -2π*t0/period - π
    function f(t)
        s = sin(2π/period*(t - t0) - phase_shift)
        T_ub = amplitude*s + level
        return T_ub
    end
    return TemperatureBC(TimeVaryingForcing(f, :Tsurf, initial_value=0.0u"°C"))
end

function maketile(
    soilprofile,
    upperbc,
    lowerbc,
    tspan,
    discretization,
    initT;
    heatop=Heat.Diffusion1D(:H),
    freezecurve=FreeWater(),
)
    heat = HeatBalance(heatop; freezecurve)
    soil_layers = map(soilprofile) do soilpara
        Ground(soilpara; heat)
    end
    strat = Stratigraphy(
        first(keys(soilprofile)) => Top(upperbc),
        soil_layers,
        1000.0u"m" => Bottom(lowerbc)
    );
    tile = Tile(strat, discretization, initT)
    p = CryoGrid.parameters(tile)
    u0, _ = initialcondition!(tile, tspan, p);
    return (; tile, u0)
end

function set_up_Tsurf_forward_problem(
    soilprofile::SoilProfile,
    discretization::DiscretizationStrategy,
    initT::VarInitializer{:T},
    tspan::NTuple{2,DateTime},
    t_knots::AbstractVector{DateTime},
    obs_depths::AbstractVector,
    observables...;
    saveat=tspan[1]:Month(1):tspan[end],
    savevars=(:T,),
    obs_period=Year(1),
    interp=Linear(),
    T₀=Param(-10.0, units=u"°C"),
    Qgeo=Param(0.053, units=u"W/m^2"),
    A₀=Param(20.0, units=u"K"),
    freezecurve=FreeWater(),
)
    Tsurf_para = piecewise_para(:Tsurf, T₀, t_knots, tspan; interp)
    amp_para = piecewise_para(:amp, A₀, t_knots, tspan; interp)
    upperbc = periodicbc(Tsurf_para, amp_para, convert_t(tspan[1]))
    lowerbc = GeothermalHeatFlux(Qgeo)
    tile, u0 = maketile(soilprofile, upperbc, lowerbc, tspan, discretization, initT; freezecurve)
    prob = CryoGridProblem(tile, u0, tspan, saveat=convert_t.(saveat), savevars=savevars)
    Ts_observable = TemperatureProfileObservable(:Ts, obs_depths, tspan, Month(1), samplerate=Day(1))
    Ts_pred_observable = TemperatureProfileObservable(:Ts_pred, obs_depths, (tspan[end]-obs_period, tspan[end]), obs_period, samplerate=Day(1))
    T_ub_observable = SimulatorObservable(:T_ub, integrator -> getstate(integrator).top.T_ub, tspan[1], tspan[1]+Day(1):Day(1):tspan[end], (1,), samplerate=Day(1))
    alt_observable = ActiveLayerThicknessObservable(:alt, extrema(saveat), samplerate=Day(1))
    forward_prob = SimulatorForwardProblem(prob, Ts_observable, Ts_pred_observable, alt_observable, T_ub_observable, observables...)
    return forward_prob
end