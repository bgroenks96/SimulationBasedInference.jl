using ComponentArrays
using Dates
using Interpolations
using Unitful

"""
    DDM(t, precip, Tair, a, b)

Simple "degree-day" snow melt model with preciptation `precip` and air temperature
`Tair` at time steps `t` as inputs. The parameters `a` and `b` correspond to the
degree-day melt factor and precipitation scaling factor respectively. These parameters
may be passed as vectors which would then correspond to an ensemble of values sampled
from some prior distribution.
"""
DDM(t, precip, Tair, a::Number, b::Number; Tsnow=1.0, Tmelt=0.0) = DDM(t, precip, Tair, [a], [b]; Tsnow, Tmelt)
function DDM(
    t::AbstractVector{DateTime},
    precip::AbstractVector{<:Real},
    Tair::AbstractVector{<:Real},
    a::AbstractVector{<:Number},
    b::AbstractVector{<:Number};
    Tsnow=1.0,
    Tmelt=0.0
)
    # DDM: A simple degree day snowmelt model
    # Coded for a single cell ensemble run, but could
    # also easily be extended to multiple cells stored as ensemble chunks.
    
    Nt=length(t); # Number of time steps.
    Ne=length(a); # Number of ensemble members (can be just 1).
    D_old=0; # Initial condition (no snow).
    D=zeros(eltype(a), Nt, Ne);
    
    b0=zeros(size(b));
    a0=zeros(size(a));
    
    for j=1:Nt
        Pj=precip[j];
        Tj=Tair[j];
        cansnow = Tj <= Tsnow; # Snow is possible.
        # For simplicity rain does not contribute to SWE
        bj = cansnow ? b : b0
        ddj=Tj-Tmelt; # Degree day for this day
        Aj=Pj.*bj; # Accumulation for this day
        aj = ddj > zero(ddj) ? a : a0
        Mj=ddj.*aj;
        D_new=D_old.+Aj.-Mj;
        D_new=max.(D_new,0);
        D[j,:].=D_new;
        D_old=D_new;
    end

    return D
end

"""
    DDM_ode(t, precip, Tair)

Same basic model as `DDM` but defines the problem as a continuous ODE with non-autonomous inputs
`precip` and `Tair`. Returns a `NamedTuple` of the form: `(; dudt, resid, u0, p)` where `dudt`
is the flux (derivative of the state at time `t`), `resid` is a residual function that is nonzero
when the state is outside of the physical boudns (i.e. negative), `u0` is the initial state, and `p`
are the parameters of the system.
"""
function DDM_ode(
    t::AbstractVector{DateTime},
    precip::AbstractVector{<:Real},
    Tair::AbstractVector{<:Real},
)
    t_real = ustrip.(u"s", float.(Dates.datetime2epochms.(t)).*u"ms")
    tspan = extrema(t_real)
    # linear interpolation for Tair
    Tair_interp = interpolate((t_real,), Tair, Gridded(Linear()))
    # constant interpolation for precipitation
    precip_interp = interpolate((t_real,), precip./(24*3600), Gridded(Constant()))
    # set up parameter vector
    p = ComponentVector(
        a = 1.0,
        b = 1.0,
        Tsnow = 1.0,
        Tmelt = 0.0,
    )

    # dudt function defining the ODE
    DDM_dudt(u, p, t::DateTime) = DDM_dudt(u, p, Dates.datetime2epochms(t))
    function DDM_dudt(u, p, t::Number)
        # unpack parameters
        a, b, Tsnow, Tmelt = p
        # get current forcing values
        Tair_t = Tair_interp(t)
        precip_t = precip_interp(t)
        # adjust parameters based on temperature
        b = ifelse(Tair_t <= Tsnow, b, zero(b))
        a = ifelse(Tair_t > Tmelt, a, zero(a))
        # accumulation
        acc = precip_t*b
        # compute snow melt flux
        dd = ifelse(u[1] > zero(eltype(u)), Tair_t - Tmelt, zero(eltype(u)))
        dudt = acc - dd*a./(24*3600.0)
        return zeros(size(u)) .+ dudt
    end

    # residual function which is nonzero when the state is non-negative
    function DDM_resid(u, p)
        return min.(u, zero(eltype(u)))
    end

    return (dudt=DDM_dudt, resid=DDM_resid, u0=[0.0], p=p, tspan=tspan)
end
