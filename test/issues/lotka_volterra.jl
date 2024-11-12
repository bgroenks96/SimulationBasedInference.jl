using OrdinaryDiffEq
using SimulationBasedInference
using Random

const rng = Random.MersenneTwister(1234);

# ---------------------------------------------------------------------------- #
#region Model
# ---------------------------------------------------------------------------- #
function lotka_volterra!(du,u,p,t)
    x, y = u
    α = p[1]
    β, γ, δ = 1.0, 3.0, 1.0
    du[1] = (α - β*y)x # dx =
    du[2] = (δ*x - γ)y # dy =
end
p = [1.5]
u0 = [1.0,1.0]
tspan = (0.0, 10.0)
odeprob = ODEProblem(lotka_volterra!, u0, tspan, p)
tsave = range(0.0, 10.0, 101)            #dt = 0.1

# ---------------------------------------------------------------------------- #
#region Data Preparation
# ---------------------------------------------------------------------------- #
# two measurements 
ode_solver = Tsit5();
sol = solve(odeprob, ode_solver; saveat=0.1)
odedata = Array(sol) + 0.8 * randn(size(Array(sol)))
u1 = odedata[1,:]
u2 = odedata[2,:]
Nt = length(u1)

### Define observables
# observable = ODEObservable(:y, odeprob, tsave, samplerate=0.01)
y1 = SimulatorObservable(:u1, state -> state.u[1,:], tspan[1], tsave, (1,), samplerate=minimum(diff(tsave)))
y2 = SimulatorObservable(:u2, state -> state.u[2,:], tspan[1], tsave, (1,), samplerate=minimum(diff(tsave)))

forward_prob = SimulatorForwardProblem(odeprob, y1, y2)
sol = solve(forward_prob, Tsit5())

# ---------------------------------------------------------------------------- #
#region 2. Set prior
# ---------------------------------------------------------------------------- #
model_prior = prior(α=Beta(2,2));

prior_y1 = prior(prior_y1 = InverseGamma(2, 3));
prior_y2 = prior(prior_y2 = InverseGamma(2, 3));

lik_y1 = SimulatorLikelihood(IsoNormal, y1, u1, prior_y1, :u1)
lik_y2 = SimulatorLikelihood(IsoNormal, y2, u2, prior_y2, :u2)

# inference_prob = SimulatorInferenceProblem(forward_prob, model_prior, lik_y1, lik_y2)       
inference_prob = SimulatorInferenceProblem(forward_prob, Tsit5(), model_prior, lik_y1, lik_y2);

# ---------------------------------------------------------------------------- #
#region 3. EnIS
# ---------------------------------------------------------------------------- #
enis_sol = solve(inference_prob, EnIS(), ensemble_size=128, rng=rng);
