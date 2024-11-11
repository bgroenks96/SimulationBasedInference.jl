using OrdinaryDiffEq
using Plots

using SimulationBasedInference

using DifferentialEquations
using LinearAlgebra
using StatsPlots

using Random

const rng = Random.MersenneTwister(1234);
# ---------------------------------------------------------------------------- #
#region Model
# ---------------------------------------------------------------------------- #
function lotka_volterra!(du,u,p,t)
    x, y = u
    α, β, γ, δ = p
    du[1] = (α - β*y)x # dx
    du[2] = (δ*x - γ)y # dy
end
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0,1.0]
tspan = (0.0, 10.0)
odeprob = ODEProblem(lotka_volterra!,u0,tspan, p)
tsave = range(0.0, 10.0,101)            #dt = 0.1

# ---------------------------------------------------------------------------- #
#region Data Preparation
# ---------------------------------------------------------------------------- #
ode_solver = Tsit5();
sol = solve(odeprob, ode_solver; saveat=0.1)
odedata = Array(sol) + 0.8 * randn(size(Array(sol)))
u1 = odedata[1,:]
u2 = odedata[2,:]
Nt = length(u1)

### Define observables
y1 = SimulatorObservable(:u1, state -> state.u[1,:], (Nt,))
y2 = SimulatorObservable(:u2, state -> state.u[2,:], (Nt,))

function DAE_problem(ode_func, tspan, solver)
    function DAE_simulation(θ)
        prob = ODEProblem(ode_func,u0,tspan, θ)
        sol = solve(prob, solver, saveat = tsave)
        # return hcat(sol.u...)
        return hcat(sol.u...)
    end
    return DAE_simulation
end

forward_prob = SimulatorForwardProblem(DAE_problem(lotka_volterra!, tspan, ode_solver), p, y1, y2)    


# ---------------------------------------------------------------------------- #
#region Set prior
# ---------------------------------------------------------------------------- #
# parameter prior
model_prior = prior(α=truncated(Normal(1.3, 3), 0, 3), β = truncated(Normal(1.2, 3), 0, 5), γ = truncated(Normal(1.2, 3), 0, 5), δ = truncated(Normal(1.2, 3), 0, 5))
# noise scale prior
prior_y1 = prior(prior_y1 = InverseGamma(2,3));
prior_y2 = prior(prior_y2 = InverseGamma(2,3));

# likelihood of the obsevation/noise model.
lik_y1 = SimulatorLikelihood(IsoNormal, y1, u1, prior_y1, :u1) 
lik_y2 = SimulatorLikelihood(IsoNormal, y2, u2, prior_y2, :u2)

inference_prob = SimulatorInferenceProblem(forward_prob, model_prior, lik_y1, lik_y2)
# inference_prob = SimulatorInferenceProblem(forward_prob, Tsit5(), model_prior, lik_y1, lik_y2);

# ---------------------------------------------------------------------------- #
#region EnIS: ensemble importance sampling
# ---------------------------------------------------------------------------- #
enis_sol = solve(inference_prob, EnIS(), ensemble_size=128, rng=rng);
PosteriorStats.summarize(enis_sol)

prior_ens = get_transformed_ensemble(enis_sol)
prior_ens_mean = mean(prior_ens, dims=2)[:,1]
prior_ens_std = std(prior_ens, dims=2)[:,1]

# u1, x
prior_ens_obs = Array(get_observables(enis_sol).u1);
prior_ens_obs_mean = mean(prior_ens_obs, dims=2)[:,1]
prior_ens_obs_std = std(prior_ens_obs, dims=2)
nothing #hide

importance_weights = get_weights(enis_sol)
nothing #hide

posterior_obs_mean_enis = mean(prior_ens_obs, weights(importance_weights), 2)[:,1]
posterior_obs_std_enis = std(prior_ens_obs, weights(importance_weights), 2)[:,1]

# u2, y
prior_ens_obs2 = Array(get_observables(enis_sol).u2);
prior_ens_obs_mean2 = mean(prior_ens_obs2, dims=2)[:,1]
prior_ens_obs_std2 = std(prior_ens_obs2, dims=2)
nothing #hide

posterior_obs_mean_enis2 = mean(prior_ens_obs2, weights(importance_weights), 2)[:,1]
posterior_obs_std_enis2 = std(prior_ens_obs2, weights(importance_weights), 2)[:,1]

# Plot result
plt_EnIS = plot(tsave, sol[1, :], label="True solution", c=:black, linewidth=2, title="Posterior predictions (EnIS)",layout=(1, 2));
xlabel!("x");
plot!(plt_EnIS, tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.5, linewidth=2);
plot!(plt_EnIS, tsave, posterior_obs_mean_enis, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_enis, alpha=0.7, linewidth=2);
scatter!(plt_EnIS, tsave, u1, label="Noisy observations", c=:orange);

plot!(plt_EnIS, tsave, sol[2, :], label="True solution", c=:black, linewidth=2, title="Posterior predictions (EnIS)", subplot=2);
xlabel!("x");
plot!(plt_EnIS, tsave, prior_ens_obs_mean2, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std2, alpha=0.5, linewidth=2, subplot=2);
plot!(plt_EnIS, tsave, posterior_obs_mean_enis2, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_enis2, alpha=0.7, linewidth=2, subplot=2);
scatter!(plt_EnIS, tsave, u1, label="Noisy observations", c=:orange, subplot=2);
plot!(plt_EnIS, size=(2000,600), ylims=(0,10))

# ---------------------------------------------------------------------------- #
#region EKS Ensemble Kalman Sampler
# ---------------------------------------------------------------------------- #
eks_sol = solve(inference_prob, EKS(), ensemble_size=128, rng=rng, verbose=false)
simdata = SimulationBasedInference.sample_ensemble_predictive(eks_sol)
PosteriorStats.summarize(eks_sol)

posterior_eks = get_transformed_ensemble(eks_sol)
posterior_mean_eks = mean(posterior_eks, dims=2)

posterior_obs_eks1 = get_observables(eks_sol).u1
posterior_obs_mean_eks1 = mean(posterior_obs_eks1, dims=2)[:,1]
posterior_obs_std_eks1 = std(posterior_obs_eks1, dims=2)[:,1]
posterior_obs_eks2 = get_observables(eks_sol).u2
posterior_obs_mean_eks2 = mean(posterior_obs_eks2, dims=2)[:,1]
posterior_obs_std_eks2 = std(posterior_obs_eks2, dims=2)[:,1]

plt_EKS = plot(tsave, sol[1, :], label="True solution", c=:black, linewidth=2, title="Posterior predictions (EKS)",layout=(1,2));
xlabel!("x");
plot!(plt_EKS,tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.5, linewidth=2);
plot!(plt_EKS, tsave, posterior_obs_mean_eks1, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_eks1, alpha=0.7, linewidth=2);
scatter!(plt_EKS, tsave, u1, label="Noisy observations", c=:orange);

plot!(plt_EKS, tsave, sol[2, :], label="True solution", c=:black, linewidth=2, title="Posterior predictions (EKS)", subplot=2);
xlabel!("x");
plot!(plt_EKS, tsave, prior_ens_obs_mean2, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std2, alpha=0.5, linewidth=2, subplot=2);
plot!(plt_EKS, tsave, posterior_obs_mean_eks2, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_eks2, alpha=0.7, linewidth=2, subplot=2);
scatter!(plt_EKS, tsave, u2, label="Noisy observations", c=:orange, subplot=2);
plot!(plt_EKS, size=(2000,600), ylims=(0,10))

# ---------------------------------------------------------------------------- #
#region ESMDA ensemble smoother
# ---------------------------------------------------------------------------- #
esmda_sol = solve(inference_prob, ESMDA(), ensemble_size=128, rng=rng);
nothing #hide

posterior_esmda = get_transformed_ensemble(esmda_sol)
posterior_mean_esmda = mean(posterior_esmda, dims=2)

posterior_obs_esmda = get_observables(esmda_sol).u1
posterior_obs_mean_esmda = mean(posterior_obs_esmda, dims=2)[:,1]
posterior_obs_std_esmda = std(posterior_obs_esmda, dims=2)[:,1]

posterior_obs_esmda2 = get_observables(esmda_sol).u2
posterior_obs_mean_esmda2 = mean(posterior_obs_esmda2, dims=2)[:,1]
posterior_obs_std_esmda2 = std(posterior_obs_esmda2, dims=2)[:,1]

plt_EKS = plot(tsave, sol[1, :], label="True solution", c=:black, linewidth=2, title="Posterior predictions (ESMDA)",layout=(1,2));
plot!(tsave, prior_ens_obs_mean, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std, alpha=0.5, linewidth=2);
xlabel!("x");
plot!(plt_EKS, tsave, posterior_obs_mean_esmda, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_esmda, alpha=0.7, linewidth=2);
scatter!(plt_EKS, tsave, u1, label="Noisy observations", c=:orange);

plot!(plt_EKS, tsave, sol[2, :], label="True solution", c=:black, linewidth=2, title="Posterior predictions (ESMDA)", subplot=2);
xlabel!("x");
plot!(tsave, prior_ens_obs_mean2, label="Prior", c=:gray, linestyle=:dash, ribbon=2*prior_ens_obs_std2, alpha=0.5, linewidth=2, subplot=2);
plot!(plt_EKS, tsave, posterior_obs_mean_esmda2, label="Posterior", c=:blue, linestyle=:dash, ribbon=2*posterior_obs_std_esmda2, alpha=0.7, linewidth=2, subplot=2);
scatter!(plt_EKS, tsave, u2, label="Noisy observations", c=:orange, subplot=2);
plot!(plt_EKS, size=(2000,600), ylims=(0,10))
