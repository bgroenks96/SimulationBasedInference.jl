using SimulationBasedInference

import CairoMakie as Makie
import Random

const rng = Random.MersenneTwister(1234);

# For this example, we set up a trivial forward model which
# uses the parameters themselves as the output. This corresponds
# to a simple statistical model where the unknowns are the means
# of the data-generating process.
n_obs = 20
p_true = ComponentVector(x₁=-0.42, x₂=0.35)
p0 = p_true
f(x) = x*ones(1,n_obs)
observable = SimulatorObservable(:x, x -> x.u, (2,n_obs))
forward_prob = SimulatorForwardProblem(f, p0, observable)

# Set univariate normal distributions as priors
model_prior = SBI.prior(x₁=Normal(0,1), x₂=Normal(0,1))

# Generate noisy observations
noisy_obs = reshape(p_true, :, 1) .+ 0.1*randn(rng, 2, n_obs)

# Assign a Gaussian likelihood (with the true variance) for the obsevation/noise model
noise_scale_prior = LogNormal(log(0.2), 1.0)
lik = IsotropicGaussianLikelihood(observable, noisy_obs, noise_scale_prior);
inference_prob = SimulatorInferenceProblem(forward_prob, model_prior, lik);
eks_sol = solve(inference_prob, EKS(), ensemble_size=1024, rng=rng);

ens_1 = get_transformed_ensemble(eks_sol, 1)
ens_2 = get_transformed_ensemble(eks_sol, 2)
ens_3 = get_transformed_ensemble(eks_sol, 3)
ens_4 = get_transformed_ensemble(eks_sol, 4)

Makie.with_theme(fontsize=18) do
    fig = Makie.Figure(size=(900,700))
    grid = fig[1:2,1:2] = Makie.GridLayout(default_colgap=30)
    d1 = MvNormal(mean(ens_1, dims=2)[:,1], cov(ens_1, dims=2))
    d2 = MvNormal(mean(ens_2, dims=2)[:,1], cov(ens_2, dims=2))
    d3 = MvNormal(mean(ens_3, dims=2)[:,1], cov(ens_3, dims=2))
    d4 = MvNormal(mean(ens_4, dims=2)[:,1], cov(ens_4, dims=2))
    # Iteration 1
    ax1 = Makie.Axis(grid[1,1], title="Iteration 1", xlabel="x₁", ylabel="x₂", xticks=-1.0:0.5:1.0, yticks=-1.0:0.5:1.0)
    Makie.contourf!(ax1, -1.0:0.01:1.0, -1.0:0.01:1.0, (x₁,x₂) -> pdf(d1, [x₁,x₂]), colormap=:Blues, overdraw=false)
    so = Makie.scatter!(ax1, noisy_obs[1,:], noisy_obs[2,:], color=(:black,0.5), marker=:x, markersize=10.0)
    se = Makie.scatter!(ax1, ens_1[1,:], ens_1[2,:], color=(:green,0.4))
    st = Makie.scatter!(ax1, p_true..., color=:red, marker=:star4, markersize=15.0)
    Makie.xlims!(ax1, -1, 1)
    Makie.ylims!(ax1, -1, 1)
    # Iteration 2
    ax2 = Makie.Axis(grid[1,2], title="Iteration 2", xlabel="x₁", ylabel="x₂")
    Makie.contourf!(ax2, -1.0:0.01:1.0, -1.0:0.01:1.0, (x₁,x₂) -> pdf(d2, [x₁,x₂]), colormap=:Blues, overdraw=false)
    Makie.scatter!(ax2, noisy_obs[1,:], noisy_obs[2,:], color=(:black,0.5), marker=:x, markersize=12.0)
    Makie.scatter!(ax2, ens_2[1,:], ens_2[2,:], color=(:green,0.4))
    Makie.scatter!(ax2, p_true..., color=:red, marker=:star4, markersize=15.0)
    Makie.xlims!(ax2, -1, 1)
    Makie.ylims!(ax2, -1, 1)
    # Iteration 3
    ax3 = Makie.Axis(grid[2,1], title="Iteration 3", xlabel="x₁", ylabel="x₂")
    Makie.contourf!(ax3, -1.0:0.01:1.0, -1.0:0.01:1.0, (x₁,x₂) -> pdf(d3, [x₁,x₂]), colormap=:Blues, overdraw=false)
    Makie.scatter!(ax3, noisy_obs[1,:], noisy_obs[2,:], color=(:black,0.5), marker=:x, markersize=12.0)
    Makie.scatter!(ax3, ens_3[1,:], ens_3[2,:], color=(:green,0.1))
    Makie.scatter!(ax3, p_true..., color=:red, marker=:star4, markersize=15.0)
    Makie.xlims!(ax3, -1, 1)
    Makie.ylims!(ax3, -1, 1)
    # Iteration 4
    ax4 = Makie.Axis(grid[2,2], title="Iteration 4", xlabel="x₁", ylabel="x₂")
    Makie.contourf!(ax4, -1.0:0.01:1.0, -1.0:0.01:1.0, (x₁,x₂) -> pdf(d4, [x₁,x₂]), colormap=:Blues, overdraw=false)
    Makie.scatter!(ax4, noisy_obs[1,:], noisy_obs[2,:], color=(:black,0.5), marker=:x, markersize=12.0)
    Makie.scatter!(ax4, ens_4[1,:], ens_4[2,:], color=(:green,0.05))
    Makie.scatter!(ax4, p_true..., color=:red, marker=:star4, markersize=15.0)
    Makie.xlims!(ax4, -1, 1)
    Makie.ylims!(ax4, -1, 1)
    # link axes
    Makie.linkaxes!(ax1, ax2, ax3, ax4)
    # hide axis decorations
    Makie.hidexdecorations!(ax1, label=true, ticklabels=true, ticks=false)
    Makie.hidexdecorations!(ax2, label=true, ticklabels=true, ticks=false)
    Makie.hideydecorations!(ax2, label=true, ticklabels=true, ticks=false)
    Makie.hideydecorations!(ax4, label=true, ticklabels=true, ticks=false)
    # draw legend
    Makie.axislegend(ax4, [st,so,se], ["True mean", "Noisy obs.", "Ensemble"], position=:rb)
    Makie.save("examples/gaussian2D/plots/eks_gaussian.pdf", fig)
    fig
end