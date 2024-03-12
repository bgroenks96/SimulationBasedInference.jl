module SimulationBasedInferenceMLJGPExt

using SimulationBasedInference
using SimulationBasedInference.Emulators

using KernelFunctions
using MLJModels
using MLJGaussianProcesses: GPR, default_linear_mean, positive
using Optim

const PCA = @load PCA pkg=MultivariateStats verbosity=0

function rbf_kernel(θ)
    θ.σf²*(SqExponentialKernel() ∘ ScaleTransform((θ.ℓ)^2 / 2))
end

function initial_kernel_params(::typeof(rbf_kernel))
    θ_init = (σf²=positive(1.0), ℓ=positive(1.0))
    return θ_init
end

function rq_kernel(θ)
    θ.σf²*(RationalQuadraticKernel(α=θ.α) ∘ ScaleTransform((θ.ℓ)^2 / 2))
end

function initial_kernel_params(::typeof(rq_kernel))
    θ_init = (σf²=positive(1.0), ℓ=positive(1.0), α=positive(2.0))
    return θ_init
end

function SBI.stacked_emulator(
    ::Type{GPR},
    data::EmulatorData;
    target_transform=CenteredTarget(data),
    kernel=rbf_kernel,
    θ_init=initial_kernel_params(kernel),
    σ²=1e-6,
    optimizer=NelderMead(),
    pca_variance_ratio=pca_variance_ratio,
)
    # Gaussian process regressor
    emulator_algorithm = GPR(;μ=default_linear_mean, k=kernel, θ_init, σ², optimizer)
    # PCA transform; helps to ensure positive definiteness of the Gram matrix;
    # here we use a variance ratio of 1.0 because we do not need to reduce the dimensionality.
    pca = PCA(variance_ratio=pca_variance_ratio)
    # define emulator model pipeline
    emulator_model = Standardizer() |> pca |> emulator_algorithm
    # set up emulator
    emulator = StackedMLEmulator(emulator_model, data, target_transform)
    return emulator
end

end
