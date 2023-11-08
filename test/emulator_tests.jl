using SimulationBasedInference.Emulators
using Distributions
using MLJ, MLJGaussianProcesses, DecisionTree
using MLJGaussianProcesses.Optim
using KernelFunctions
using LinearAlgebra
using Statistics
using Test

@testset "DecorrelatedTarget" begin
    X = randn(1000,5)
    Y = reduce(
        hcat,
        [vcat(X[i,1].*X[i,2], X[i,2].*X[i,3], X[i,3].*X[i,4], X[i,4].*X[i,5]) for i in 1:1000]
    ) |> transpose
    data = EmulatorData(X, Y)
    decor = DecorrelatedTarget(data)
    @test decor.mean == mean(Y, dims=1)[1,:]
    Z = transform_target(decor, Y)
    @test isapprox(cov(Z), zeros(size(Z,2),size(Z,2)) + I)
    Y_r = inverse_transform_target(decor, Z)
    @test isapprox(Y_r, Y)
end

@testset "GP emulator" begin
    function gpr_kernel(θ)
        θ.σf²*(Matern32Kernel() ∘ ScaleTransform((θ.ℓ)^2 / 2) + WhiteKernel())
    end
    X = randn(1000,5)
    Y = reduce(
        hcat,
        [vcat(X[i,1].*X[i,2], X[i,2].*X[i,3], X[i,3].*X[i,4], X[i,4].*X[i,5]) .+ 0.01randn(4) for i in 1:1000]
    ) |> transpose
    data = EmulatorData(X, Y)
    decor = DecorrelatedTarget(data)
    emulator = StackedMLEmulator(GPR(k=gpr_kernel, optimizer=NelderMead()), data, decor)
    @test eltype(emulator.models) <: Machine{<:GPR}
    fitted_emulator = MLJ.fit!(emulator, verbosity=0)
    @test all(map(m -> m.state == 1, fitted_emulator.models))
    preds = MLJ.predict(emulator, X)
    @test all(map(d -> isa(d, MvNormal), preds))
end
