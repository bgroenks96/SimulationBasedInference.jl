using SimulationBasedInference.Emulators
using Distributions
using KernelFunctions
using LinearAlgebra
using Random
using Statistics
using Test

# autodiff
using ForwardDiff, Zygote

@testset "Decorrelated" begin
    rng = MersenneTwister(1234)
    X = randn(rng, 5,1000)
    Y = reduce(
        hcat,
        [vcat(X[1,i].*X[2,i], X[2,i].*X[3,i], X[3,i].*X[4,i], X[4,i].*X[5,i]) for i in 1:1000]
    )
    data = EmulatorData(X, Y)
    decor = Decorrelated(data.Y)
    @test decor.mean == mean(Y, dims=2)[:,1]
    Z = Emulators.apply(decor, Y)
    @test isapprox(cov(Z, dims=2), zeros(size(Z,1),size(Z,1)) + I)
    Y_r = Emulators.apply_inverse(decor, Z)
    @test isapprox(Y_r, Y)
end

@testset "GP emulator" begin
    rng = MersenneTwister(1234)
    X = randn(rng, 5, 128)
    Y = X[1:end-1,:].*X[2:end,:] .+ 0.1.*randn(rng, 4, 128)
    data = EmulatorData(X, Y)
    output_transform = Decorrelated(data.Y, frac=1.0)
    emulator = StackedEmulator(data, GPRegressor(); output_transform)
    fitted_emulator = Emulators.fit!(emulator, verbosity=0)
    preds = Emulators.predict(emulator, X)
    @test all(map(d -> isa(d, MvNormal), preds))

    # test GP forward vs. reverse grad
    gpr = GPRegressor()
    Emulators.fit!(gpr, X, Y[1,:], verbosity=0)
    Emulators.predict(gpr, X[:,1:1])
    zgrad = Zygote.gradient(x -> mean(Emulators.predict(gpr, reshape(x,:,1))[1]), X[:,1])[1]
    fgrad = ForwardDiff.gradient(x -> mean(Emulators.predict(gpr, reshape(x,:,1))[1]), X[:,1])
    @test isapprox(zgrad, fgrad)
end
