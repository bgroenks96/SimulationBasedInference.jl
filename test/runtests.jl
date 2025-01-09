using Test

include("testcases.jl")

@testset "Observables" begin
    include("observable_tests.jl")
end

@testset "Priors" begin
    include("prior_tests.jl")
end

@testset "Problem interface" begin
    include("problem_tests.jl")
end

@testset "Ensembles" begin
    include("ensembles/runtests.jl")
end

@testset "Emulators" begin
    include("emulator_tests.jl")
end

@testset "Turing integration" begin
    include("TuringExt/turing_tests.jl")
end

@testset "Regression tests" begin
    include("issues/regression_tests.jl")
end
