using Test

include("test_problems.jl")

@testset "Observables" begin
    include("observables_tests.jl")
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

# @testset "Emulators" begin
#     include("emulators/emulator_tests.jl")
# end

@testset "Turing integration" begin
    include("TuringExt/turing_tests.jl")
end

@testset "Regression tests" begin
    include("issues/regression_tests.jl")
end
