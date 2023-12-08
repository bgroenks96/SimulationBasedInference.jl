using Test

@testset "Observables" begin
    include("observable_tests.jl")
end

@testset "Priors" begin
    include("prior_tests.jl")
end

@testset "Problems" begin
    include("problem_tests.jl")
end

@testset "Ensembles" begin
    include("Ensembles/runtests.jl")
end

@testset "Emulators" begin
    include("emulator_tests.jl")
end

@testset "Turing integration" begin
    include("TuringExt/turing_tests.jl")
end
