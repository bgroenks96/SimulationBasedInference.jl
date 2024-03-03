########### NOTICE ############
# Despite the name of the file, these tests are NOT assessing whether or not the code has become intelligent.
# Rather, they assess whether or not the code is entirely stupid :)
###############################

using SimulationBasedInference

using Random
using Test
using Turing

@model function testmodel1()
    x ~ Normal(0,1)
    p ~ Beta(1,1)
    return p*x
end

@testset "Turing priors" begin
    rng = Random.MersenneTwister(1234)
    m = testmodel1()
    prior = TuringPrior(m)
    draw = rand(prior)
    @test isa(draw, ComponentVector)
    @test haskey(draw, :x) && haskey(draw, :p)
    lp = logprob(prior, draw)
    @test lp == Turing.logprior(prior.model, (x=draw.x, p=draw.p))
    draws = sample(rng, prior, 100, progress=false, verbose=false)
    @test isa(draws, Vector{<:AbstractVector})
    # test bijectors
    b = bijector(prior)
    @test length(b(draw)) == length(draw)
end
