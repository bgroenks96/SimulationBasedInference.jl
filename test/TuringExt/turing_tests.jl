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
    m_prior = TuringPrior(m)
    draw = rand(m_prior)
    @test isa(draw, ComponentVector)
    @test haskey(draw, :x) && haskey(draw, :p)
    lp = logprob(m_prior, draw)
    @test lp == Turing.logprior(m_prior.model, (x=draw.x, p=draw.p))
    chain = sample(rng, m_prior, 100, progress=false, verbose=false)
    @test isa(chain, MCMCChains.Chains)
end
