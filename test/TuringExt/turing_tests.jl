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
    p = rand(m_prior)
    @test isa(p, NamedTuple)
    @test haskey(p, :x) && haskey(p, :p)
    chain = sample(rng, m_prior, 100)
    @test isa(chain, MCMCChains.Chains)
end
