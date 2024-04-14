using SimulationBasedInference
using Test

@testset "Simulation data storage" begin
    @testset "SimulationArrayStorage" begin
        storage = SimulationArrayStorage()
        # store a single entry
        x = zeros(10)
        y = ones(2)
        store!(storage, x, y)
        @test length(storage) == 1
        @test getinputs(storage, 1) == x
        @test getoutputs(storage, 1) == y
        SBI.clear!(storage)
        @test length(storage) == 0
        # store batch
        X = [x x.+1]
        store!(storage, X, [y,y.+1])
        @test length(storage) == 2
        @test getinputs(storage, 1) == x
        @test getoutputs(storage, 1) == y
        @test getinputs(storage, 2) == x.+1
        @test getoutputs(storage, 2) == y.+1
        @test isa(getinputs(storage), Vector)
        @test isa(getoutputs(storage), Vector)
        SBI.clear!(storage)
        # store with metadata
        X = [x x.+1]
        store!(storage, X, [y,y.+1], iter=1)
        @test length(storage) == 2
        metadata = getmetadata(storage)
        @test metadata[1] == metadata[2] == (iter=1,)
        SBI.clear!(storage)
        # store with non-array output
        X = [x x.+1]
        store!(storage, X, [(y=y,),(y=y.+1,)])
        @test length(storage) == 2
        @test getoutputs(storage, 1) == (y=y,)
        @test getoutputs(storage, 2) == (y=y.+1,)
    end
end
