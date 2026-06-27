using SimulationBasedInference
using JLD2
using Test

@testset "Simulation data storage" begin
    @testset "DataSeries (views over a backend)" begin
        data = SimulationData()
        # a persistent output series
        s = getdata(data, :a)
        @test s isa SimulationBasedInference.DataSeries
        @test length(s) == 0
        store!(s, [1.0, 2.0])
        store!(s, [3.0, 4.0])
        @test length(s) == 2
        @test s[1] == [1.0, 2.0]
        @test s[2] == [3.0, 4.0]
        @test collect(s) == [[1.0, 2.0], [3.0, 4.0]]
        @test first(s) == [1.0, 2.0]
        @test last(s) == [3.0, 4.0]
        @test [x for x in s] == collect(s)
        empty!(s)
        @test length(s) == 0
    end

    @testset "SimulationData" begin
        data = SimulationData(input=zeros(3))
        @test getinputs(data) == zeros(3)
        setinputs!(data, ones(3))
        @test getinputs(data) == ones(3)
        # persistent outputs (per observable name)
        store!(data, :a, [1.0])
        store!(data, :a, [2.0])
        store!(data, :b, [10.0])
        @test getoutput(data, :a) == [[1.0], [2.0]]
        @test getoutput(data, :b) == [[10.0]]
        outs = getoutputs(data)
        @test keys(outs) == (:a, :b)
        @test outs.a == [[1.0], [2.0]]
        # transient scratch buffers are not supported in direct disk backend
        # clear resets all series for the simulation
        empty!(data)
        @test keys(getoutputs(data)) == ()
    end

    @testset "SimulationDataSet (in-memory)" begin
        dataset = SimulationDataSet()
        @test length(dataset) == 0
        # allocate a fresh simulation and populate it
        d1 = allocate!(dataset; iter=1, member=1)
        setinputs!(d1, [1.0, 2.0])
        store!(d1, :y, [0.5])
        @test length(dataset) == 1
        @test getinputs(dataset, 1) == [1.0, 2.0]
        @test getmetadata(dataset, 1)[:iter] == 1
        @test getmetadata(dataset, 1)[:member] == 1
        @test getoutputs(dataset, 1).y == [[0.5]]
        # append an externally-constructed simulation (copied into the backend)
        external = SimulationData(input=[3.0, 4.0])
        store!(external, :y, [1.5])
        store!(dataset, external; iter=1, member=2)
        @test length(dataset) == 2
        @test getinputs(dataset, 2) == [3.0, 4.0]
        @test getmetadata(dataset, 2)[:member] == 2
        # second iteration
        d3 = allocate!(dataset; iter=2, member=1)
        setinputs!(d3, [5.0, 6.0])
        @test iterations(dataset) == 2
        # iterating yields (input, outputs, metadata) triples
        triples = collect(dataset)
        @test length(triples) == 3
        @test triples[1][1] == [1.0, 2.0]
        @test triples[1][3][:iter] == 1
        @test getinputs(dataset)[2] == [3.0, 4.0]
        empty!(dataset)
        @test length(dataset) == 0
    end

    @testset "OnDiskSimulationDataSet (disk)" begin
        mktempdir() do dir
            file = File(format"JLD2", joinpath(dir, "sims.jld2"))
            dataset = OnDiskSimulationDataSet(file)
            @test length(dataset) == 0
            # store two simulations (streamed to disk)
            d1 = SimulationData(input=[1.0, 2.0])
            store!(d1, :y, [0.5])
            store!(d1, :y, [0.6])
            store!(dataset, d1; iter=1, member=1)
            d2 = SimulationData(input=[3.0, 4.0])
            store!(d2, :y, [1.5])
            store!(dataset, d2; iter=1, member=2)
            @test length(dataset) == 2
            # read back simulations
            @test getinputs(dataset, 1) == [1.0, 2.0]
            @test getmetadata(dataset, 1)[:iter] == 1
            @test getoutputs(dataset, 1).y == [[0.5], [0.6]]
            @test getinputs(dataset, 2) == [3.0, 4.0]
            reopened = OnDiskSimulationDataSet(file)
            @test length(reopened) == 2
            @test getinputs(reopened, 1) == [1.0, 2.0]
            @test getoutputs(reopened, 2).y == [[1.5]]
        end
    end
end
