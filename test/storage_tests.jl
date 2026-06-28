using SimulationBasedInference
using SimulationBasedInference: InMemoryStorage, InMemoryStorageHandle,
    allocate!, getmetadata, setmetadata!,
    get_output_buffer, output_names, store_output!, get_outputs, output_length, has_output,
    store_scratch!, get_scratch, scratch_length, has_scratch, scratch_names,
    num_simulations, isopen
using FileIO, JLD2
using Test

@testset "Simulation data storage" begin
    @testset "DataBuffer (views over a backend)" begin
        data = SimulationData()
        # a persistent output series
        buffer = get_output_buffer(data, :a)
        @test buffer isa SimulationBasedInference.DataBuffer
        @test length(buffer) == 0
        store!(buffer, [1.0, 2.0])
        store!(buffer, [3.0, 4.0])
        @test length(buffer) == 2
        @test buffer[1] == [1.0, 2.0]
        @test buffer[2] == [3.0, 4.0]
        @test collect(buffer) == [[1.0, 2.0], [3.0, 4.0]]
        @test first(buffer) == [1.0, 2.0]
        @test last(buffer) == [3.0, 4.0]
        @test [x for x in buffer] == collect(buffer)
        empty!(buffer)
        @test length(buffer) == 0
    end

    @testset "SimulationData" begin
        data = SimulationData(inputs=zeros(3))
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
        external = SimulationData(inputs=[3.0, 4.0])
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
            dataset = OnDiskSimulationDataSet(format"JLD2", dir)
            @test length(dataset) == 0
            # store two simulations
            d1 = allocate!(dataset, [1.0, 2.0], iter=1)
            store!(d1, :y, [0.5])
            store!(d1, :y, [0.6])
            d2 = allocate!(dataset, [3.0, 4.0], iter=1)
            setmetadata!(d2, extra="foo") # test setmetdata!
            store!(d2, :y, [1.5])
            @test length(dataset) == 2
            # read back simulations
            @test getinputs(dataset, 1) == [1.0, 2.0]
            @test getmetadata(dataset, 1)[:iter] == 1
            @test getoutputs(dataset, 1).y == [[0.5], [0.6]]
            @test getinputs(dataset, 2) == [3.0, 4.0]
            @test getmetadata(dataset, 2)[:iter] == 1
            @test getmetadata(dataset, 2)[:extra] == "foo"
            reopened = OnDiskSimulationDataSet(format"JLD2", dir)
            @test length(reopened) == 2
            @test getinputs(reopened, 1) == [1.0, 2.0]
            @test getoutputs(reopened, 2).y == [[1.5]]
        end
    end

    # ========================================================================
    # Handle-based API Tests (In-Memory Backend)
    # ========================================================================
    @testset "StorageHandle - InMemory" begin
        backend = InMemoryStorage()
        
        # Test allocate first simulation and open handle for it
        i1 = SimulationBasedInference.allocate!(backend, [1.0, 2.0]; iter=1, member=1)
        handle = open(backend, i1)
        @test handle isa InMemoryStorageHandle
        @test handle.backend === backend
        @test handle.sim_id == i1
        
        # Test that we can access data through the handle
        @test SimulationBasedInference.getinputs(handle, i1) == [1.0, 2.0]
        @test SimulationBasedInference.getmetadata(handle, i1)[:iter] == 1
        
        # Test output operations with handle (fast path - no auto-open/close)
        store_output!(handle, i1, :y, [0.5])
        store_output!(handle, i1, :y, [0.6])
        @test get_outputs(handle, i1, :y) == [[0.5], [0.6]]
        @test output_length(handle, i1, :y) == 2
        @test has_output(handle, i1, :y)
        
        # Test scratch operations with handle (handle-specific storage)
        store_scratch!(handle, i1, :temp, "data1")
        store_scratch!(handle, i1, :temp, "data2")
        @test get_scratch(handle, i1, :temp, 1) == "data1"
        @test get_scratch(handle, i1, :temp, 2) == "data2"
        @test scratch_length(handle, i1, :temp) == 2
        @test has_scratch(handle, i1, :temp)
        @test :temp in scratch_names(handle, i1)
        
        # Test allocate second simulation and open handle for it
        i2 = SimulationBasedInference.allocate!(backend, [3.0, 4.0]; iter=1, member=2)
        @test i2 == 2
        @test num_simulations(backend) == 2
        
        # Verify scratch is isolated between simulations (stored per-simulation in handle)
        handle2 = open(backend, i2)
        store_scratch!(handle2, i2, :temp2, "data3")
        @test has_scratch(handle2, i2, :temp2)  # Scratch exists for this sim
        
        # Close handles - should clear scratch but preserve backend data
        close(handle)
        close(handle2)
        
        # After close, scratch should be cleared
        handle3 = open(backend, i1)
        @test has_scratch(handle3, i1, :temp) == false  # Cleared on close
        
        # But backend data should persist
        @test SimulationBasedInference.getinputs(handle3, i1) == [1.0, 2.0]
        @test get_outputs(handle3, i1, :y) == [[0.5], [0.6]]
        
        close(handle3)
    end

    # ========================================================================
    # Handle-based API Tests (JLD2 Disk-Backed Backend)
    # ========================================================================
    @testset "StorageHandle - JLD2" begin
        mktempdir() do dir
            dataset = OnDiskSimulationDataSet(format"JLD2", dir)
            
            # Allocate first simulation and open handle for it
            i1 = SimulationBasedInference.allocate!(dataset.backend, [1.0, 2.0]; iter=1)
            handle = open(dataset.backend, i1)
            
            # Test that we got a valid handle (type check via isa with module prefix)
            @test handle !== nothing
            @test isopen(handle)
            
            # Test output operations with handle (writes to disk)
            store_output!(handle, i1, :y, [0.5])
            store_output!(handle, i1, :y, [0.6])
            
            # Test read operations with same handle (fast path - no re-open)
            @test SimulationBasedInference.getinputs(handle, i1) == [1.0, 2.0]
            @test get_outputs(handle, i1, :y) == [[0.5], [0.6]]
            
            # Test scratch operations (in-memory for ParallelJLD2StorageHandle)
            store_scratch!(handle, i1, :scratch_data, "temp_value")
            @test get_scratch(handle, i1, :scratch_data, 1) == "temp_value"
            
            # Close handle - should persist to disk and clear scratch
            close(handle)
            
            # Reopen and verify data persisted, scratch cleared
            handle2 = open(dataset.backend, i1)
            @test SimulationBasedInference.getinputs(handle2, i1) == [1.0, 2.0]
            @test get_outputs(handle2, i1, :y) == [[0.5], [0.6]]
            @test has_scratch(handle2, i1, :scratch_data) == false  # Cleared on close
            
            # Allocate second simulation and add more data
            i2 = SimulationBasedInference.allocate!(dataset.backend, [5.0, 6.0]; iter=2)
            handle3 = open(dataset.backend, i2)
            store_output!(handle3, i2, :z, [1.0])
            
            close(handle2)
            close(handle3)
            
            # Verify final state by reopening fresh
            handle4 = open(dataset.backend, 1)
            @test num_simulations(dataset.backend) == 2
            @test SimulationBasedInference.getinputs(handle4, 1) == [1.0, 2.0]
            @test get_outputs(handle4, 1, :y) == [[0.5], [0.6]]
            close(handle4)
            
            handle5 = open(dataset.backend, 2)
            @test SimulationBasedInference.getinputs(handle5, 2) == [5.0, 6.0]
            @test get_outputs(handle5, 2, :z) == [[1.0]]
            close(handle5)
        end
    end

    # ========================================================================
    # Handle Resource Safety Tests (finalize destructor)
    # ========================================================================
    @testset "Handle Resource Safety - finalize" begin
        backend = InMemoryStorage()
        
        # Allocate simulation and create handle
        i = SimulationBasedInference.allocate!(backend, [1.0])
        handle = open(backend, i)
        store_scratch!(handle, i, :temp, "should_be_cleared")
        
        # Force garbage collection to trigger finalize
        handle_ref = handle  # Keep reference for later check
        handle = nothing
        GC.gc()
        
        # Create new handle and verify scratch was cleared by finalize
        handle2 = open(backend, i)
        @test has_scratch(handle2, i, :temp) == false
        close(handle2)
    end

    # ========================================================================
    # Handle Scratch Isolation Tests  
    # ========================================================================
    @testset "Handle Scratch Isolation" begin      
        backend = InMemoryStorage()
        
        # Allocate simulations and create handles for each
        i1 = SimulationBasedInference.allocate!(backend, [1.0])
        handle1 = open(backend, i1)
        store_scratch!(handle1, i1, :shared, "from_handle1")
        
        i2 = SimulationBasedInference.allocate!(backend, [2.0])
        handle2 = open(backend, i2)
        store_scratch!(handle2, i2, :shared, "from_handle2")
        
        # Each handle sees only its own scratch
        @test get_scratch(handle1, i1, :shared, 1) == "from_handle1"
        @test get_scratch(handle2, i2, :shared, 1) == "from_handle2"
        
        # Close handle1 - its scratch should be cleared
        close(handle1)
        
        # Reopen and verify handle1's scratch is gone but handle2's persists  
        handle3 = open(backend, i1)
        @test has_scratch(handle3, i1, :shared) == false  # Cleared when handle1 closed
        
        # Note: handle2 still exists, so its scratch should persist
        @test get_scratch(handle2, i2, :shared, 1) == "from_handle2"
        
        close(handle2)
        close(handle3)
    end
end  # End of "Simulation data storage" testset
