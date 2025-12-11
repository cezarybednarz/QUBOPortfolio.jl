include("../../src/QUBOPortfolio.jl")

function test_heuristic()

    function check_solution_bits(result::QUBOPortfolio.Result, instance::QUBOPortfolio.Instance)
        # Check if the solution bits are valid
        for bit in result.solution
            @test bit in [0, 1]
        end

        # Check if the length of solution bits matches the number of variables in the instance
        @test length(result.solution) == length(QUBOTools.variables(instance.qubo_instance))

        # Check if the sultion bits correspond to the energy of the instance
        calculated_energy = QUBOTools.energy(instance.qubo_instance, result.solution)
        @test result.energy == calculated_energy
    end

    @testset "Heuristic execution" begin
        # Load a test instance
        test_instance = QUBOPortfolio.get_instance_from_path("test_example1.qh")

        # Load the Brute Force algorithm
        brute_force_algorithm = QUBOPortfolio.Heuristic(QUBOPortfolio.BRUTEFORCE, "Brute Force", Dict())
        # Load the heuristics
        mqlib_heuristic = QUBOPortfolio.load_all_mqlib_heuristics()[1]
        tamc_heuristic = QUBOPortfolio.load_all_tamc_heuristics()[5]
        metaheuristicsjl_heuristic = QUBOPortfolio.load_all_metaheuristicsjl_heuristics()[1]
        simulated_bifurcation_heuristic = QUBOPortfolio.load_all_simulated_bifurcation_heuristics()[1]

        # Run the heuristics on the test instance
        brute_force_result = QUBOPortfolio.run_heuristic(brute_force_algorithm, test_instance)
        tamc_result = QUBOPortfolio.run_heuristic(tamc_heuristic, test_instance)
        mqlib_result = QUBOPortfolio.run_heuristic(mqlib_heuristic, test_instance)
        metaheuristicsjl_heuristic = QUBOPortfolio.run_heuristic(metaheuristicsjl_heuristic, test_instance)
        simulated_bifurcation_result = QUBOPortfolio.run_heuristic(simulated_bifurcation_heuristic, test_instance)

        # Result comparison
        # TODO debug the TAMC heuristic execution
        # @test isapprox(brute_force_result.energy, tamc_result.energy; atol=1e-5)
        @test isapprox(brute_force_result.energy, mqlib_result.energy; atol=1e-5)
        @test isapprox(brute_force_result.energy, metaheuristicsjl_heuristic.energy; atol=1e-5)
        @test isapprox(brute_force_result.energy, simulated_bifurcation_result.energy; atol=1e-5)

        # Check solution bits
        check_solution_bits(mqlib_result, test_instance)
        check_solution_bits(metaheuristicsjl_heuristic, test_instance)
        check_solution_bits(tamc_result, test_instance)
        check_solution_bits(simulated_bifurcation_result, test_instance)
    end

    @testset "run_heuristics_on_dataset" begin
        # Mock Heuristics
        h1 = QUBOPortfolio.Heuristic(type=QUBOPortfolio.BRUTEFORCE, name="h1")
        h2 = QUBOPortfolio.Heuristic(type=QUBOPortfolio.BRUTEFORCE, name="h2")
        heuristics = [h1, h2]

        # Mock Instances
        instance1 = QUBOPortfolio.get_instance_from_path("test_example1.qh")
        instance2 = QUBOPortfolio.get_instance_from_path("test_example2.qh")
        instances = [instance1, instance2]

        results, times = QUBOPortfolio.run_heuristics_on_dataset(heuristics, instances; repeat=1, verbosity=false)
        @test length(results) == 4
        @info results
        @test results[("h1", instance1.name)] == -29.0
        @test results[("h2", instance1.name)] == -29.0
        @test results[("h1", instance2.name)] == -1.0
        @test results[("h2", instance2.name)] == -1.0
    end
end

test_heuristic()
