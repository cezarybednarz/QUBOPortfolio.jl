include("../src/QUBOPortfolio.jl")

# Test heuristics:
#  - TAMC
#  - MQLib
#  - MetaheuristicsJL (PSO, ABC)
#  - Simulated Bifurcation
# against brute-force optimal solutions on small random instances

using Test
using .QUBOPortfolio.QUBOTools

NUM_TEST_CASES = 20
MAX_TEST_INSTANCE_SIZE = 5 # should be at least 2

# Initialize counters for optimal solutions
optimal_counts = Dict(
    "TAMC" => 0,
    "MQLib" => 0,
    "MetaheuristicsJL" => 0,
    "SimulatedBifurcation" => 0
)

for test_idx in 1:NUM_TEST_CASES
    INSTANCE_SIZE = rand(3:MAX_TEST_INSTANCE_SIZE)
    test_instance = QUBOPortfolio.get_random_instance(INSTANCE_SIZE)

    # Load the heuristics
    brute_force = QUBOPortfolio.Heuristic(type=QUBOPortfolio.BRUTEFORCE, name="Brute Force", hyperparameters=Dict())
    mqlib_heuristic = QUBOPortfolio.load_all_mqlib_heuristics()[1]
    tamc_heuristic = QUBOPortfolio.load_all_tamc_heuristics()[1]
    metaheuristicsjl_heuristic = QUBOPortfolio.load_all_metaheuristicsjl_heuristics()[1]
    simulated_bifurcation_heuristic = QUBOPortfolio.load_all_simulated_bifurcation_heuristics()[1]

    # Run the heuristics on the test instance
    brute_force_result = QUBOPortfolio.run_heuristic(brute_force, test_instance)[1]
    tamc_result = QUBOPortfolio.run_heuristic(tamc_heuristic, test_instance, "/home/cezary/Studia/Magisterka/tamc/target/release/tamc")[1]
    mqlib_result = QUBOPortfolio.run_heuristic(mqlib_heuristic, test_instance)[1]
    metaheuristicsjl_result = QUBOPortfolio.run_heuristic(metaheuristicsjl_heuristic, test_instance)[1]
    simulated_bifurcation_result = QUBOPortfolio.run_heuristic(simulated_bifurcation_heuristic, test_instance)[1]

    @info ">>>>>>>>> Test case #$test_idx <<<<<<<<<"
    @info "Times:"
    @info "  BruteForce: $(round(brute_force_result.time_taken, digits=2))s"
    @info "  TAMC: $(round(tamc_result.time_taken, digits=2))s"
    @info "  MQLib: $(round(mqlib_result.time_taken, digits=2))s"
    @info "  MetaheuristicsJL: $(round(metaheuristicsjl_result.time_taken, digits=2))s"
    @info "  SimulatedBifurcation: $(round(simulated_bifurcation_result.time_taken, digits=2))s"
    QUBOPortfolio.check_solution_bits(brute_force_result, test_instance)
    QUBOPortfolio.check_solution_bits(tamc_result, test_instance)
    QUBOPortfolio.check_solution_bits(mqlib_result, test_instance)
    QUBOPortfolio.check_solution_bits(metaheuristicsjl_result, test_instance)
    QUBOPortfolio.check_solution_bits(simulated_bifurcation_result, test_instance)

    # Check if the heuristic solutions are optimal
    if isapprox(tamc_result.energy, brute_force_result.energy, atol=1e-5)
        optimal_counts["TAMC"] += 1
    end
    if isapprox(mqlib_result.energy, brute_force_result.energy, atol=1e-5)
        optimal_counts["MQLib"] += 1
    end
    if isapprox(metaheuristicsjl_result.energy, brute_force_result.energy, atol=1e-5)
        optimal_counts["MetaheuristicsJL"] += 1
    end
    if isapprox(simulated_bifurcation_result.energy, brute_force_result.energy, atol=1e-5)
        optimal_counts["SimulatedBifurcation"] += 1
    end

    for (heuristic, count) in optimal_counts
        percentage = (count / test_idx) * 100
        @info "$heuristic: $count optimal solutions ($percentage%)"
    end
end

@info ">>>>>>>>> Summary <<<<<<<<<"
for (heuristic, count) in optimal_counts
    percentage = (count / NUM_TEST_CASES) * 100
    @info "$heuristic: $count optimal solutions ($percentage%)"
end
