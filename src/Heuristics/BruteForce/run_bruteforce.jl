
function run_bruteforce_on_instance(instance::QUBOPortfolio.Instance, num_solutions_to_check=Inf)::QUBOPortfolio.Result
    time_taken = @elapsed begin
        # Get the number of variables in the QUBO instance
        num_vars = QUBOPortfolio.num_qubo_variables(instance)

        # Generate all possible solutions (2^num_vars)
        all_solutions = collect(Iterators.product(Iterators.repeated(0:1, num_vars)...))

        best_energy = Inf
        best_solution = nothing

        for solution in all_solutions
            energy = QUBOPortfolio.calculate_energy_from_solution(instance, collect(solution))
            if energy < best_energy
                best_energy = energy
                best_solution = solution
            end

            # Stop if we have checked the desired number of solutions
            num_solutions_to_check -= 1
            if num_solutions_to_check <= 0
                break
            end
        end
    end

    return QUBOPortfolio.Result(
        energy=best_energy,
        solution=collect(best_solution),
        time_taken=time_taken,
        used_heuristics=[Heuristic(type=QUBOPortfolio.BRUTEFORCE, name="BruteForce")]
    )
end
