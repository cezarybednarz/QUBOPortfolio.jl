
function check_solution_bits(result::QUBOPortfolio.Result, instance::QUBOPortfolio.Instance)
    # Check if the solution bits are valid
    for bit in result.solution
        if !(bit in [0, 1])
            @error "Invalid bit in solution: $bit. Solution must be binary (0 or 1)."
        end
    end

    calculated_energy = QUBOPortfolio.calculate_energy_from_solution(instance, result.solution)
    @info "Reported energy: $(result.energy), Calculated energy: $calculated_energy"
    if !isapprox(result.energy, calculated_energy, atol=1e-3)
        @error "Energy mismatch! Reported: $(result.energy), Calculated: $calculated_energy"
    end
end
