@kwdef struct Result
    energy::Float64
    solution::Vector{Int} = Int[]
    time_taken::Float64 = 0.0
    # TODO make this a single Heuristic instead of a vector
    used_heuristics::Vector{Heuristic} = Heuristic[]
end


function calculate_energy_from_solution(instance::QUBOPortfolio.Instance, solution::Vector{Int})::Float64
    calculated_energy = 0.0
    linear_terms = Dict(QUBOTools.linear_terms(instance.qubo_instance))
    quadratic_terms = Dict(QUBOTools.quadratic_terms(instance.qubo_instance))

    # Map variable indices to a contiguous range
    all_vars = Set{Int}()
    for v in keys(linear_terms)
        push!(all_vars, v)
    end
    for (v1, v2) in keys(quadratic_terms)
        push!(all_vars, v1)
        push!(all_vars, v2)
    end
    variable_map = Dict(v => i for (i, v) in enumerate(sort(collect(all_vars))))

    # Calculate energy
    for (var, coeff) in linear_terms
        calculated_energy += coeff * solution[variable_map[var]]
    end
    for ((var1, var2), coeff) in quadratic_terms
        calculated_energy += coeff * solution[variable_map[var1]] * solution[variable_map[var2]]
    end
    return calculated_energy
end
