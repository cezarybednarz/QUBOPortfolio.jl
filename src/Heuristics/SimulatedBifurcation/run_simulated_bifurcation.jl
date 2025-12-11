
function run_simulated_bifurcation_on_instance(instance::QUBOPortfolio.Instance, hyperparameters::Dict{String, Any},
    heuristic_name::String)::QUBOPortfolio.Result
    time_taken = @elapsed begin
        linear_terms = Dict(QUBOTools.linear_terms(instance.qubo_instance))
        quadratic_terms = Dict(QUBOTools.quadratic_terms(instance.qubo_instance))
        edges = Vector{Any}()
        for (i, v) in linear_terms
            push!(edges, [i - 1, i - 1, v])
        end
        for ((i, j), v) in quadratic_terms
            push!(edges, [i - 1, j - 1, v])
        end

        python_executable = joinpath(@__DIR__, "PythonScript", "venv", "bin", "python")
        python_script = joinpath(@__DIR__, "PythonScript", "sb.py")

        n = QUBOPortfolio.num_qubo_variables(instance)
        payload = Dict("n" => n, "edges" => edges, "hyperparameters" => hyperparameters)
        payload_json = JSON.json(payload)

        buf = IOBuffer(payload_json)

        result_json = readchomp(pipeline(buf, `$python_executable $python_script`))

        parsed_result = JSON.parse(result_json)

        vector = parsed_result["result_bits"]
        energy = parsed_result["energy"]
    end

    return QUBOPortfolio.Result(
        energy=energy,
        solution=vector,
        time_taken=time_taken,
        used_heuristics=[Heuristic(type=QUBOPortfolio.SIMULATED_BIFURCATION, name=heuristic_name, hyperparameters=hyperparameters)]
    )
end
