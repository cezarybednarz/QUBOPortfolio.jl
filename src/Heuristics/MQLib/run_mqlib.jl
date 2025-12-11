
function run_mqlib_on_instance(heuristic_name::String, instance::QUBOPortfolio.Instance)::Result
    time_taken = @elapsed begin
        model = qubo_to_jump_model(instance.qubo_instance)
        set_optimizer(model, MQLib.Optimizer)
        MQLib.set_heuristic(model, heuristic_name)
        MOI.set(model, MOI.Silent(), true)
        JuMP.optimize!(model)
        variables = JuMP.all_variables(model)
        solution_bits = Int.(JuMP.value.(variables))
    end
    return Result(
        energy=objective_value(model),
        solution=solution_bits,
        time_taken=time_taken,
        used_heuristics=[Heuristic(type=QUBOPortfolio.MQLIB, name=heuristic_name)]
    )
end
