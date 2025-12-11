
function qubo_to_jump_model(qubo_model::QUBOTools.Model{V,T,U}) where {V,T,U}
    jump_model = JuMP.Model()

    linear_terms = Dict(collect(QUBOTools.linear_terms(qubo_model)))
    quadratic_terms = Dict(collect(QUBOTools.quadratic_terms(qubo_model)))

    variables = collect(QUBOTools.variables(qubo_model))

    # somehow QUBOTools.variables(qubo_model) is not working, had to fill in the variables manually
    for v in keys(linear_terms)
        push!(variables, v)
    end
    for (v1, v2) in keys(quadratic_terms)
        push!(variables, v1)
        push!(variables, v2)
    end

    unique!(variables)

    var_map = Dict{V,VariableRef}()
    for v in variables
        var_map[v] = @variable(jump_model, base_name="x_$v", binary=true)
    end

    set_silent(jump_model)

    # todo there is a bug somewhere (run at the end of loader.jl and see )
    @objective(jump_model, Min, sum(quadratic_terms[(v1,v2)] * var_map[v1] * var_map[v2] for (v1,v2) in keys(quadratic_terms)))
    @objective(jump_model, Min, objective_function(jump_model) + sum(linear_terms[v] * var_map[v] for v in keys(linear_terms)))

    return jump_model
end

function qubo_to_max_cut(qubo_model::QUBOTools.Model{V,T,U}) where {V,T,U}
    num_variables = length(QUBOTools.variables(qubo_model))
    g_empty = SimpleGraph{Int}(num_variables + 1) # +1 for auxiliary node
    g = SimpleWeightedGraph(g_empty)

    quadratic_terms = Dict(collect(QUBOTools.quadratic_terms(qubo_model)))
    linear_terms = Dict(collect(QUBOTools.linear_terms(qubo_model)))
    variables = collect(QUBOTools.variables(qubo_model))

    var_index = Dict{V, Int}()
    for (i, v) in enumerate(variables)
        var_index[v] = i
    end

    for ((v_i, v_j), w) in quadratic_terms
        i = var_index[v_i]
        j = var_index[v_j]
        add_edge!(g, i, j, -w)
    end

    for (v_i, c_i) in linear_terms
        i = var_index[v_i]
        add_edge!(g, 0, i, c_i)
    end

    return g
end

function max_cut_to_qubo(max_cut_instance::SimpleWeightedGraph)
    linear_terms = Dict{Int,Float64}()
    quadratic_terms = Dict{Tuple{Int,Int},Float64}()

    for e in edges(max_cut_instance)
        v1 = src(e)
        v2 = dst(e)
        w = weight(e)
        if !haskey(linear_terms, v1)
            linear_terms[v1] = 0.0
        end
        if !haskey(linear_terms, v2)
            linear_terms[v2] = 0.0
        end
        if !haskey(quadratic_terms, (v1,v2))
            quadratic_terms[(v1,v2)] = 0.0
        end
        linear_terms[v1] -= w
        linear_terms[v2] -= w
        quadratic_terms[(v1,v2)] += 2.0 * w
    end

    return QUBOTools.Model(linear_terms, quadratic_terms)
end
