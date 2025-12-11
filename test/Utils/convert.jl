include("../../src/QUBOPortfolio.jl")

using Test
using Graphs
using SimpleWeightedGraphs
using QUBOLib
using JuMP
using GLPK
using MQLib

import QUBO

QUBOTools = QUBO.QUBOTools

function test_convert()
    @testset "Conversion tests" begin
        @testset "QUBO to JuMP model conversion" begin
            linear_terms = Dict{Int, Float64}(1 => 2.0, 2 => 2.0)
            quadratic_terms = Dict{Tuple{Int, Int}, Float64}((1, 2) => -100.0)  # Add x*y term
            qubo_model = QUBOTools.Model{Int, Float64, Int}(linear_terms, quadratic_terms)

            # convert QUBO model to JuMP model
            jump_model = QUBOPortfolio.qubo_to_jump_model(qubo_model)

            # run optimization
            set_optimizer(jump_model, MQLib.Optimizer)
            MQLib.set_heuristic(jump_model, "BURER2002")
            MOI.set(jump_model, MOI.Silent(), true)
            optimize!(jump_model)

            # check the results
            @test objective_value(jump_model) ≈ -96.0 atol=1e-6
            vars_out = all_variables(jump_model)
            for v in vars_out
                @test value(v) ≈ 1.0 atol=1e-6
            end
            empty!(jump_model)
        end

        @testset "MAX-CUT to QUBO conversion" begin
            # MAX-CUT instance
            g = SimpleWeightedGraph(3)
            add_edge!(g, 1, 2, 1.0)
            add_edge!(g, 2, 3, 1.0)
            add_edge!(g, 1, 3, 1.0)

            # convert MAX-CUT to QUBO model
            qubo_model = QUBOPortfolio.max_cut_to_qubo(g)

            # check
            quadratic = collect(QUBOTools.quadratic_terms(qubo_model))
            linear = collect(QUBOTools.linear_terms(qubo_model))
            @test [quadratic[i][2] for i in 1:length(quadratic)] == [2.0, 2.0, 2.0]
            @test [linear[i][2] for i in 1:length(linear)] == [-2.0, -2.0, -2.0]
        end

        @testset "QUBO to MAX-CUT conversion" begin
            linear_terms = Dict{Int, Float64}(1 => 2.0, 2 => 2.0, 3 => 2.0)
            quadratic_terms = Dict{Tuple{Int, Int}, Float64}((1, 2) => -2.0, (2, 3) => -2.0, (3, 1) => -2.0)
            qubo_model = QUBOTools.Model{Int, Float64, Int}(linear_terms, quadratic_terms)
            g = QUBOPortfolio.qubo_to_max_cut(qubo_model)

            @test Graphs.nv(g) == 4
            edges = collect(Graphs.edges(g))
            @test [(src(e), dst(e)) for e in edges] == [(1, 2), (1, 3), (2, 3)]
        end
    end
end

test_convert()
