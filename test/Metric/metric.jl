include("../../src/QUBOPortfolio.jl")

using Graphs
using SimpleWeightedGraphs

function test_metric()
    @testset "Metric summary stats calculation" begin
        @test QUBOPortfolio.get_summary([0.0, 0.0]) ==
            QUBOPortfolio.Summary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)

        summary_1_2 = QUBOPortfolio.get_summary([1.0, 2.0])
        @test summary_1_2.min == 1.0
        @test summary_1_2.max == 2.0
        @test summary_1_2.mean == 1.5
        @test isapprox(summary_1_2.stdev, 0.707, atol = 0.01)
        @test isapprox(summary_1_2.log_kurtosis, 0.693, atol = 0.01)
        @test isapprox(summary_1_2.log_abs_skew, 0.0, atol = 0.01)
        @test isapprox(summary_1_2.skew_positive, 1.0, atol = 0.01)
        @test summary_1_2.constant == 0.0
    end

    @testset "Graph metrics calculation" begin
        g = SimpleWeightedGraph(4)
        add_edge!(g, 1, 2, 1.0)
        add_edge!(g, 2, 3, 1.0)
        add_edge!(g, 3, 1, 1.0)
        add_edge!(g, 3, 4, -3.0)

        @test QUBOPortfolio.get_percent_pos(g) == 0.75
        @test QUBOPortfolio.get_chromatic(g) == 0.75
        @test QUBOPortfolio.get_disconnected(g) == 0.0

        @test QUBOPortfolio.get_core(g) == [0.5, 0.5, 0.5, 0.25]
        @test isapprox(QUBOPortfolio.get_avg_neighbour(g), [0.66,0.66,1,0.33], atol = 0.01)
        @test isapprox(QUBOPortfolio.get_clust(g)[1], 0.0, atol = 1.0)
        @test isapprox(QUBOPortfolio.get_avg_deg_conn(g), [2.5, 1.66, 3.0], atol = 0.01)
        @test QUBOPortfolio.get_mis(g) == 0.5

        spectral_metrics = QUBOPortfolio.get_spectral_metrics(g)
        @test isapprox(spectral_metrics[1], 0.40, atol = 0.01)
        @test isapprox(spectral_metrics[2], 0.12, atol = 0.01)
        @test isapprox(spectral_metrics[3], 0.27, atol = 0.01)
    end

    @testset "Classifying instances" begin
        instance_1 = QUBOPortfolio.get_instance_from_path("test_example1.qh")
        instance_2 = QUBOPortfolio.get_instance_from_path("test_example2.qh")

        classified_instance_1 = QUBOPortfolio.classify(instance_1)
        classified_instance_2 = QUBOPortfolio.classify(instance_2)
        @test classified_instance_1[5] == QUBOPortfolio.Metric("disconnected", 1.0)
        @test classified_instance_2[5] == QUBOPortfolio.Metric("disconnected", 1.0)

        classified_instances = QUBOPortfolio.classify([instance_1, instance_2])
        @test classified_instances["test_example1.qh"][5] == QUBOPortfolio.Metric("disconnected", 1.0)

        classified_df = QUBOPortfolio.classify_to_df([instance_1, instance_2])
        @test classified_df[1, "disconnected"] == 1.0
    end

end

test_metric()
