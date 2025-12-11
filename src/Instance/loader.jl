include("../Utils/convert.jl")

function get_instance_from_path(file_name::AbstractString)::QUBOPortfolio.Instance
    # get file from 'data' folder
    path = joinpath(@__DIR__, "../../data", file_name)
    model = QUBOTools.read_model(path)
    return QUBOPortfolio.Instance(
        name=file_name,
        qubo_instance=model,
        max_cut_instance=qubo_to_max_cut(model)
    )
end

function get_random_instance(N = 5)::QUBOPortfolio.Instance
    random_qubo_instance = QUBOLib.Synthesis.generate(QUBOLib.Synthesis.SherringtonKirkpatrick(N))
    QUBOPortfolio.Instance(
        name="random_instance_$(N)",
        qubo_instance=random_qubo_instance,
        max_cut_instance=qubo_to_max_cut(random_qubo_instance)
    )
end

function get_collection_names_from_qubolib()::Vector{String}
    collection_names = String[]

    QUBOLib.access() do index
        collections_df = try
            DBInterface.execute(QUBOLib.database(index), "SELECT collection FROM Collections;") |> DataFrame
        catch
            @warn "Error fetching collections"
            return collection_names
        end

        for collection in collect(String, collections_df[!, :collection])
            instance_count_df = try
                DBInterface.execute(QUBOLib.database(index), "SELECT COUNT(*) as count FROM Instances WHERE collection = ?;", (collection,)) |> DataFrame
            catch
                @warn "Error fetching instance count for collection $collection"
                continue
            end

            instance_count = instance_count_df[1, :count]
            println("Collection: $collection, Size: $instance_count")

            push!(collection_names, collection)
        end
    end

    collection_names
end

function get_dataset_from_qubolib(max_per_collection::Int)::Vector{QUBOPortfolio.AbstractInstance}
    instances = Vector{QUBOPortfolio.AbstractInstance}()

    QUBOLib.access() do index
        collections_df = try
            DBInterface.execute(QUBOLib.database(index), "SELECT collection FROM Collections;") |> DataFrame
        catch
            @warn "Error fetching collections"
            return instances
        end

        for collection in collect(String, collections_df[!, :collection])
            @info "Loading instances from collection: $collection"
            instance_ids_df = try
                DBInterface.execute(QUBOLib.database(index), "SELECT instance FROM Instances WHERE collection = ?;", (collection,)) |> DataFrame
            catch
                @warn "Error fetching instances for collection $collection"
                continue
            end

            if nrow(instance_ids_df) > max_per_collection
                instance_ids_df = first(instance_ids_df, max_per_collection)
            end

            instance_ids = collect(Int, instance_ids_df[!, :instance])
            loaded_instances = [QUBOLib.load_instance(index, id) for id in instance_ids]

            instance_names_df = try
                DBInterface.execute(QUBOLib.database(index), "SELECT name FROM Instances WHERE collection = ? ORDER BY instance ASC;", (collection,)) |> DataFrame
            catch
                @warn "Error fetching instance names for collection $collection"
                instance_names = String[]
            end

            instance_names = collect(String, instance_names_df[!, :name])

            for (i, loaded_instance) in enumerate(loaded_instances)
                instance_name = i <= length(instance_names) ? instance_names[i] : "instance_$(collection)_$(i)" # Custom instance names
                !haskey(loaded_instance.metadata, "name") && (loaded_instance.metadata["name"] = instance_name)

                instance = QUBOPortfolio.Instance(
                    name=instance_name,
                    qubo_instance=loaded_instance,
                    max_cut_instance=qubo_to_max_cut(loaded_instance)
                )
                push!(instances, instance)
            end
        end
    end

    instances
end

function load_zipped_max_cut_instance(full_path_name::AbstractString)::QUBOPortfolio.Instance
    # unzip the folder and read the file
    zip_folder = ZipFile.Reader(full_path_name)
    file = zip_folder.files[1]

    content = ""
    try
        content = read(file, String)
    catch e
        println("Error reading file $file: $e")
        throw(e)
    finally
        close(zip_folder)
    end

    lines = split(content, '\n')

    graph_edges = []
    graph_size = 0

    for line in lines
        if !isempty(line)
            parts = split(line, [' '])
            if parts[1] == "#"
                continue
            end
            if length(parts) == 2
                graph_size = parse(Int, parts[1])
                continue
            end
            v1 = parse(Int, parts[1])
            v2 = parse(Int, parts[2])
            w = parse(Float64, parts[3])
            push!(graph_edges, (v1, v2, w))
        end
    end

    g = SimpleWeightedGraph(graph_size)

    for (v1, v2, w) in graph_edges
        add_edge!(g, v1, v2, w)
    end

    return QUBOPortfolio.Instance(
        name=basename(full_path_name),
        qubo_instance=max_cut_to_qubo(g),
        max_cut_instance=g
    )
end

function load_mqlib_instances(full_path_name::AbstractString, num_instances::Int, min_file_size::Int, max_file_size::Int = 30000000)::Vector{QUBOPortfolio.AbstractInstance}
    all_files = readdir(full_path_name)

    instances = Vector{QUBOPortfolio.AbstractInstance}()
    for file in all_files
        path = joinpath(full_path_name, file)
        file_size = filesize(path)
        if file_size >= max_file_size || file_size <= min_file_size
            continue
        end
        # just set the path, the instance will be loaded when needed
        instance = QUBOPortfolio.InstanceWithPath(instance=nothing, path_to_load_from=path)
        push!(instances, instance)
        if length(instances) == num_instances
            break
        end
    end

    if length(instances) !== num_instances
        @warn "Loaded $(length(instances)) instances from $full_path_name, expected $num_instances"
    end

    return instances
end
