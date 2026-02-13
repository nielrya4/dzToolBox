module SourceAnalysisHelpers
using XLSX
using NamedArrays
using MatrixTensorFactor
using SedimentSourceAnalysis
using JSON
using Random
using DataFrames

function clean_inf(data)
    if isa(data, Array)
        return [clean_inf(x) for x in data]
    elseif isa(data, Dict)
        return Dict(k => clean_inf(v) for (k, v) in data)
    elseif data == -Inf
        return "null"  # Or another placeholder value
    else
        return data
    end
end

function transform(input_file_path::AbstractString, transformed_file_path::AbstractString)
    df = DataFrame()
    XLSX.openxlsx(input_file_path, enable_cache=false) do f
        sheet = f[1]
        df = DataFrame(XLSX.gettable(sheet))
    end
    sinks = unique(df[!, "SINK ID"])
    measurements = names(df)[3:end]

    counts = Dict{String,Int}()
    for value in df[!, "SINK ID"]
        counts[value] = get(counts, value, 0) + 1
    end
    highest_sink_size = maximum(values(counts))

    dataframes = Vector{DataFrame}()
    sheet_names = Vector{String}()

    for measurement in measurements
        current_measurement_data = DataFrame()
        for sink in sinks
            current_sink_column = subset(df, :("SINK ID") => ByRow(==(sink)))[!, measurement]
            length_diff = highest_sink_size - length(current_sink_column)
            current_measurement_data[!, :($sink)] = [current_sink_column; fill(missing, length_diff)]
        end
        push!(dataframes, current_measurement_data)
        push!(sheet_names, measurement)
    end
    XLSX.openxlsx(transformed_file_path, mode="w") do xf
        for i in eachindex(sheet_names)
            sheet_name = sheet_names[i]
            df = dataframes[i]
            if i == firstindex(sheet_names)
                sheet = xf[1]
                XLSX.rename!(sheet, sheet_name)
                XLSX.writetable!(sheet, df, write_columnnames=false)
            else
                sheet = XLSX.addsheet!(xf, sheet_name)
                XLSX.writetable!(sheet, df, write_columnnames=false)
            end
        end
    end
end

function create_input_viz_data(path::String)
    sinks = read_raw_data(path)::Vector{Sink}

    sink1 = sinks[begin]
    inner_percentile = 95
    alpha_ = 0.9
    bandwidths = default_bandwidth.(collect(eachmeasurement(sink1)), alpha_, inner_percentile)
    raw_densities = make_densities.(sinks; bandwidths, inner_percentile)
    densities, domains = standardize_KDEs(raw_densities) #Made it to here so far
    densitytensor = DensityTensor(densities, domains, sinks)
    setsourcename!(densitytensor, "sink")

    measurements = getmeasurements(densitytensor)
    sinks = getsourcenames(densitytensor)
    measurement_data = []
    for measurement in measurements
        domain = getdomain(densitytensor, measurement)
        densities = eachdensity(densitytensor, measurement)
        grouped_data = [] # TODO: use a Dict instead of an array
        # assumes each vector of densities have the same length
        for (domain_val, density_index) in zip(domain, eachindex(densities[1]))
            grouped_data_point = Dict("domain" => domain_val)
            for (sink, density) in zip(sinks, densities)
                grouped_data_point["sink $sink"] = density[density_index]
            end
            push!(grouped_data, grouped_data_point)
        end
        mDict = Dict("name" => measurement, "data" => grouped_data)
        push!(measurement_data, mDict)
    end
    Dict("measurement_data" => measurement_data, "sinks" => ["sink $sink" for sink in sinks])
end

function rank_sources(path::String)
    sinks = read_raw_data(path)::Vector{Sink}
    sink1 = sinks[begin]
    inner_percentile = 95
    alpha_ = 0.9
    bandwidths = default_bandwidth.(collect(eachmeasurement(sink1)), alpha_, inner_percentile)
    raw_densities = make_densities.(sinks; bandwidths, inner_percentile)
    densities, domains = standardize_KDEs(raw_densities)
    densitytensor = DensityTensor(densities, domains, sinks)

    # Input Viz Graph Data
    measurements = getmeasurements(densitytensor)
    sinks_inp = getsourcenames(densitytensor)
    measurement_data = []
    for measurement in measurements
        domain = getdomain(densitytensor, measurement)
        densities = eachdensity(densitytensor, measurement)
        grouped_data = []
        # assumes each vector of densities have the same length
        for (domain_val, density_index) in zip(domain, eachindex(densities[1]))
            grouped_data_point = Dict("domain" => domain_val)
            for (sink, density) in zip(sinks_inp, densities)
                grouped_data_point["sink $sink"] = density[density_index]
            end
            push!(grouped_data, grouped_data_point)
        end
        mDict = Dict("name" => measurement, "data" => grouped_data)
        push!(measurement_data, mDict)
    end

    Y = array(densitytensor)
    ranks = 1:min(size(Y)[1], 10)+1
    maxiter = 6000
    tol = 1e-5
    Cs, Fs, all_rel_errors, norm_grads, dist_Ncones = ([] for _ in 1:5)
    Y_fibres = eachslice(Y, dims=(1, 2))
    Y_fibres ./= sum.(Y_fibres)
    for rank in ranks
        C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, rank; projection=:nnscale, maxiter, tol, rescale_Y=false)
        push!.(
            (Cs, Fs, all_rel_errors, norm_grads, dist_Ncones),
            (C, F, rel_errors, norm_grad, dist_Ncone)
        )
    end
    relative_errors = map(x -> x[end], all_rel_errors)
    standard_relative_errors = standard_curvature(map(x -> x[end], relative_errors))

    all_rel_errors = all_rel_errors[1:end-1]
    relative_errors = relative_errors[1:end-1]
    standard_relative_errors = standard_relative_errors[1:end-1]

    best_rank = argmax(standard_relative_errors)
    C, F, rel_errors, norm_grad, dist_Ncone = getindex.(
        (Cs, Fs, all_rel_errors, norm_grads, dist_Ncones),
        best_rank
    )

    # Rank Loss Graph Data
    standard_rank_loss_data = []
    for (rank, rel_error) in zip(ranks, standard_relative_errors)
        push!(standard_rank_loss_data, Dict("rank" => rank, "loss" => rel_error))
    end

    rank_loss_data = []
    for (rank, rel_error) in zip(ranks, relative_errors)
        push!(rank_loss_data, Dict("rank" => rank, "loss" => rel_error))
    end

    F = DensityTensor(F, domains, getmeasurements(densitytensor))
    setsourcename!(F, "learned source")
    C = NamedArray(C, dimnames=("sink", "learned source"))

    # Learned Coefficients Graph Data
    learned_sources = ["source $source" for source in names(C, 2)]
    learned_coefficients = []
    for value in names(C, 1)
        learned_coefficient = Dict("name" => "sink $value", "data" => C[:"sink"=>value])
        push!(learned_coefficients, learned_coefficient)
    end

    # Learned Densities Graph Data
    learned_densities = []
    # No need to normalize since every distribution on the same plot has the same scale
    sources = getsourcenames(F)
    sourcename = getsourcename(F)
    for (name, measurement, domain) in zip(getmeasurements(F), eachmeasurement(F), getdomains(F))
        learned_densities_per_measurement = []
        for (index, value) in enumerate(domain)
            point = Dict("domain" => value)
            for source in sources
                point["source $source"] = measurement[source, index]
            end
            push!(learned_densities_per_measurement, point)
        end
        h = Dict("name" => name, "data" => learned_densities_per_measurement)
        push!(learned_densities, h)
    end

    # Source Attribution Graph Data
    source_identification_per_sink = []
    for (sink_number, sink) in zip(sinks_inp, sinks)
        source_indexes, source_likelihoods = zip(
            map(g -> estimate_which_source(g, F, all_likelihoods=true), sink)...)
        loglikelihood_ratios = confidence_score(source_likelihoods)
        source_identification = Dict("sources" => collect(source_indexes), "loglikelihood_ratios" => loglikelihood_ratios)
        push!(source_identification_per_sink, Dict("name" => "sink $sink_number", "data" => source_identification))
    end

    clean_inf(Dict(
        "measurement_data" => measurement_data,
        "sinks" => ["sink $sink" for sink in sinks_inp],
        "standard_rank_loss_data" => standard_rank_loss_data,
        "rank_loss_data" => rank_loss_data,
        "best_rank" => best_rank,
        "learned_densities" => learned_densities,
        "learned_densities_sources" => ["source $source" for source in sources],
        "learned_coefficients" => learned_coefficients,
        "source_identification_per_sink" => source_identification_per_sink,
        "learned_coefficients_sources" => learned_sources))
end


function rank_sources_custom_rank(path::String, rank::Int)
    sinks = read_raw_data(path)::Vector{Sink}
    sink1 = sinks[begin]
    inner_percentile = 95
    alpha_ = 0.9
    bandwidths = default_bandwidth.(collect(eachmeasurement(sink1)), alpha_, inner_percentile)
    raw_densities = make_densities.(sinks; bandwidths, inner_percentile)
    densities, domains = standardize_KDEs(raw_densities)
    densitytensor = DensityTensor(densities, domains, sinks)

    # Input Viz Graph Data
    measurements = getmeasurements(densitytensor)
    sinks_inp = getsourcenames(densitytensor)
    measurement_data = []
    for measurement in measurements
        domain = getdomain(densitytensor, measurement)
        densities = eachdensity(densitytensor, measurement)
        grouped_data = [] # TODO: use a Dict instead of an array
        # assumes each vector of densities have the same length
        for (domain_val, density_index) in zip(domain, eachindex(densities[1]))
            grouped_data_point = Dict("domain" => domain_val)
            for (sink, density) in zip(sinks_inp, densities)
                grouped_data_point["sink $sink"] = density[density_index]
            end
            push!(grouped_data, grouped_data_point)
        end
        mDict = Dict("name" => measurement, "data" => grouped_data)
        push!(measurement_data, mDict)
    end

    Y = array(densitytensor)
    maxiter = 6000
    tol = 1e-5
    Y_fibres = eachslice(Y, dims=(1, 2))
    Y_fibres ./= sum.(Y_fibres)
    C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(Y, rank; projection=:nnscale, maxiter, tol, rescale_Y=false)
    best_rank = rank

    F = DensityTensor(F, domains, getmeasurements(densitytensor))
    setsourcename!(F, "learned source")
    C = NamedArray(C, dimnames=("sink", "learned source"))

    # Learned Coefficients Graph Data
    learned_sources = ["source $source" for source in names(C, 2)]
    learned_coefficients = []
    for value in names(C, 1)
        learned_coefficient = Dict("name" => "sink $value", "data" => C[:"sink"=>value])
        push!(learned_coefficients, learned_coefficient)
    end

    # Learned Densities Graph Data
    # Learned Densities Graph Data
    learned_densities = []
    # No need to normalize since every distribution on the same plot has the same scale
    sources = getsourcenames(F)
    sourcename = getsourcename(F)
    for (name, measurement, domain) in zip(getmeasurements(F), eachmeasurement(F), getdomains(F))
        learned_densities_per_measurement = []
        for (index, value) in enumerate(domain)
            point = Dict("domain" => value)
            for source in sources
                point["source $source"] = measurement[source, index]
            end
            push!(learned_densities_per_measurement, point)
        end
        h = Dict("name" => name, "data" => learned_densities_per_measurement)
        push!(learned_densities, h)
    end

    # Source Attribution Graph Data
    source_identification_per_sink = []
    for (sink_number, sink) in zip(sinks_inp, sinks)
        source_indexes, source_likelihoods = zip(
            map(g -> estimate_which_source(g, F, all_likelihoods=true), sink)...)
        loglikelihood_ratios = confidence_score(source_likelihoods)
        source_identification = Dict("sources" => collect(source_indexes), "loglikelihood_ratios" => loglikelihood_ratios)
        push!(source_identification_per_sink, Dict("name" => "sink $sink_number", "data" => source_identification))
    end


    clean_inf(Dict(
        "measurement_data" => measurement_data,
        "sinks" => ["sink $sink" for sink in sinks_inp],
        "best_rank" => best_rank,
        "learned_densities" => learned_densities,
        "learned_densities_sources" => ["source $source" for source in sources],
        "learned_coefficients" => learned_coefficients,
        "source_identification_per_sink" => source_identification_per_sink,
        "learned_coefficients_sources" => learned_sources))
end

end
