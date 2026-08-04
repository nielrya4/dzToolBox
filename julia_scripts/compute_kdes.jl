#!/usr/bin/env julia
"""
Compute empirical KDEs for multivariate samples using the SedimentSourceAnalysis package.

Usage: julia compute_kdes.jl <input_excel> <output_json>

This script calls create_input_viz_data() from dzgrainalyzer_helpers.jl
to compute standardized KDEs for all samples and features.
"""

using Pkg

# Install required packages if not already installed
packages = ["XLSX", "NamedArrays", "DataFrames", "JSON", "MatrixTensorFactor", "SedimentSourceAnalysis"]
for pkg in packages
    if !haskey(Pkg.project().dependencies, pkg)
        if pkg == "MatrixTensorFactor"
            Pkg.add(url="https://github.com/MPF-Optimization-Laboratory/MatrixTensorFactor.jl.git", rev="main")
        elseif pkg == "SedimentSourceAnalysis"
            Pkg.add(url="https://github.com/njericha/Sediment-Source-Analysis.jl.git")
        else
            Pkg.add(pkg)
        end
    end
end

# Include the dzgrainalyzer helper module
include("dzgrainalyzer_helpers.jl")
using .SourceAnalysisHelpers
using JSON

function main()
    if length(ARGS) != 2
        println(stderr, "Usage: julia compute_kdes.jl <input_excel> <output_json>")
        exit(1)
    end

    input_file = ARGS[1]
    output_file = ARGS[2]

    try
        println("Computing empirical KDEs for $input_file")

        # Call the create_input_viz_data function to compute KDEs
        results = Dict{String, Any}(SourceAnalysisHelpers.create_input_viz_data(input_file))

        # Add status field
        results["status"] = "success"

        # Write results to JSON file
        open(output_file, "w") do f
            JSON.print(f, results, 2)
        end

        println("KDE computation complete. Results written to $output_file")

    catch e
        println(stderr, "Error during KDE computation: $e")
        println(stderr, stacktrace(catch_backtrace()))

        # Write error to JSON
        error_result = Dict(
            "status" => "error",
            "error" => string(e),
            "stacktrace" => string(stacktrace(catch_backtrace()))
        )

        open(output_file, "w") do f
            JSON.print(f, error_result, 2)
        end

        exit(1)
    end
end

main()
