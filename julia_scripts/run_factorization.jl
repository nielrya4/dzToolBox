#!/usr/bin/env julia
"""
Run tensor factorization with a custom rank using the original dzgrainalyzer code.

Usage: julia run_factorization.jl <input_excel> <rank> <output_json>

This script calls rank_sources_custom_rank() from dzgrainalyzer_helpers.jl
which matches the original dzgrainalyzer implementation exactly.
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
    if length(ARGS) != 3
        println(stderr, "Usage: julia run_factorization.jl <input_excel> <rank> <output_json>")
        exit(1)
    end

    input_file = ARGS[1]
    rank = parse(Int, ARGS[2])
    output_file = ARGS[3]

    try
        println("Running factorization with rank=$rank on $input_file")

        # Call the original dzgrainalyzer function
        results = SourceAnalysisHelpers.rank_sources_custom_rank(input_file, rank)

        # Add status field
        results["status"] = "success"

        # Write results to JSON file
        open(output_file, "w") do f
            JSON.print(f, results, 2)
        end

        println("Factorization complete. Results written to $output_file")

    catch e
        println(stderr, "Error during factorization: $e")
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