#!/usr/bin/env julia

# Test script to debug rank selection
using MatrixTensorFactor
using SedimentSourceAnalysis
using Statistics

# Read the test data
input_path = "/home/ryan/PycharmProjects/dzToolBox/static/global/docs/example_tensor_data.xlsx"
sinks = read_raw_data(input_path)

println("Number of sinks: ", length(sinks))

# Use exact same parameters as original
sink1 = sinks[begin]
inner_percentile = 95
alpha_ = 0.9

# Calculate bandwidths
bandwidths = default_bandwidth.(collect(eachmeasurement(sink1)), alpha_, inner_percentile)
println("Bandwidths: ", bandwidths)

# Generate KDEs
raw_densities = make_densities.(sinks; bandwidths, inner_percentile)
densities, domains = standardize_KDEs(raw_densities)
densitytensor = DensityTensor(densities, domains, sinks)

# Build tensor array
Y = array(densitytensor)
println("Tensor shape: ", size(Y))

# Test with 1:11 like original
ranks = 1:11
maxiter = 6000
tol = 1e-5

# Storage for results
Cs, Fs, all_rel_errors, norm_grads, dist_Ncones = ([] for _ in 1:5)

# Normalize fibers (same as original)
Y_fibres = eachslice(Y, dims=(1, 2))
Y_fibres ./= sum.(Y_fibres)

# Run factorization for each rank
for rank in ranks
    println("Processing rank $rank...")
    C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(
        Y, rank;
        projection=:nnscale,
        maxiter=maxiter,
        tol=tol,
        rescale_Y=false
    )
    push!(Cs, C)
    push!(Fs, F)
    push!(all_rel_errors, rel_errors)
    push!(norm_grads, norm_grad)
    push!(dist_Ncones, dist_Ncone)
end

# Get final relative errors
relative_errors = map(x -> x[end], all_rel_errors)
println("\nRelative errors (all ranks 1-11): ", relative_errors)

# Calculate curvature EXACTLY like original
standard_relative_errors = standard_curvature(relative_errors)
println("Standard curvature (all ranks): ", standard_relative_errors)

# Exclude last rank from selection (same as original)
all_rel_errors_truncated = all_rel_errors[1:end-1]
relative_errors_truncated = relative_errors[1:end-1]
standard_relative_errors_truncated = standard_relative_errors[1:end-1]

println("\nRelative errors (truncated 1-10): ", relative_errors_truncated)
println("Standard curvature (truncated 1-10): ", standard_relative_errors_truncated)

# Find best rank
best_rank = argmax(standard_relative_errors_truncated)
println("\nBest rank index: ", best_rank)
println("Best rank value: ", ranks[best_rank])
println("Max curvature: ", standard_relative_errors_truncated[best_rank])
