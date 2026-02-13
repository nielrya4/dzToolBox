#!/usr/bin/env julia
# Test if fiber normalization is working correctly

using MatrixTensorFactor
using Statistics

# Create test tensor
Y = rand(5, 10, 3)
println("Original Y min/max: ", minimum(Y), " / ", maximum(Y))
println("Original Y sum of first fiber: ", sum(Y[1, :, :]))

# Normalize fibers like our code does
Y_fibres = eachslice(Y, dims=(1, 2))
Y_fibres ./= sum.(Y_fibres)

println("\nAfter normalization:")
println("Y min/max: ", minimum(Y), " / ", maximum(Y))
println("Sum of first fiber: ", sum(Y[1, :, :]))
println("Sum of all fibers (should all be 1.0):")
for i in 1:size(Y, 1)
    fiber_sum = sum(Y[i, :, :])
    println("  Fiber $i: ", fiber_sum)
end
