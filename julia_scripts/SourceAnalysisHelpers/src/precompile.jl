# Runs the scripts' code paths on small synthetic workbooks while the package precompiles,
# so the compiled code is cached on disk. Without this, every Julia subprocess spent
# 10-20 seconds compiling before doing any real work.
#
# SedimentSourceAnalysis's DensityTensor stores the domains as NTuple{n_measurements} axis
# names, so its code compiles separately for each number of measurements. The workload
# covers the common range; other counts still work, they just compile on first use.

const WORKLOAD_MEASUREMENT_COUNTS = 2:12

# Same layout the Celery tasks write: one sheet per measurement, one column per sink,
# one row per grain, no headers, shorter sinks padded with missing.
function write_workload_workbook(path, n_measurements, rng)
    sink_sizes = [40, 32, 25]
    XLSX.openxlsx(path, mode="w") do xf
        for i in 1:n_measurements
            sheet = i == 1 ? xf[1] : XLSX.addsheet!(xf, "m$i")
            i == 1 && XLSX.rename!(sheet, "m$i")
            for (col, size) in enumerate(sink_sizes), row in 1:size
                sheet[row, col] = 10.0 * i + randn(rng) + col
            end
        end
    end
end

function run_workload(dir)
    rng = MersenneTwister(0)
    path = joinpath(dir, "workload.xlsx")
    json_path = joinpath(dir, "workload.json")
    for n_measurements in WORKLOAD_MEASUREMENT_COUNTS
        write_workload_workbook(path, n_measurements, rng)
        bandwidths = Any[nothing for _ in 1:n_measurements]
        bandwidths[2] = 0.5
        n_samples, bandwidth_overrides = parse_kde_options(JSON.json(Dict("n_samples" => 16, "bandwidths" => bandwidths)))
        results = [
            create_input_viz_data(path; n_samples, bandwidth_overrides),
            rank_sources_custom_rank(path, 2; n_samples, bandwidth_overrides),
            select_rank(path; n_samples, bandwidth_overrides),
        ]
        # The scripts write to a file, and JSON.print specializes on the IO type
        for result in results
            open(f -> JSON.print(f, result, 2), json_path, "w")
        end
    end
end

@setup_workload begin
    @compile_workload begin
        # Silence SedimentSourceAnalysis's per-sheet @info and select_rank's progress output
        with_logger(NullLogger()) do
            redirect_stdout(devnull) do
                mktempdir(run_workload)
            end
        end
    end
end
