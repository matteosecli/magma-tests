using Pkg
#Pkg.resolve()
Pkg.instantiate()

using BenchmarkTools
using CUDA
using Libdl
using LinearAlgebra
using MKL_jll

include("CUDACompat.jl")
include("MAGMACompat.jl")

# using Pkg, Artifacts, MAGMA_jll
# Pkg.Artifacts.ensure_artifact_installed("MAGMA",Artifacts.find_artifacts_toml(pathof(MAGMA_jll)))
# Base.compilecache(Base.identify_package("MAGMA_jll"))
# Base.compilecache(Base.identify_package("Magma"))
# using CUDA
# CUDA.set_runtime_version!(v"12.8")  # At least 12.6.2 is needed
# using Magma
# Magma.LibMagma.set_libmagma_path(...)  # Set the path to the Magma library if necessary

function runTest(testTypes, Nvalues)
    for N in Nvalues
        println("\n===== Testing N = $N =====")
        for T in testTypes
            println("\n===== Testing $T =====")

            println("LinearAlgebra (CPU):")
            @btime eigvals(randn($T, $N, $N))

            println("CUDA (GPU):")
            @btime eigvals(CuMatrix(randn($T,$N, $N)))

            println("MAGMA (GPU):")
            @btime eigvals_magma(randn($T, $N, $N))
        end
    end
end

println("===== Running MAGMA tests =====")
println(rpad("MAGMA version:", 23), lpad(magma_version(), 8))
println(rpad("CUDA driver version:", 23), lpad(CUDA.driver_version(), 8))
println(rpad("CUDA runtime version:", 23), lpad(CUDA.runtime_version(), 8))
println(BLAS.get_config()) # Print BLAS configuratio`n

Nvalues = isempty(ARGS) ? [1024] : parse.(Int, ARGS) # Matrix sizes to test
testTypes = [Float32, Float64, ComplexF32, ComplexF64]

# Run tests with default LinearAlgebra provider
runTest(testTypes, Nvalues)

if MKL_jll.is_available()
    using MKL

    # Run tests for MKL
    println("\n===== Switching to MKL =====")
    # Print MKL version here
    #println(rpad("MKL version:", 23), lpad(MKL_jll.version(), 8))
    println(BLAS.get_config()) # Print BLAS configuration

    # Run tests with MKL LinearAlgebra provider
    runTest(testTypes, Nvalues)
end

if !isempty(Libdl.find_library("libnvpl_blas_ilp64_gomp"))
    using NVPL

    # Run tests for NVPL
    println("\n===== Switching to NVPL =====")
    # Print NVPL version here
    println(rpad("NVPL BLAS version:", 23), lpad(NVPL.blas_get_version(), 8))
    println(BLAS.get_config()) # Print BLAS configuration

    # Run tests with NVPL LinearAlgebra provider
    runTest(testTypes, Nvalues)
end