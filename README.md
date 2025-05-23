# MAGMA Tests

This is just a small collection of scripts to test [MAGMA](https://github.com/icl-utk-edu/magma) via [MAGMA_jll](https://github.com/JuliaBinaryWrappers/MAGMA_jll.jl) and [Magma.jl](https://github.com/matteosecli/Magma.jl).

MAAGMA is built by linking against [libblastrampoline](https://github.com/JuliaLinearAlgebra/libblastrampoline), so it's also easy to compare across different BLAS providers such as [MKL](https://github.com/JuliaLinearAlgebra/MKL.jl) and [NVPL](https://github.com/matteosecli/NVPL.jl).

Right now, I'm just testing `Xgeev`.

## Usage

It should be enough to just clone and run (with Julia installed):
```bash
git clone https://github.com/matteosecli/magma-tests.git && cd magma-tests
julia --project testXgeev.jl N | tee results.log
```
where `N` is the matrix size or a sequence of sizes to test.