using LinearAlgebra
using Magma

function magma_version()
    Magma.magma_init();
    major = Ref{Magma.LibMagma.magma_int_t}(-1);
    minor = Ref{Magma.LibMagma.magma_int_t}(-1);
    micro = Ref{Magma.LibMagma.magma_int_t}(-1);
    Magma.LibMagma.magma_version(major, minor, micro);
    Magma.magma_finalize();
    any(x -> x[] < 0, (major, minor, micro)) && error("Error while getting MAGMA version");
    return VersionNumber(major[], minor[], micro[])
end

function eigvals_magma(A::AbstractMatrix)
    hA = Array(A);
    Magma.magma_init();
    results = nothing;
    if Magma.LibMagma.magma_num_gpus() == 1
        results = Magma.geev!('N','N',hA);
    else
        results = Magma.geev_m!('N','N',hA);
    end
    Magma.magma_finalize();
    return results[1];
end

function eigen_magma(A::AbstractMatrix)
    hA = Array(A);
    Magma.magma_init();
    results = nothing;
    if Magma.LibMagma.magma_num_gpus() == 1
        results = Magma.geev!('V','V',hA);
    else
        results = Magma.geev_m!('V','V',hA);
    end
    Magma.magma_finalize();
    return results;
end