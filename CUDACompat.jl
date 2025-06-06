# Extending CUDA.jl; consider PR-ing these changes

using LinearAlgebra: Eigen, eigen, eigvals, BlasComplex, BlasReal
using CUDA: CuMatrix, issymmetric, ishermitian
using CUDA.CUSOLVER: syevd!, heevd!, Xgeev!

# eigen

function LinearAlgebra.eigen(A::CuMatrix{T}) where {T<:BlasReal}
    A2 = copy(A)
    W, _, VR = Xgeev!('N', 'V', A2)
    C = Complex{T}
    U = CuMatrix{C}([1.0 1.0; im -im])
    VR = CuMatrix{C}(VR)
    h_W = collect(W)
    n = length(W)
    j = 1
    while j <= n
        if imag(h_W[j]) == 0
            j += 1
        else
            VR[:, j:(j + 1)] .= VR[:, j:(j + 1)] * U
            j += 2
        end
    end
    return Eigen(W, VR)
end
function LinearAlgebra.eigen(A::CuMatrix{T}) where {T<:BlasComplex}
    A2 = copy(A)
    r = Xgeev!('N', 'V', A2)
    return Eigen(r[1], r[3])
end

# eigvals

function LinearAlgebra.eigvals(A::Symmetric{T, <:CuMatrix}) where {T <: BlasReal}
    A2 = copy(A.data)
    return syevd!('N', 'U', A2)
end
function LinearAlgebra.eigvals(A::Hermitian{T, <:CuMatrix}) where {T <: BlasComplex}
    A2 = copy(A.data)
    return heevd!('N', 'U', A2)
end
function LinearAlgebra.eigvals(A::Hermitian{T, <:CuMatrix}) where {T <: BlasReal}
    return eigvals(Symmetric(A))
end

function LinearAlgebra.eigvals(A::CuMatrix{T}) where {T <: BlasReal}
    A2 = copy(A)
    return Xgeev!('N', 'N', A2)[1]
end
function LinearAlgebra.eigvals(A::CuMatrix{T}) where {T <: BlasComplex}
    A2 = copy(A)
    return Xgeev!('N', 'N', A2)[1]
end

# eigvecs

function LinearAlgebra.eigvecs(A::Symmetric{T, <:CuMatrix}) where {T <: BlasReal}
    E = eigen(A)
    return E.vectors
end
function LinearAlgebra.eigvecs(A::Hermitian{T, <:CuMatrix}) where {T <: BlasComplex}
    E = eigen(A)
    return E.vectors
end
function LinearAlgebra.eigvecs(A::Hermitian{T, <:CuMatrix}) where {T <: BlasReal}
    return eigvecs(Symmetric(A))
end

function LinearAlgebra.eigvecs(A::CuMatrix{T}) where {T <: BlasReal}
    E = eigen(A)
    return E.vectors
end
function LinearAlgebra.eigvecs(A::CuMatrix{T}) where {T <: BlasComplex}
    E = eigen(A)
    return E.vectors
end