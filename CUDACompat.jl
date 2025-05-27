# Extending CUDA.jl; consider PR-ing these changes

using LinearAlgebra: Eigen, eigen, eigvals, BlasComplex, BlasReal
using CUDA: CuMatrix, issymmetric, ishermitian
using CUDA.CUSOLVER: syevd!, heevd!, Xgeev!

# eigen

function LinearAlgebra.eigen(A::CuMatrix{T}) where {T<:BlasReal}
    A2 = copy(A)
    r = Xgeev!('N', 'V', A2)
    Eigen(r[1], r[3])
end
function LinearAlgebra.eigen(A::CuMatrix{T}) where {T<:BlasComplex}
    A2 = copy(A)
    r = Xgeev!('N', 'V', A2)
    Eigen(r[1], r[3])
end

# eigvals

function LinearAlgebra.eigvals(A::Symmetric{T,<:CuMatrix}) where {T<:BlasReal}
    A2 = copy(A.data)
    syevd!('N', 'U', A2)[1]
end
function LinearAlgebra.eigvals(A::Hermitian{T,<:CuMatrix}) where {T<:BlasComplex}
    A2 = copy(A.data)
    heevd!('N', 'U', A2)[1]
end
function LinearAlgebra.eigvals(A::Hermitian{T,<:CuMatrix}) where {T<:BlasReal}
    eigvals(Symmetric(A))
end

function LinearAlgebra.eigvals(A::CuMatrix{T}) where {T<:BlasReal}
    A2 = copy(A)
    Xgeev!('N', 'N', A2)[1]
end
function LinearAlgebra.eigvals(A::CuMatrix{T}) where {T<:BlasComplex}
    A2 = copy(A)
    Xgeev!('N', 'N', A2)[1]
end

# eigvecs

function LinearAlgebra.eigvecs(A::Symmetric{T,<:CuMatrix}) where {T<:BlasReal}
    A2 = copy(A.data)
    syevd!('V', 'U', A2)[2]
end
function LinearAlgebra.eigvecs(A::Hermitian{T,<:CuMatrix}) where {T<:BlasComplex}
    A2 = copy(A.data)
    heevd!('V', 'U', A2)[2]
end
function LinearAlgebra.eigvecs(A::Hermitian{T,<:CuMatrix}) where {T<:BlasReal}
    eigvecs(Symmetric(A))
end

function LinearAlgebra.eigvecs(A::CuMatrix{T}) where {T<:BlasReal}
    A2 = copy(A)
    Xgeev!('N', 'V', A2)[3]
end
function LinearAlgebra.eigvecs(A::CuMatrix{T}) where {T<:BlasComplex}
    A2 = copy(A)
    Xgeev!('N', 'V', A2)[3]
end
