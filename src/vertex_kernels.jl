

# ================================================================
#     AbstractVertexKernel
# ================================================================

abstract type AbstractVertexKernel end

function (kernel::AbstractVertexKernel)((g1, v1), (g2, v2))

    return kernel(g1, v1, g2, v2)
end

# ================================================================
#     ConstVertexKernel
# ================================================================

struct ConstVertexKernel <: AbstractVertexKernel
    c::Float64
end

(kernel::ConstVertexKernel)(g1, v1, g2, v2) = kernel.c

# ================================================================
#     DiracVertexKernel
# ================================================================

struct DiracVertexKernel <: AbstractVertexKernel end

(kernel::DiracVertexKernel)(g1::AbstractValGraph, v1, g2::AbstractValGraph, v2) =
    (get_vertexval(g1, v1, :) == get_vertexval(g2, v2, :)) ? 1.0 : 0.0

# ================================================================
#     DotVertexKernel
# ================================================================

struct DotVertexKernel <: AbstractVertexKernel end

(kernel::DotVertexKernel)(g1::AbstractValGraph, v1, g2::AbstractValGraph, v2) =
    Float64(dot(get_vertexval(g1, v1, :), get_vertexval(g2, v2, :)))

