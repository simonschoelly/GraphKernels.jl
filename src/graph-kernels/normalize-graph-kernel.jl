

"""
    NormalizeGraphKernel(k::AbstractGraphKernel) <: AbstractGraphKernel

Graph kernel `k̃`  that wraps around a kernel `k` and scales the output such 
`k̃(g1, g2) = k(g1, g2) / sqrt(k(g1, g1), k(g2, g2)).
"""
struct NormalizeGraphKernel{IK<:AbstractGraphKernel} <: AbstractGraphKernel

    inner_kernel::IK
end

function preprocessed_form(kernel::NormalizeGraphKernel, g::AbstractGraph)

    inner = kernel.inner_kernel
    pre_inner = preprocessed_form(inner, g)
    k_ii = apply_preprocessed(inner, pre_inner, pre_inner)
    return (k_ii, pre_inner)
end

function apply_preprocessed(kernel::NormalizeGraphKernel, pre1, pre2)

    inner = kernel.inner_kernel
    k_11, pre_inner1 = pre1
    k_22, pre_inner2 = pre2

    k_12 = apply_preprocessed(inner, pre_inner1, pre_inner2)

    return k_12 / sqrt(k_11 * k_22)
end

