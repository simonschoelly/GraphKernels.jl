
"""
    WeisfeilerLehmanGraphKernel(;base_kernel, num_iterations=5) <: AbstractGraphKernel
"""
struct WeisfeilerLehmanGraphKernel{BK<:AbstractGraphKernel} <: AbstractGraphKernel

    base_kernel::BK
    num_iterations::Int

    function WeisfeilerLehmanGraphKernel(base_kernel::AbstractGraphKernel, num_iterations::Integer)

        num_iterations >= 1 || throw(DomainError(num_iterations, "num_iterations must be >= 1"))
        return new{typeof(base_kernel)}(base_kernel, num_iterations)
    end
end

function WeisfeilerLehmanGraphKernel(;base_kernel::AbstractGraphKernel, num_iterations::Integer=5)

    return WeisfeilerLehmanGraphKernel(base_kernel, num_iterations)
end

function preprocessed_form(kernel::WeisfeilerLehmanGraphKernel, g::AbstractGraph)

    nvg = nv(g)
    num_iterations = kernel.num_iterations

    vertexvals = Matrix{UInt}(undef, nvg, num_iterations)

    for u in vertices(g)
        vertexvals[u, 1] = hash(get_vertexval(g, u, :))
    end

    sort_buffer = Vector{UInt}(undef, Δ(g))
    for i in 2:num_iterations
        for u in vertices(g)
            deg_u = degree(g, u)
            for (j, v) in enumerate(neighbors(g, u))
                sort_buffer[j] = vertexvals[u, i-1]
            end
            sort!(@view sort_buffer[1:deg_u])
            h = vertexvals[u, i-1]
            for j in 1:deg_u
                h = hash(sort_buffer[j], h)
            end
            vertexvals[u, i] = h
        end
    end

    return [preprocessed_form(kernel.base_kernel, ReplacedVertexVals(g, vertexvals[:, i])) for i in  1:num_iterations]
end

function apply_preprocessed(kernel::WeisfeilerLehmanGraphKernel, pre1, pre2)

    return sum(t -> apply_preprocessed(kernel.base_kernel, t[1], t[2]), zip(pre1, pre2))
end

