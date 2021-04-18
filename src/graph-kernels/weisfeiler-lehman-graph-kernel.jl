
"""
    WeisfeilerLehmanGraphKernel(;base_kernel, vertex_labels=Auto(), num_iterations=5) <: AbstractGraphKernel

A graph kernel based on the Weisfeiler-Lehman isomorphism test.

The kernel is run for multiple iterations on a graph, and adjacent vertex values for each vertex
are collected and then hashed to a new vertex value. Each iteration generates a new graph with replaced
vertex values. Afterwards `base_kernel` is run on all the new graphs and the results are summed up
to generate a final value.

# Keywords
- `base_kernel`: The kernel that is applied to the newly generated graphs.
- `vertex_labels`: Which vertex labels to consider for generating the initial hash value of each vertex.
    Either `Auto`, `:` (all), or a tuple of vertex value keys.
- `num_iterations`: For how many iterations we generate graphs with new vertex values.

# References
https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
"""
struct WeisfeilerLehmanGraphKernel{BK<:AbstractGraphKernel, VL <: LabelsType} <: AbstractGraphKernel

    base_kernel::BK
    vertex_labels::VL
    num_iterations::Int

    function WeisfeilerLehmanGraphKernel(;base_kernel::AbstractGraphKernel, vertex_labels=Auto(), num_iterations::Integer=5)

        num_iterations >= 1 || throw(DomainError(num_iterations, "num_iterations must be >= 1"))

        return new{typeof(base_kernel), typeof(vertex_labels)}(base_kernel, vertex_labels, num_iterations)
    end
end


# TODO it might be useful to run the base_kernel (or a separate kernel) on the
# initial graph without replaced vertex values. One might also consider keeping
# a part of the initial vertex values for each graph.
function preprocessed_form(kernel::WeisfeilerLehmanGraphKernel, g::AbstractGraph)

    nvg = nv(g)
    num_iterations = kernel.num_iterations

    vertexvals = Matrix{UInt}(undef, nvg, num_iterations)
    _fill_initial_vertexvals!(vertexvals, g, kernel)

    sort_buffer = Vector{UInt}(undef, Δ(g))
    for i in 2:num_iterations
        @inbounds for u in vertices(g)
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

    return [preprocessed_form(kernel.base_kernel, ReplacedVertexVals(g, @inbounds vertexvals[:, i])) for i in  1:num_iterations]
end

function _fill_initial_vertexvals!(vertexvals, g, kernel)

    vertex_labels = _labels(kernel.vertex_labels, vertexvals_type(g))
    for u in vertices(g)
        # TODO we can iteratively calculate the hash instead of creating a tuple
        @inbounds vertexvals[u, 1] = hash(tuple((get_vertexval(g, u, i) for i in vertex_labels)...))
    end
end

function apply_preprocessed(kernel::WeisfeilerLehmanGraphKernel, pre1, pre2)

    return sum(t -> apply_preprocessed(kernel.base_kernel, t[1], t[2]), zip(pre1, pre2))
end

