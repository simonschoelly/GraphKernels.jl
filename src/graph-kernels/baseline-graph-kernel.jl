
using DataStructures: counter, inc!

const LabelsType = Union{Auto, Colon, Tuple{Vararg{Int}}, Tuple{Vararg{Symbol}}}

"""
    NoGraphBaselineGraphKernel(;vertex_labels=Auto(), edge_labels=Auto(), gamma=0.5)

A graph kernel that does not use structural information of the graph. Instead vertices and
edges are considered as sets and features are created by splitting these sets into classes
according to their labels. Then the kernel function is calculated by applying an RBF kernel
on these features.

This kernel provides a good baseline, as to verify if a graph kernel, that uses structural
information, is meaningful on some data.

# Keywords
- `vertex_labels`: Which vertex labels to consider. Either `Auto`, `:` (all), or a tuple of vertex value keys.
- `edge_labels`: Which edge labels to consider. Either `Auto`, `:` (all), or a tuple of edge value keys.
- `gamma`: Scaling factor for the RBF kernel.

# Examples
```jldoctest
julia> k1 = BaselineGraphKernel(gamma=2.0); # custom gamma value

# no vertex and edge labels. In that case, only the number of vertices and edges are considered.
julia> k2 = BaselineGraphKernel(vertex_labels=(), edge_labels=());

# all vertex labels, and edge labels for the keys :a and :b are considered.
julia> k3 = BaselineGraphKernel(vertex_labels=:, edge_labels=(:a, :b));

# Suitable vertex and edge label keys are inferred from the types of these labels.
julia> k4 = BaselineGraphKernel(vertex_labels=Auto(), edge_labels=Auto()));
```

# References
<https://mlai.cs.uni-bonn.de/publications/schulz2019gem.pdf>
"""
struct NoGraphBaselineGraphKernel{VL <: LabelsType, EL <: LabelsType}<: AbstractGraphKernel

    vertex_labels::VL
    edge_labels::EL
    gamma::Float64

    function NoGraphBaselineGraphKernel(;vertex_labels=Auto(), edge_labels=Auto(), gamma=0.5)

        (gamma > 0 && isfinite(gamma)) || throw(DomainError(gamma, "gamma must be 0 < gamma < ∞"))

        return new{typeof(vertex_labels), typeof(edge_labels)}(vertex_labels, edge_labels, Float64(gamma))
    end
end

function preprocessed_form(kernel::NoGraphBaselineGraphKernel, g::AbstractValGraph)

    vertex_labels = _labels(kernel.vertex_labels, vertexvals_type(g))
    edge_labels = _labels(kernel.edge_labels, edgevals_type(g))

    # TODO, we should deduce the correct type instead of using Any
    vertex_class_sizes = counter(Any)
    for v in vertices(g)
        class_key = tuple((get_vertexval(g, v, i) for i in vertex_labels)...)
        inc!(vertex_class_sizes, class_key)
    end

    edge_class_sizes = counter(Any)
    for e in edges(g, :)
        class_key = tuple((get_edgeval(e, i) for i in edge_labels)...)
        inc!(edge_class_sizes, class_key)
    end

    return (vertex_class_sizes=vertex_class_sizes, edge_class_sizes=edge_class_sizes)
end

_labels(::Colon, types) = fieldnames(types)

function _is_suitable_label_type(T)

    return T <: Union{Integer, AbstractString, AbstractChar, Symbol}
end

function _labels(::Auto, types)

    return filter(i -> _is_suitable_label_type(fieldtype(types, i)), fieldnames(types))
end

# TODO we could actually verify that the keys are valid
_labels(keys::Tuple, types) = keys

function apply_preprocessed(kernel::NoGraphBaselineGraphKernel, sizes1, sizes2)

    squared_sum = 0
    # Some labels might occur in only one of the graphs. Therefore we need to loop
    # over the labels of both graphs and ensure that the squared difference of the count
    # of some label is used exactly once.
    for (class1, count1) in sizes1.vertex_class_sizes
        count2 = sizes2.vertex_class_sizes[class1]
        squared_sum += (count1 - count2)^2
    end
    for (class2, count2) in sizes2.vertex_class_sizes
        count1 = sizes1.vertex_class_sizes[class2]
        if count1 == 0
            squared_sum += count2^2
        end
    end

    for (class1, count1) in sizes1.edge_class_sizes
        count2 = sizes2.edge_class_sizes[class1]
        squared_sum += (count1 - count2)^2
    end
    for (class2, count2) in sizes2.edge_class_sizes
        count1 = sizes1.edge_class_sizes[class2]
        if count1 == 0
            squared_sum += count2^2
        end
    end

    return exp(-kernel.gamma * squared_sum)
end

