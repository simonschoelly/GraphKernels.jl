
"""
    Auto

A singleton type to denote that some argument to a function or constructor should
be automatically determined.
"""
struct Auto end

const LabelsType = Union{Auto, Colon, Tuple{Vararg{Int}}, Tuple{Vararg{Symbol}}}

_labels(::Colon, types) = fieldnames(types)

function _is_suitable_label_type(T)

    return T <: Union{Integer, AbstractString, AbstractChar, Symbol}
end

function _labels(::Auto, types)

    return filter(i -> _is_suitable_label_type(fieldtype(types, i)), fieldnames(types))
end

# TODO we could actually verify that the keys are valid
_labels(keys::Tuple, types) = keys

