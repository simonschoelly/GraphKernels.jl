
struct GraphSVMModel
    svm::LIBSVM.SVM
    kernel::AbstractGraphKernel
    graphs::AbstractVector
end


function svmtrain(graphs::AbstractVector{<:AbstractGraph}, labels, kernel::AbstractGraphKernel; kwargs...)

    n = length(graphs)

    X = vcat(transpose(1:n), kernelmatrix(kernel, graphs))
    println("Finished calculating kernelmatrix")

    svm = svmtrain(X, labels, kernel=Kernel.Precomputed; kwargs...)

    return GraphSVMModel(svm, kernel, graphs)
end

function svmpredict(model::GraphSVMModel, unpredicted_graphs::AbstractVector{<:AbstractGraph})

    graphs = model.graphs
    kernel = model.kernel

    m = length(graphs)
    n = length(unpredicted_graphs)

    X = Matrix{Float64}(undef, m + 1, n)
    X[1, :] = 1:n
    X[2:end, :] = kernelmatrix(kernel, graphs, unpredicted_graphs)

    return svmpredict(model.svm, X)[1] # for simplicity return only the labels for now
end
