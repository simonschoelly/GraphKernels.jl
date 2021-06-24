
struct GraphSVMModel
    svm::LIBSVM.SVM
    kernel::KernelFunctions.Kernel
    graphs::AbstractVector
end


function svmtrain(graphs::AbstractVector{<:AbstractGraph}, labels, kernel::KernelFunctions.Kernel; kwargs...)

    X = kernelmatrix(kernel, graphs)
    svm = svmtrain(X, labels, kernel=Kernel.Precomputed; kwargs...)

    return GraphSVMModel(svm, kernel, graphs)
end

function svmpredict(model::GraphSVMModel, unpredicted_graphs::AbstractVector{<:AbstractGraph})

    graphs = model.graphs
    kernel = model.kernel

    # TODO might only be necessary to do the calculations for support vectors
    X = kernelmatrix(kernel, graphs, unpredicted_graphs)
    return svmpredict(model.svm, X)[1] # for simplicity return only the labels for now
end
