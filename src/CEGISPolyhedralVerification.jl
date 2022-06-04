module CEGISPolyhedralVerification

using LinearAlgebra
using JuMP

_RSC_ = JuMP.MathOptInterface.ResultStatusCode
_TSC_ = JuMP.MathOptInterface.TerminationStatusCode
_VT_ = Vector{Float64}
_MT_ = Matrix{Float64}

struct LinForm
    lin::_VT_
end
_eval(lf::LinForm, point) = dot(lf.lin, point)

include("polyhedra.jl")

include("generator.jl")
include("verifier.jl")
include("learner.jl")

end # module
