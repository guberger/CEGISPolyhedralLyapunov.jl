module TestMain

using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGARLearningCLF.jl")
else
    using CEGARLearningCLF
end
CLC = CEGARLearningCLF

# Temporary fix
function HiGHS._check_ret(ret::Cint) 
    if ret != Cint(0) && ret != Cint(1)
        error( 
            "Encountered an error in HiGHS (Status $(ret)). Check the log " * 
            "for details.", 
        ) 
    end 
    return 
end 

sleep(0.1) # used for good printing
println("Started test")

## Parameters
method = CLC.PolyhedralPointwise(2)
A1 = [-0.5 1.0; -1.0 -0.5]
A2 = [-0.3 0.0; -0.5 -0.3]
A_list = [A1, A2]
params = (Gain=5, tol_faces=1e-5, tol_derivative=1e-3, print_period=1)
solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Solving
c_list, x_dx_list = CLC.learning_clf_process(method, A_list, params, solver)

end # TestMain