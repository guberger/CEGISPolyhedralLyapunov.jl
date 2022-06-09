module ExampleGenDiff

using JuMP
using Gurobi

include("../../src/CEGISPolyhedralVerification.jl")
CPV = CEGISPolyhedralVerification

nvar = 7
nloc = 1
gen = CPV.Generator(nvar, nloc)

display(gen)
nlfs = 0

datafile = "evidences"
str = readlines(string(@__DIR__, "./", datafile, ".txt"))

for (i, ln) in enumerate(str)
    ln = replace(ln, r"[\(\[\]\),]"=>"")
    words = split(ln)
    if length(words) == 10
        global nlfs = max(nlfs, parse(Int, words[2]))
        push!(gen.pos_evids, CPV.PosEvidence(
            parse(Int, words[1]),
            parse(Int, words[2]),
            parse.(Float64, words[3:9]),
            parse(Float64, words[10])
        ))
    end
    if length(words) == 22
        global nlfs = max(nlfs, parse(Int, words[2]))
        push!(gen.liecont_evids, CPV.LieContEvidence(
            parse(Int, words[1]),
            parse(Int, words[2]),
            parse.(Float64, words[3:9]),
            parse.(Float64, words[10:16]),
            parse(Float64, words[17]),
            parse(Float64, words[18]),
            parse(Float64, words[19]),
            parse(Float64, words[20]),
            parse(Float64, words[21]),
            parse(Float64, words[22])
        ))
    end
end

gen.nlfs[1] = nlfs

# display(gen)

GUROBI_ENV = Gurobi.Env()
solver = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
)

mpf, r = CPV.compute_mpf_evidence(gen, solver)

# display(mpf)
display(r)

end # module