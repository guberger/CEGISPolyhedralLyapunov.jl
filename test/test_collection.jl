using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGPolyhedralLyapunov.jl")
else
    using CEGPolyhedralLyapunov
end
CPL = CEGPolyhedralLyapunov

coll1 = CPL.linked_collection(String, [])
vec1 = collect(coll1)
np = 100
coll2 = CPL.linked_collection(String, (string(i) for i = 1:np))

@testset "coll" begin
    @test eltype(coll1) == String
    @test length(coll1) == 0
    @test vec1 == String[]
    @test eltype(coll2) == String
    @test length(coll2) == np
    for i = 1:np
        flag = false
        for x in coll2
            flag = flag || x == string(i)
        end
        @test flag
    end
end

println("\nfinished-----------------------------------------------------------")