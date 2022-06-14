using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralLyapunov.jl")
else
    using CEGISPolyhedralLyapunov
end
CPL = CEGISPolyhedralLyapunov

cone = CPL.Cone()
CPL.add_supp!(cone, [1.0, 1.0])

@testset "cone" begin
    @test [1, -2] ∈ cone
end

poly = CPL.Polyhedron()
CPL.add_halfspace!(poly, [1.0, 1.0], 1.0)

@testset "polyhedron" begin
    @test [1, -2.1] ∈ poly
end

nothing