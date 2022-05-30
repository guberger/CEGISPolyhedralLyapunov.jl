using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralLyapunov.jl")
else
    using CEGISPolyhedralLyapunov
end
CPLP = CEGISPolyhedralLyapunov.Polyhedra

cone = CPLP.Cone()
CPLP.add_supp!(cone, [1.0, 1.0])

@testset "cone" begin
    @test [1, -2] ∈ cone
end

poly = CPLP.Polyhedron()
CPLP.add_halfspace!(poly, [1.0, 1.0], 1.0)

@testset "polyhedron" begin
    @test [1, -2.1] ∈ poly
end

nothing