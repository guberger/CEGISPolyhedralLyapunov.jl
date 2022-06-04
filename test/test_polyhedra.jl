using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISPolyhedralVerification.jl")
else
    using CEGISPolyhedralVerification
end
CPV = CEGISPolyhedralVerification

cone = CPV.Cone()
CPV.add_supp!(cone, [1.0, 1.0])

@testset "cone" begin
    @test [1, -2] ∈ cone
end

poly = CPV.Polyhedron()
CPV.add_halfspace!(poly, [1.0, 1.0], 1.0)

@testset "polyhedron" begin
    @test [1, -2.1] ∈ poly
end

nothing