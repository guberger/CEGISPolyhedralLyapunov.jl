using LinearAlgebra
using PyCall
const spatial = pyimport_conda("scipy.spatial", "scipy")
using .CEGISPolyhedralLyapunov.Polyhedra: Polyhedron

function compute_vertices_2d(p::Polyhedron)
    A = zeros(length(p.halfspaces), 3)
    for (i, h) in enumerate(p.halfspaces)
        for k = 1:2
            A[i, k] = h.a[k]
        end
        A[i, 3] = h.β
    end
    x = zeros(2)
    hs = spatial.HalfspaceIntersection(A, x)
    points = [collect(point_) for point_ in eachrow(hs.intersections)]
    ch = spatial.ConvexHull(points)
    return [ch.points[i + 1, :] for i in ch.vertices]
end

# function retrieve_simplices_3d(p::Polyhedron)
#     M = length(p::Polyhedron)
#     Ahrep = zeros(M, 3)
#     bhrep = ones(M)
#     for j = 1:M
#         Ahrep[j, :] = p[j]
#     end
#     poly = HPolytope(Ahrep, bhrep)
#     points = vertices_list(poly, backend=CDDLib.Library())
#     M = length(points)
#     Points = zeros(M, 3)
#     for j = 1:M
#         Points[j, :] = points[j]
#     end
#     hull = spatial.ConvexHull(Points)
#     simplices = map(j ->
#         map(k -> hull.points[hull.simplices[j, k] + 1, :], 1:3),
#         1:size(hull.simplices, 1))
#     return simplices
# end

# norm_poly(x, p) = maximum(c -> c'*x, p)