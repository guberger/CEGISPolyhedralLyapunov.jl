using LinearAlgebra
using PyCall
const spatial = pyimport_conda("scipy.spatial", "scipy")

function compute_vertices_2d(p, x)
    A = zeros(length(p.halfspaces), 3)
    for (i, h) in enumerate(p.halfspaces)
        for k = 1:2
            A[i, k] = h.a[k]
        end
        A[i, 3] = h.β
    end
    hs = spatial.HalfspaceIntersection(A, x)
    points = collect.(eachrow(hs.intersections))
    ch = spatial.ConvexHull(points)
    return [ch.points[i + 1, :] for i in ch.vertices]
end

function compute_simplices_3d(p, x)
    A = zeros(length(p.halfspaces), 4)
    for (i, h) in enumerate(p.halfspaces)
        for k = 1:3
            A[i, k] = h.a[k]
        end
        A[i, 4] = h.β
    end
    hs = spatial.HalfspaceIntersection(A, x)
    points = collect.(eachrow(hs.intersections))
    ch = spatial.ConvexHull(points)
    simplices = [
        [ch.points[k + 1, :] for k in simplex_]
        for simplex_ in eachrow(ch.simplices)
    ]
    return simplices
end