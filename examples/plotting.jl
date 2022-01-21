using LazySets
using CDDLib

function retrieve_vertices(c_list)
    M = length(c_list)
    Ahrep = zeros(M, 2)
    bhrep = ones(M)
    for j = 1:M
        Ahrep[j, :] = c_list[j]
    end
    poly = HPolytope(Ahrep, bhrep)
    verts = vertices_list(poly, backend=CDDLib.Library())
    return convex_hull(verts)
end

norm_poly(x, c_list) = maximum(c -> c'*x, c_list)