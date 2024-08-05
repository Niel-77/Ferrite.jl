using Ferrite, FerriteGmsh, SparseArrays

using Downloads: download
logo_mesh = "logo.geo"
asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)

FerriteGmsh.Gmsh.initialize() # hide
FerriteGmsh.Gmsh.gmsh.option.set_number("General.Verbosity", 2) #hide
grid = togrid(logo_mesh);
FerriteGmsh.Gmsh.finalize();

addfacetset!(grid, "top",    x -> x[2] ≈ 1.0) # faces for which x[2] ≈ 1.0 for all nodes
addfacetset!(grid, "left",   x -> abs(x[1]) < 1e-6)
addfacetset!(grid, "bottom", x -> abs(x[2]) < 1e-6);

dim = 2
order = 1 # linear interpolation
ip = Lagrange{RefTriangle, order}()^dim # vector valued interpolation
qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
cellvalues = CellValues(qr, ip)
qr_face = FacetQuadratureRule{RefTriangle}(1)
facetvalues = FacetValues(qr_face, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> 0.0, 1))
close!(ch);

traction(x) = Vec(0.0, 20e3 * x[1])

function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for face in FacetIterator(dh, facetset)
        reinit!(facetvalues, face)
        fill!(fe_ext, 0.0)
        for qp in 1:getnquadpoints(facetvalues)
            X = spatial_coordinate(facetvalues, qp, getcoordinates(face))
            tₚ = prescribed_traction(X)
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                Nᵢ = shape_value(facetvalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        assemble!(f_ext, celldofs(face), fe_ext)
    end
    return f_ext
end

Emod = 200e3 # Young's modulus [MPa]
ν = 0.3      # Poisson's ratio [-]

Gmod = Emod / (2(1 + ν))  # Shear modulus
Kmod = Emod / (3(1 - 2ν)) # Bulk modulus
E4 = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2,2}));

function assemble_cell!(ke, cellvalues, ∂σ∂ε)
    fill!(ke, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)# shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∂σ∂ε ⊡ ∇ˢʸᵐNⱼ) ⊡ ∇Nᵢ * dΩ
            end
        end
    end
    return ke
end

function assemble_global!(K, dh, cellvalues, ∂σ∂ε)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cellvalues, cell)
        # Compute element contribution
        assemble_cell!(ke, cellvalues, ∂σ∂ε)
        # Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

K = allocate_matrix(dh)
assemble_global!(K, dh, cellvalues, E4);

f_ext = zeros(ndofs(dh))
assemble_external_forces!(f_ext, dh, getfacetset(grid, "top"), facetvalues, x->Vec(0.0, 20e3*x[1]));

apply!(K, f_ext, ch)
u = K \ f_ext;

color_data = zeros(Int, getncells(grid))
colors = Dict(
    "1" => 1, "5" => 1, # purple
    "2" => 2, "3" => 2, # red
    "4" => 3,           # blue
    "6" => 4            # green
    )
for (key, color) in colors
    for i in getcellset(grid, key)
        color_data[i] = color
    end
end

VTKGridFile("linear_elasticity", dh) do vtk
    write_solution(vtk, dh, u)
    write_cell_data(vtk, color_data, "colors")
end

using Test                              #hide
if Sys.islinux() # gmsh not os stable   #hide
    @test norm(u) ≈ 0.31742879147646924 #hide
end                                     #hide
nothing                                 #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
