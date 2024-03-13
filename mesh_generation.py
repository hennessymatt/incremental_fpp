from dolfin import *
from multiphenics import *
import gmsh
import pygmsh
import meshio
import numpy as np

def run_pygmsh(layers, bottom, right_all, top_all, left_all):
    """
    Creates a mesh for the problem using pygmsh (a Python implementation
    of gmsh).

    Outputs:
    -   The function saves a .msh file
    -   The function returns a dictionary with info about the
        numerical ids of the domains and boundaries

    """

    # preallocate
    top_points = (1 + layers) * [0]
    right_points = (1 + layers) * [0]
    left_points = (1 + layers) * [0]

    top_lines = (1 + layers) * [0]
    right_lines = (1 + layers) * [0]
    left_lines = (1 + layers) * [0]
    extra_line = (1 + layers) * [0]

    # count the physical domains and store their ids
    physical_counter = 1
    ids = {}

    # start gmsh
    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()

    res_bottom = 0.1

    # add points for the substrate
    top_points[0] = [
        model.add_point((pt[0], pt[1], 0), mesh_size = res_bottom) for pt in bottom
    ]

    # create line for substrate
    top_lines[0] = [
        model.add_line(top_points[0][i], top_points[0][i+1]) for i in range(len(top_points[0])-1)
    ]

    # create boundary for substrate
    model.synchronize()
    model.add_physical([e for e in top_lines[0]], "substrate")

    ids["substrate"] = physical_counter
    physical_counter += 1

    for n in range(1, layers + 1):

        right = right_all[n-1]
        top = top_all[n-1]
        left = left_all[n-1]

        # element size away from particle.
        # should calculate this based on dz for each layer
        res = res_bottom

        right_points[n] = [
            model.add_point((pt[0], pt[1], 0), mesh_size = res) for pt in right
        ]

        top_points[n] = [
            model.add_point((pt[0], pt[1], 0), mesh_size = res) for pt in top
        ]

        left_points[n] = [
            model.add_point((pt[0], pt[1], 0), mesh_size = res) for pt in left
        ]


        # create outline

        if n == 1:
            right_lines[n] = [model.add_line(top_points[n-1][-1], right_points[n][1])]
        else:
            right_lines[n] = [model.add_line(right_points[n-1][-1], right_points[n][1])]

        right_lines[n] += [
            model.add_line(right_points[n][i], right_points[n][i+1]) for i in range(1, len(right_points[n])-1)
        ]

        top_lines[n] = [model.add_line(right_points[n][-1], top_points[n][1])]
        top_lines[n] += [
            model.add_line(top_points[n][i], top_points[n][i+1]) for i in range(1, len(top_points[n])-1)
        ]

        left_lines[n] = [model.add_line(top_points[n][-1], left_points[n][1])]
        left_lines[n] += [
            model.add_line(left_points[n][i], left_points[n][i+1]) for i in range(1, len(left_points[n])-2)
        ]

        if n == 1:
            left_lines[n] += [model.add_line(left_points[n][-2], top_points[n-1][0])]
        else:
            left_lines[n] += [model.add_line(left_points[n][-2], top_points[n-1][-1])]




        if n == 1:
            lines = [e for e in top_lines[n-1]]
        else:
            lines = [-top_lines[n-1][-1-k] for k in range(len(top_lines[n-1]))]


        for l in [right_lines[n], top_lines[n], left_lines[n]]:
            lines += l


        # print(right_lines[n][0])
        # print(right_lines[n][-1])
        # print(top_lines[n][0])
        # print(top_lines[n][-1])
        # print(left_lines[n][0])
        # # print(left_lines[n][-1])
        # print(extra_line[n][0])



        # curve loop for fluid domain
        loop = model.add_curve_loop(lines)

        # # # create surfaces for domains
        surface = model.add_plane_surface(loop)

        model.synchronize()

        # create labelled domains and boundaries
        model.add_physical([e for e in right_lines[n]], f"right{n}")
        ids[f"right{n}"] = physical_counter
        physical_counter += 1

        model.add_physical([e for e in top_lines[n]], f"top{n}")
        ids[f"top{n}"] = physical_counter
        physical_counter += 1


        model.add_physical([e for e in left_lines[n]], f"left{n}")
        ids[f"left{n}"] = physical_counter
        physical_counter += 1

        model.add_physical([surface], f"layer{n}")
        ids[f"layer{n}"] = physical_counter
        physical_counter += 1

        print(f'finished layer {n}')


    mesh = geometry.generate_mesh(dim=2)
    gmsh.write("mesh/python_mesh.msh")
    gmsh.clear()
    geometry.__exit__()

    print('pygmsh finished successfully')

    return ids

def build_mesh(layers, bottom, right, top, left):
    """
    Main function for mesh generation.  Calls the other functions
    """

    # create the mesh in pygmsh and save as xdmf
    ids = run_pygmsh(layers, bottom, right, top, left)
    save_xdmf()

    # load in the .xdmf mesh
    mesh_file = XDMFFile(MPI.comm_world, "mesh/python_mesh.xdmf")
    mesh = Mesh()
    mesh_file.read(mesh)

    # create subdomains that separates fluid and solid domain
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile("mesh/python_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")

    subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # create bdry to separate different boundaries
    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile("mesh/python_facet_mesh.xdmf") as infile:
        infile.read(mvc, "name_to_read")

    bdry = cpp.mesh.MeshFunctionSizet(mesh, mvc)


    return mesh, subdomains, bdry, ids


def create_mesh(mesh, cell_type, prune_z=False):
    """
    Helper function used for converting a .msh mesh
    into a .xdmf mesh
    """
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
                           "name_to_read": [cell_data]})
    return out_mesh


def save_xdmf():
    """
    Loads a .msh file created using run_pygmsh and converts it
    into two xdmf files, one for the domains and another for
    the boundaries
    """

    # load .msh mesh from file
    mesh_from_file = meshio.read("mesh/python_mesh.msh")

    # find the boundaries in the mesh and save to a file
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("mesh/python_facet_mesh.xdmf", line_mesh)

    # find the domains in the mesh and save to a file
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh/python_mesh.xdmf", triangle_mesh)
