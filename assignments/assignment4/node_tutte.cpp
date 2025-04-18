#include <Eigen/Sparse>
#include <cmath>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void tutte_embedding(MyMesh& omesh)
{
    // TODO: Implement Tutte Embedding Algorithm.
    // Initialization
    int n_vertices = omesh.n_vertices();
    std::vector<int> ori2mat(n_vertices, 0);

    // Label the boundary vertecies
    for (const auto& halfedge_handle : omesh.halfedges())
        if (halfedge_handle.is_boundary()) {
            ori2mat[halfedge_handle.to().idx()] = -1;
            ori2mat[halfedge_handle.from().idx()] = -1;
        }

    // Construct a dictionary of internal points
    int n_internals = 0;
    for (int i = 0; i < n_vertices; i++)
        if (ori2mat[i] != -1)
            ori2mat[i] = n_internals++;

    Eigen::SparseMatrix<double> A(n_internals, n_internals);
    Eigen::VectorXd bx(n_internals);
    Eigen::VectorXd by(n_internals);
    Eigen::VectorXd bz(n_internals);

    // Construct coefficient matrix and vector
    for (const auto& vertex_handle : omesh.vertices()) {
        int mat_idx = ori2mat[vertex_handle.idx()];
        if (mat_idx == -1)
            continue;
        bx(mat_idx) = 0;
        by(mat_idx) = 0;
        bz(mat_idx) = 0;

        int Aii = 0;
        for (const auto& halfedge_handle : vertex_handle.outgoing_halfedges()) {
            const auto& v1 = halfedge_handle.to();
            int mat_idx1 = ori2mat[v1.idx()];
            Aii++;
            if (mat_idx1 == -1) {
                // Boundary points
                bx(mat_idx) += omesh.point(v1)[0];
                by(mat_idx) += omesh.point(v1)[1];
                bz(mat_idx) += omesh.point(v1)[2];
            }
            else
                // Internal points
                A.coeffRef(mat_idx, mat_idx1) = -1;
        }
        A.coeffRef(mat_idx, mat_idx) = Aii;
    }

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        solver;
    solver.compute(A);
    Eigen::VectorXd ux = bx;
    ux = solver.solve(ux);
    Eigen::VectorXd uy = by;
    uy = solver.solve(uy);
    Eigen::VectorXd uz = bz;
    uz = solver.solve(uz);

    // Update new positions
    for (const auto& vertex_handle : omesh.vertices()) {
        int idx = ori2mat[vertex_handle.idx()];
        if (idx != -1) {
            omesh.point(vertex_handle)[0] = ux(idx);
            omesh.point(vertex_handle)[1] = uy(idx);
            omesh.point(vertex_handle)[2] = uz(idx);
        }
    }
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(tutte)
{
    // Function content omitted
    b.add_input<Geometry>("Input");

    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(tutte)
{
    // Function content omitted

    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        std::cerr << "Tutte Parameterization: Need Geometry Input."
                  << std::endl;
        return false;
    }

    auto mesh = input.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    for (int i = 0; i < vertices.size(); i++) {
        omesh.add_vertex(
            OpenMesh::Vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
    }

    // Add faces
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    omesh.request_vertex_normals();
    omesh.request_face_normals();
    omesh.update_normals();

    // Perform Tutte Embedding
    tutte_embedding(omesh);

    // Convert back to Geometry
    pxr::VtArray<pxr::GfVec3f> tutte_vertices;
    for (const auto& v : omesh.vertices()) {
        const auto& p = omesh.point(v);
        tutte_vertices.push_back(pxr::GfVec3f(p[0], p[1], p[2]));
    }
    pxr::VtArray<int> tutte_faceVertexIndices;
    pxr::VtArray<int> tutte_faceVertexCounts;
    for (const auto& f : omesh.faces()) {
        size_t count = 0;
        for (const auto& vf : f.vertices()) {
            tutte_faceVertexIndices.push_back(vf.idx());
            count += 1;
        }
        tutte_faceVertexCounts.push_back(count);
    }

    Geometry tutte_geometry;
    auto tutte_mesh = std::make_shared<MeshComponent>(&tutte_geometry);

    tutte_mesh->set_vertices(tutte_vertices);
    tutte_mesh->set_face_vertex_indices(tutte_faceVertexIndices);
    tutte_mesh->set_face_vertex_counts(tutte_faceVertexCounts);
    tutte_geometry.attach_component(tutte_mesh);
    // Set the output of the nodes
    params.set_output("Output", tutte_geometry);
    return true;
}

NODE_DECLARATION_UI(tutte);
NODE_DEF_CLOSE_SCOPE
