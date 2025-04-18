#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <iostream>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void tutte_embedding(MyMesh& omesh)
{
    // Separate internal and boundary vertices
    std::vector<MyMesh::VertexHandle> internal_vertices;
    std::vector<MyMesh::VertexHandle> boundary_vertices;

    for (MyMesh::VertexIter v_it = omesh.vertices_begin();
         v_it != omesh.vertices_end();
         ++v_it) {
        if (omesh.is_boundary(*v_it)) {
            boundary_vertices.push_back(*v_it);
        }
        else {
            internal_vertices.push_back(*v_it);
        }
    }

    if (internal_vertices.empty())
        return;

    // Create mapping from internal vertices to indices
    std::unordered_map<MyMesh::VertexHandle, int> vertex_to_index;
    for (int i = 0; i < internal_vertices.size(); ++i) {
        vertex_to_index[internal_vertices[i]] = i;
    }

    const int n = internal_vertices.size();
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> triplets;

    Eigen::VectorXd B_x(n), B_y(n), B_z(n);
    B_x.setZero();
    B_y.setZero();
    B_z.setZero();

    // Build Laplace matrix and RHS vectors
    for (int i = 0; i < internal_vertices.size(); ++i) {
        MyMesh::VertexHandle vh = internal_vertices[i];
        int degree = 0;
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

        for (MyMesh::VertexVertexIter vv_it = omesh.vv_begin(vh);
             vv_it.is_valid();
             ++vv_it) {
            MyMesh::VertexHandle neighbor_vh = *vv_it;
            if (omesh.is_boundary(neighbor_vh)) {
                // Accumulate boundary neighbor positions
                sum_x += omesh.point(neighbor_vh)[0];
                sum_y += omesh.point(neighbor_vh)[1];
                sum_z += omesh.point(neighbor_vh)[2];
                degree++;
            }
            else {
                // Add entry for internal neighbor
                auto it = vertex_to_index.find(neighbor_vh);
                if (it != vertex_to_index.end()) {
                    int j = it->second;
                    triplets.emplace_back(i, j, -1.0);
                    degree++;
                }
            }
        }

        // Diagonal entry with degree
        triplets.emplace_back(i, i, degree);

        // Set RHS values
        B_x[i] = sum_x;
        B_y[i] = sum_y;
        B_z[i] = sum_z;
    }

    A.setFromTriplets(triplets.begin(), triplets.end());

    // Solve linear systems for each coordinate
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success)
        return;

    Eigen::VectorXd X = solver.solve(B_x);
    Eigen::VectorXd Y = solver.solve(B_y);
    Eigen::VectorXd Z = solver.solve(B_z);

    // Update vertex positions
    for (int i = 0; i < internal_vertices.size(); ++i) {
        MyMesh::VertexHandle vh = internal_vertices[i];
        omesh.set_point(vh, MyMesh::Point(X[i], Y[i], Z[i]));
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
