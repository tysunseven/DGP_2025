#include <pxr/base/vt/array.h>

#include <cstdint>
#include <vector>
#include <unordered_set>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

typedef enum { kEdgeBased, kVertexBased } FaceNeighborType;

void getFaceArea(MyMesh &mesh, std::vector<float> &area)
{
    area.resize(mesh.n_faces());
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         ++f_it) {
        MyMesh::FaceHandle fh = *f_it;
        MyMesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(fh);
        MyMesh::Point p0 = mesh.point(*fv_it);
        ++fv_it;
        MyMesh::Point p1 = mesh.point(*fv_it);
        ++fv_it;
        MyMesh::Point p2 = mesh.point(*fv_it);
        MyMesh::Point v1 = p1 - p0;
        MyMesh::Point v2 = p2 - p0;
        float a = (v1 % v2).norm() / 2.0f;
        area[fh.idx()] = a;
    }
}

void getFaceCentroid(MyMesh &mesh, std::vector<MyMesh::Point> &centroid)
{
    centroid.resize(mesh.n_faces());
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         ++f_it) {
        MyMesh::FaceHandle fh = *f_it;
        MyMesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(fh);
        MyMesh::Point p0 = mesh.point(*fv_it);
        ++fv_it;
        MyMesh::Point p1 = mesh.point(*fv_it);
        ++fv_it;
        MyMesh::Point p2 = mesh.point(*fv_it);
        centroid[fh.idx()] = (p0 + p1 + p2) / 3.0f;
    }
}

void getFaceNormal(MyMesh &mesh, std::vector<MyMesh::Normal> &normals)
{
    mesh.request_face_normals();
    mesh.update_face_normals();
    normals.resize(mesh.n_faces());
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         ++f_it) {
        normals[f_it->idx()] = mesh.normal(*f_it);
    }
}

void getFaceNeighbor(
    MyMesh &mesh,
    MyMesh::FaceHandle fh,
    std::vector<MyMesh::FaceHandle> &face_neighbor)
{
    face_neighbor.clear();
    std::unordered_set<int> visited;
    visited.insert(fh.idx());
    for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(fh); fv_it.is_valid();
         ++fv_it) {
        MyMesh::VertexHandle vh = *fv_it;
        for (MyMesh::VertexFaceIter vf_it = mesh.vf_begin(vh); vf_it.is_valid();
             ++vf_it) {
            MyMesh::FaceHandle neighbor_fh = *vf_it;
            if (neighbor_fh != fh && !visited.count(neighbor_fh.idx())) {
                face_neighbor.push_back(neighbor_fh);
                visited.insert(neighbor_fh.idx());
            }
        }
    }
}

void getAllFaceNeighbor(
    MyMesh &mesh,
    std::vector<std::vector<MyMesh::FaceHandle>> &all_face_neighbor,
    bool include_central_face)
{
    all_face_neighbor.resize(mesh.n_faces());
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         ++f_it) {
        MyMesh::FaceHandle fh = *f_it;
        std::vector<MyMesh::FaceHandle> neighbors;
        getFaceNeighbor(mesh, fh, neighbors);
        if (include_central_face) {
            neighbors.push_back(fh);
        }
        all_face_neighbor[fh.idx()] = neighbors;
    }
}

void updateVertexPosition(
    MyMesh &mesh,
    std::vector<MyMesh::Normal> &filtered_normals,
    int iteration_number,
    bool fixed_boundary)
{
    std::vector<MyMesh::Point> new_points(mesh.n_vertices());
    std::vector<MyMesh::Point> centroid;

    for (int iter = 0; iter < iteration_number; iter++) {
        getFaceCentroid(mesh, centroid);
        for (MyMesh::VertexIter v_it = mesh.vertices_begin();
             v_it != mesh.vertices_end();
             ++v_it) {
            MyMesh::VertexHandle vh = *v_it;
            if (fixed_boundary && mesh.is_boundary(vh)) {
                new_points[vh.idx()] = mesh.point(vh);
                continue;
            }
            MyMesh::Point sum(0.0, 0.0, 0.0);
            int count = 0;
            for (MyMesh::VertexFaceIter vf_it = mesh.vf_begin(vh);
                 vf_it.is_valid();
                 ++vf_it) {
                MyMesh::FaceHandle fh = *vf_it;
                int idx = fh.idx();
                MyMesh::Point c = centroid[idx];
                MyMesh::Normal n = filtered_normals[idx];
                MyMesh::Point p = mesh.point(vh);
                float d = (p - c).dot(n);
                MyMesh::Point proj = p - d * n;
                sum += proj;
                count++;
            }
            if (count > 0) {
                new_points[vh.idx()] = sum / float(count);
            }
            else {
                new_points[vh.idx()] = mesh.point(vh);
            }
        }
        for (MyMesh::VertexIter v_it = mesh.vertices_begin();
             v_it != mesh.vertices_end();
             ++v_it) {
            mesh.set_point(*v_it, new_points[v_it->idx()]);
        }
    }
}

float getSigmaC(
    MyMesh &mesh,
    std::vector<MyMesh::Point> &face_centroid,
    float multiple_sigma_c)
{
    float sigma_c = 0.0;
    float num = 0.0;
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         f_it++) {
        MyMesh::Point ci = face_centroid[f_it->idx()];
        for (MyMesh::FaceFaceIter ff_it = mesh.ff_iter(*f_it); ff_it.is_valid();
             ff_it++) {
            MyMesh::Point cj = face_centroid[ff_it->idx()];
            sigma_c += (ci - cj).length();
            num++;
        }
    }
    sigma_c *= multiple_sigma_c / num;

    return sigma_c;
}

void update_filtered_normals_local_scheme(
    MyMesh &mesh,
    std::vector<MyMesh::Normal> &filtered_normals,
    float multiple_sigma_c,
    int normal_iteration_number,
    float sigma_s)
{
    filtered_normals.resize(mesh.n_faces());
    std::vector<std::vector<MyMesh::FaceHandle>> all_face_neighbor;
    getAllFaceNeighbor(mesh, all_face_neighbor, false);
    std::vector<MyMesh::Normal> previous_normals;
    getFaceNormal(mesh, previous_normals);
    std::vector<float> face_area;
    getFaceArea(mesh, face_area);
    std::vector<MyMesh::Point> face_centroid;
    getFaceCentroid(mesh, face_centroid);
    float sigma_c = getSigmaC(mesh, face_centroid, multiple_sigma_c);

    for (int iter = 0; iter < normal_iteration_number; iter++) {
        for (size_t i = 0; i < all_face_neighbor.size(); ++i) {
            MyMesh::Point ci = face_centroid[i];
            MyMesh::Normal ni = previous_normals[i];
            float total_weight = 0.0f;
            MyMesh::Normal sum(0.0f, 0.0f, 0.0f);
            for (const auto &fh_j : all_face_neighbor[i]) {
                int j = fh_j.idx();
                MyMesh::Point cj = face_centroid[j];
                MyMesh::Normal nj = previous_normals[j];
                float dist = (ci - cj).length();
                float wc = exp(-(dist * dist) / (2 * sigma_c * sigma_c));
                float wn = (ni - nj).length();
                float ws = exp(-(wn * wn) / (2 * sigma_s * sigma_s));
                float weight = face_area[j] * wc * ws;
                sum += weight * nj;
                total_weight += weight;
            }
            if (total_weight > 0) {
                sum /= total_weight;
                sum.normalize();
                filtered_normals[i] = sum;
            }
            else {
                filtered_normals[i] = ni;
            }
        }
        previous_normals = filtered_normals;
    }
}


void bilateral_normal_filtering(
    MyMesh &mesh,
    float sigma_s,
    int normal_iteration_number,
    float multiple_sigma_c)
{
    std::vector<MyMesh::Normal> filtered_normals;
    update_filtered_normals_local_scheme(
        mesh,
        filtered_normals,
        multiple_sigma_c,
        normal_iteration_number,
        sigma_s);

    updateVertexPosition(mesh, filtered_normals, normal_iteration_number, true);
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mesh_smoothing)
{
    b.add_input<Geometry>("Mesh");
    b.add_input<float>("Sigma_s").default_val(0.1).min(0).max(1);
    b.add_input<int>("Iterations").default_val(1).min(0).max(30);
    b.add_input<float>("Multiple Sigma C").default_val(1.0).min(0).max(10);

    b.add_output<Geometry>("Smoothed Mesh");
}

NODE_EXECUTION_FUNCTION(mesh_smoothing)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
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

    // Perform bilateral normal filtering
    float sigma_s = params.get_input<float>("Sigma_s");
    int iterations = params.get_input<int>("Iterations");
    float multiple_sigma_c = params.get_input<float>("Multiple Sigma C");

    bilateral_normal_filtering(omesh, sigma_s, iterations, multiple_sigma_c);

    // Convert back to Geometry
    pxr::VtArray<pxr::GfVec3f> smoothed_vertices;
    for (const auto &v : omesh.vertices()) {
        const auto &p = omesh.point(v);
        smoothed_vertices.push_back(pxr::GfVec3f(p[0], p[1], p[2]));
    }
    pxr::VtArray<int> smoothed_faceVertexIndices;
    pxr::VtArray<int> smoothed_faceVertexCounts;
    for (const auto &f : omesh.faces()) {
        size_t count = 0;
        for (const auto &vf : f.vertices()) {
            smoothed_faceVertexIndices.push_back(vf.idx());
            count += 1;
        }
        smoothed_faceVertexCounts.push_back(count);
    }

    Geometry smoothed_geometry;
    auto smoothed_mesh = std::make_shared<MeshComponent>(&smoothed_geometry);
    smoothed_mesh->set_vertices(smoothed_vertices);
    smoothed_mesh->set_face_vertex_indices(smoothed_faceVertexIndices);
    smoothed_mesh->set_face_vertex_counts(smoothed_faceVertexCounts);
    smoothed_geometry.attach_component(smoothed_mesh);
    params.set_output("Smoothed Mesh", smoothed_geometry);

    return true;
}

NODE_DECLARATION_UI(mesh_smoothing);

NODE_DEF_CLOSE_SCOPE
