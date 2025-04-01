#include <pxr/base/vt/array.h>

#include <cstdint>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void getFaceArea(MyMesh &mesh, std::vector<float> &area)
{
    area.resize(mesh.n_faces());

    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         f_it++) {
        std::vector<MyMesh::Point> point;
        point.resize(3);
        int index = 0;
        for (MyMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it);
             fv_it.is_valid();
             fv_it++) {
            point[index] = mesh.point(*fv_it);
            index++;
        }
        MyMesh::Point edge1 = point[1] - point[0];
        MyMesh::Point edge2 = point[1] - point[2];
        float S = 0.5 * (edge1 % edge2).length();
        area[(*f_it).idx()] = S;
    }
}

void getFaceCentroid(MyMesh &mesh, std::vector<MyMesh::Point> &centroid)
{
    centroid.resize(mesh.n_faces(), MyMesh::Point(0.0, 0.0, 0.0));
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         f_it++) {
        MyMesh::Point pt = mesh.calc_face_centroid(*f_it);
        centroid[(*f_it).idx()] = pt;
    }
}

void getFaceNormal(MyMesh &mesh, std::vector<MyMesh::Normal> &normals)
{
    mesh.request_face_normals();
    mesh.update_face_normals();

    normals.resize(mesh.n_faces());
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         f_it++) {
        MyMesh::Normal n = mesh.normal(*f_it);
        normals[f_it->idx()] = n;
    }
}

void getFaceNeighbor(
    MyMesh &mesh,
    MyMesh::FaceHandle fh,
    std::vector<MyMesh::FaceHandle> &face_neighbor)
{
    face_neighbor.clear();

    std::set<int> neighbor_face_index;
    neighbor_face_index.clear();

    for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(fh); fv_it.is_valid();
         fv_it++) {
        for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*fv_it);
             vf_it.is_valid();
             vf_it++) {
            if ((*vf_it) != fh)
                neighbor_face_index.insert(vf_it->idx());
        }
    }

    for (std::set<int>::iterator iter = neighbor_face_index.begin();
         iter != neighbor_face_index.end();
         ++iter) {
        face_neighbor.push_back(MyMesh::FaceHandle(*iter));
    }
}

void getAllFaceNeighbor(
    MyMesh &mesh,
    std::vector<std::vector<MyMesh::FaceHandle>> &all_face_neighbor,
    bool include_central_face)
{
    all_face_neighbor.resize(mesh.n_faces());
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
         f_it++) {
        std::vector<MyMesh::FaceHandle> face_neighbor;
        getFaceNeighbor(mesh, *f_it, face_neighbor);
        if (include_central_face)
            face_neighbor.push_back(*f_it);
        all_face_neighbor[f_it->idx()] = face_neighbor;
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
             v_it++) {
            MyMesh::Point p = mesh.point(*v_it);
            if (fixed_boundary && mesh.is_boundary(*v_it)) {
                new_points.at(v_it->idx()) = p;
            }
            else {
                float face_num = 0.0;
                MyMesh::Point temp_point(0.0, 0.0, 0.0);
                for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it);
                     vf_it.is_valid();
                     vf_it++) {
                    MyMesh::Normal temp_normal = filtered_normals[vf_it->idx()];
                    MyMesh::Point temp_centroid = centroid[vf_it->idx()];
                    temp_point +=
                        temp_normal * (temp_normal | (temp_centroid - p));
                    face_num++;
                }
                p += temp_point / face_num;

                new_points.at(v_it->idx()) = p;
            }
        }

        for (MyMesh::VertexIter v_it = mesh.vertices_begin();
             v_it != mesh.vertices_end();
             v_it++)
            mesh.set_point(*v_it, new_points[v_it->idx()]);
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
        for (MyMesh::FaceIter f_it = mesh.faces_begin();
             f_it != mesh.faces_end();
             f_it++) {
            int index_i = f_it->idx();
            MyMesh::Normal ni = previous_normals[index_i];
            MyMesh::Point ci = face_centroid[index_i];
            std::vector<MyMesh::FaceHandle> face_neighbor =
                all_face_neighbor[index_i];
            int size = (int)face_neighbor.size();
            MyMesh::Normal temp_normal(0.0, 0.0, 0.0);
            float weight_sum = 0.0;
            for (int i = 0; i < size; i++) {
                int index_j = face_neighbor[i].idx();
                MyMesh::Normal nj = previous_normals[index_j];
                MyMesh::Point cj = face_centroid[index_j];

                float spatial_distance = (ci - cj).length();
                float spatial_weight = std::exp(
                    -0.5 * spatial_distance * spatial_distance /
                    (sigma_c * sigma_c));
                float range_distance = (ni - nj).length();
                float range_weight = std::exp(
                    -0.5 * range_distance * range_distance /
                    (sigma_s * sigma_s));

                float weight =
                    face_area[index_j] * spatial_weight * range_weight;
                weight_sum += weight;
                temp_normal += nj * weight;
            }
            temp_normal /= weight_sum;
            temp_normal.normalize();
            filtered_normals[index_i] = temp_normal;
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
