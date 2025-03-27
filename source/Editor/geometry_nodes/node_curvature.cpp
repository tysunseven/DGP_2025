#include <pxr/base/vt/array.h>

#include <OpenMesh/Core/Geometry/VectorT.hh>
#include <cmath>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;


float compute_angle(
    const OpenMesh::Vec3f& v,
    const OpenMesh::Vec3f& a,
    const OpenMesh::Vec3f& b)
{
    OpenMesh::Vec3f vec1 = a - v;
    OpenMesh::Vec3f vec2 = b - v;
    float dot = vec1.dot(vec2);
    float len1 = vec1.length();
    float len2 = vec2.length();
    return std::acos(dot / (len1 * len2));
}


float compute_cotangent(
    const OpenMesh::Vec3f& v,
    const OpenMesh::Vec3f& a,
    const OpenMesh::Vec3f& b)
{
    OpenMesh::Vec3f vec1 = a - v;
    OpenMesh::Vec3f vec2 = b - v;
    float dot = vec1.dot(vec2);
    float cross = vec1.cross(vec2).length();
    return cross != 0 ? dot / cross : 0.0f;
}

void compute_mean_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& mean_curvature)
{
    mean_curvature.clear();
    mean_curvature.reserve(omesh.n_vertices());

    for (MyMesh::ConstVertexIter v_it = omesh.vertices_begin();
         v_it != omesh.vertices_end();
         ++v_it) {
        MyMesh::VertexHandle vh = *v_it;
        OpenMesh::Vec3f v_pos = omesh.point(vh);
        OpenMesh::Vec3f sum(0.0f, 0.0f, 0.0f);
        double area = 0.0;


        for (MyMesh::ConstVertexFaceIter vf_it = omesh.cvf_iter(vh);
             vf_it.is_valid();
             ++vf_it) {
            MyMesh::FaceHandle fh = *vf_it;

            MyMesh::ConstFaceVertexIter fv_it = omesh.cfv_iter(fh);
            std::array<MyMesh::VertexHandle, 3> face_vertices = { *fv_it,
                                                                  *(++fv_it),
                                                                  *(++fv_it) };


            OpenMesh::Vec3f p0 = omesh.point(face_vertices[0]);
            OpenMesh::Vec3f p1 = omesh.point(face_vertices[1]);
            OpenMesh::Vec3f p2 = omesh.point(face_vertices[2]);
            OpenMesh::Vec3f cross = (p1 - p0).cross(p2 - p0);
            area += cross.length() / 6.0; 
        }


        for (MyMesh::ConstVertexOHalfedgeIter voh_it = omesh.cvoh_iter(vh);
             voh_it.is_valid();
             ++voh_it) {
            MyMesh::HalfedgeHandle he = *voh_it;
            MyMesh::VertexHandle vj = omesh.to_vertex_handle(he);
            float cot_sum = 0.0f;


            if (omesh.face_handle(he).is_valid()) {
                MyMesh::VertexHandle vk =
                    omesh.to_vertex_handle(omesh.next_halfedge_handle(he));
                cot_sum +=
                    compute_cotangent(omesh.point(vk), v_pos, omesh.point(vj));
            }


            MyMesh::HalfedgeHandle opp_he = omesh.opposite_halfedge_handle(he);
            if (omesh.face_handle(opp_he).is_valid()) {
                MyMesh::VertexHandle vl =
                    omesh.to_vertex_handle(omesh.next_halfedge_handle(opp_he));
                cot_sum +=
                    compute_cotangent(omesh.point(vl), v_pos, omesh.point(vj));
            }


            OpenMesh::Vec3f diff = omesh.point(vj) - v_pos;
            sum += diff * cot_sum;
        }


        float H = area > 1e-6 ? (sum / (2.0f * area)).length() : 0.0f;
        mean_curvature.push_back(H);
    }
}

void compute_gaussian_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& gaussian_curvature)
{
    gaussian_curvature.clear();
    gaussian_curvature.reserve(omesh.n_vertices());

    for (MyMesh::ConstVertexIter v_it = omesh.vertices_begin();
         v_it != omesh.vertices_end();
         ++v_it) {
        MyMesh::VertexHandle vh = *v_it;
        double angle_sum = 0.0;
        double area = 0.0;


        for (MyMesh::ConstVertexFaceIter vf_it = omesh.cvf_iter(vh);
             vf_it.is_valid();
             ++vf_it) {
            MyMesh::FaceHandle fh = *vf_it;

            MyMesh::ConstFaceVertexIter fv_it = omesh.cfv_iter(fh);
            std::array<MyMesh::VertexHandle, 3> face_vertices = { *fv_it,
                                                                  *(++fv_it),
                                                                  *(++fv_it) };


            int idx = 0;
            while (face_vertices[idx] != vh)
                ++idx;


            OpenMesh::Vec3f p = omesh.point(face_vertices[idx]);
            OpenMesh::Vec3f p1 = omesh.point(face_vertices[(idx + 1) % 3]);
            OpenMesh::Vec3f p2 = omesh.point(face_vertices[(idx + 2) % 3]);
            angle_sum += compute_angle(p, p1, p2);


            OpenMesh::Vec3f cross = (p1 - p).cross(p2 - p);
            area += cross.length() / 6.0;
        }


        float K = area > 1e-6 ? (2 * M_PI - angle_sum) / area : 0.0f;
        gaussian_curvature.push_back(K);
    }
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mean_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Mean Curvature");
}

NODE_EXECUTION_FUNCTION(mean_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
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

    // Compute mean curvature
    pxr::VtArray<float> mean_curvature;
    mean_curvature.reserve(omesh.n_vertices());

    compute_mean_curvature(omesh, mean_curvature);

    params.set_output("Mean Curvature", mean_curvature);

    return true;
}

NODE_DECLARATION_UI(mean_curvature);

NODE_DECLARATION_FUNCTION(gaussian_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Gaussian Curvature");
}

NODE_EXECUTION_FUNCTION(gaussian_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
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

    // Compute Gaussian curvature
    pxr::VtArray<float> gaussian_curvature;
    gaussian_curvature.reserve(omesh.n_vertices());

    compute_gaussian_curvature(omesh, gaussian_curvature);

    params.set_output("Gaussian Curvature", gaussian_curvature);

    return true;
}

NODE_DECLARATION_UI(gaussian_curvature);

NODE_DEF_CLOSE_SCOPE
