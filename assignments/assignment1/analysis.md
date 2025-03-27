# 作业一分析

## OpenMesh半边数据结构

### 1. 半边结构概述

半边数据结构（Half-edge Data Structure）是计算机图形学中表示多边形网格的一种高效数据结构。其核心思想是将每条几何边分解为两条有方向的半边（half-edge），每条半边属于一个面并指向特定方向。

半边结构的主要特点：
- 支持高效的网格遍历操作
- 便于修改网格拓扑结构
- 能够快速获取相邻元素

![Half-edge](../../images/analysis_1_1.png)

### 2. OpenMesh中的半边结构实现

OpenMesh库使用半边结构实现了网格的存储与操作。根据OpenMesh官方文档，半边结构由以下核心元素组成：

1. **顶点（Vertex）**：
   - 存储坐标信息
   - 引用一个从该顶点出发的半边（outgoing halfedge）

2. **面（Face）**：
   - 引用一个围绕它的半边

3. **半边（Halfedge）**：
   - 引用它指向的目标顶点
   - 引用它所属的面
   - 引用同一面内的下一个半边（按逆时针方向）
   - 引用它的对边（opposite halfedge）
   - 可选：引用同一面内的前一个半边

半边结构使得我们可以轻松地：
- 围绕一个面循环，枚举其所有顶点、半边或相邻面
- 从顶点的半边开始，迭代到其前一个半边的对边，从而围绕该顶点循环，获取其一环邻域的邻居、进入/外出半边或相邻面

### 3. 常用遍历操作

OpenMesh提供了多种遍历操作，便于在网格上进行计算：

以下仅进行简单介绍，具体请参考[**官方文档**](https://www.graphics.rwth-aachen.de/media/openmesh_static/Documentations/OpenMesh-10.0-Documentation/a06153.html)

迭代器（Iterators）：

```cpp
MyMesh mesh;
 
// 迭代所有顶点
for (MyMesh::VertexIter v_it=mesh.vertices_begin(); v_it!=mesh.vertices_end(); ++v_it) 
   ...; // do something with *v_it, v_it->, or *v_it
 
// 迭代所有半边
for (MyMesh::HalfedgeIter h_it=mesh.halfedges_begin(); h_it!=mesh.halfedges_end(); ++h_it) 
   ...; // do something with *h_it, h_it->, or *h_it
 
// 迭代所有边
for (MyMesh::EdgeIter e_it=mesh.edges_begin(); e_it!=mesh.edges_end(); ++e_it) 
   ...; // do something with *e_it, e_it->, or *e_it
 
// 迭代所有面
for (MyMesh::FaceIter f_it=mesh.faces_begin(); f_it!=mesh.faces_end(); ++f_it) 
   ...; // do something with *f_it, f_it->, or *f_it
```

此外还提供了`const`版本：

- ConstVertexIter
- ConstHalfedgeIter
- ConstEdgeIter
- ConstFaceIter

如果在迭代时删除了元素，需要使用跳过迭代器（Skipping Iterators）：

- vertices_sbegin()
- edges_sbegin()
- halfedges_sbegin()
- faces_sbegin()

循环器（Circulators）：

```cpp
/**************************************************
 * Vertex circulators
 **************************************************/
 
// 顶点-顶点循环器
VertexVertexIter OpenMesh::PolyConnectivity::vv_iter (VertexHandle _vh);
 
// 顶点-入射半边循环器
VertexIHalfedgeIter OpenMesh::PolyConnectivity::vih_iter (VertexHandle _vh);
 
// 顶点-出射半边循环器
VertexOHalfedgeIter OpenMesh::PolyConnectivity::voh_iter (VertexHandle _vh);
 
// 顶点-边循环器
VertexEdgeIter OpenMesh::PolyConnectivity::ve_iter (VertexHandle _vh);
 
// 顶点-面循环器
VertexFaceIter OpenMesh::PolyConnectivity::vf_iter (VertexHandle _vh);
 
/**************************************************
 * Face circulators
 **************************************************/
 
// 面-顶点循环器
FaceVertexIter OpenMesh::PolyConnectivity::fv_iter (FaceHandle _fh);
 
// 面-半边循环器
FaceHalfedgeIter OpenMesh::PolyConnectivity::fh_iter (FaceHandle _fh);
 
// 面-边循环器
FaceEdgeIter OpenMesh::PolyConnectivity::fe_iter (FaceHandle _fh);
 
// 面-面循环器
FaceFaceIter OpenMesh::PolyConnectivity::ff_iter (FaceHandle _fh);
 
/**************************************************
 * Edge circulators
 **************************************************/
 
// 边-顶点循环器
EdgeVertexIter OpenMesh::PolyConnectivity::ev_iter (EdgeHandle _eh);
 
// 边-半边循环器
EdgeHalfedgeIter OpenMesh::PolyConnectivity::eh_iter (EdgeHandle _eh);
 
// 边-面循环器
EdgeFaceIter OpenMesh::PolyConnectivity::ef_iter (EdgeHandle _eh);
```

以上的循环器是不保证顺序，如果需要按照特定顺序遍历，需要在`iter`前加上`ccw`（逆时针）或`cw`（顺时针）：

```cpp
VertexVertexIter vvit = mesh.vv_iter(some_vertex_handle);          // fastest (clock or counterclockwise)
VertexVertexCWIter vvcwit = mesh.vv_cwiter(some_vertex_handle);    // clockwise
VertexVertexCCWIter vvccwit = mesh.vv_ccwiter(some_vertex_handle); // counter-clockwise
```

同样，循环器也有`const`版本，只需在`iter`前加上`c`：

```cpp
ConstVertexVertexIter cvvit = mesh.cvv_iter(some_vertex_handle);
```


### 4. 在网格最短路径计算中的应用

## Dijkstra算法

### 1. 算法原理

Dijkstra算法是一种解决带权有向图上单源最短路径问题的贪心算法。其核心思想是从源点开始，逐步扩展到整个图，每次选择当前已知最短路径的顶点进行扩展。

核心方法 Relaxation：

- 记起始顶点为`s`
- 当前已经找到了从`s`到`u`的最短路径`dist[u]`
- 对于`u`的每个邻接顶点`v`，当前已知的最短路径为`dist[v]`，可能有值，也可能为无穷大
- 而通过`u`到达`v`的路径长度为`dist[u] + length(u, v)`
- 如果`dist[u] + length(u, v) < dist[v]`，则更新`dist[v]`为`dist[u] + length(u, v)`

![Relaxation](../../images/analysis_1_2.png)


### 2. 算法步骤

1. 初始化：
   - 设置起点距离为0，其他顶点距离为无穷大
   - 所有顶点标记为未访问
   - 起点加入优先队列

2. 主循环：
   - 从优先队列中取出距离最小的未访问顶点`u`
   - 标记`u`为已访问
   - 对`u`的每个邻接顶点`v`：
     - 如果通过`u`到达`v`的路径比已知路径更短，更新`v`的距离和前驱
     - 将更新后的`v`加入优先队列

3. 路径重建：
   - 从终点开始，通过前驱指针回溯到起点，得到最短路径

