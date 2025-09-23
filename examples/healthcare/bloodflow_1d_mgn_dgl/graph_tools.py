# ignore_header_test
# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch as th
import numpy as np
import scipy
import dgl


def generate_types(bif_id, indices):
    """
    Generate node types.

    Generate one-hot representation of node type: 0 = branch node, 1 = junction
    node, 2 = inlet, 3 = outlet.

    Arguments:
        bif_id: numpy array containing node types as read from .vtp
        indices: dictionary containing inlet and outlets indices
    Returns:
        One-hot representation of the node type
        Inlet mask, i.e., array containing 1 at inlet index and 0 elsewhere
        Outlet maks, i.e., array containing 1 at outlet indices and 0 elsewhere

    """
    types = []
    inlet_mask = []
    outlet_mask = []
    for i, id in enumerate(bif_id):
        if id == -1:
            cur_type = 0
        else:
            cur_type = 1
        if i in indices["inlet"]:
            cur_type = 2
        elif i in indices["outlets"]:
            cur_type = 3
        types.append(cur_type)
        if cur_type == 2:
            inlet_mask.append(True)
        else:
            inlet_mask.append(False)
        if cur_type == 3:
            outlet_mask.append(True)
        else:
            outlet_mask.append(False)
    types = th.nn.functional.one_hot(th.tensor(types), num_classes=4)
    return types, inlet_mask, outlet_mask


def generate_edge_features(points, edges1, edges2):
    """
    Generate edge features.

    Returns a n x 3 array where row i contains (x_j - x_i) / |x_j - x_i|
    (node coordinates) and n is the number of nodes.
    Here, j and i are the node indices contained in row i of the edges1 and
    edges2 inputs. The second output is |x_j - x_i|.

    Arguments:
        points: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
    Returns:
        n x 3 numpy array containing x_j - x_i
        n dimensional numpy array containing |x_j - x_i|

    """
    rel_position = []
    rel_position_norm = []
    nedges = len(edges1)
    for i in range(nedges):
        diff = points[edges2[i], :] - points[edges1[i], :]
        ndiff = np.linalg.norm(diff)
        rel_position.append(diff / ndiff)
        rel_position_norm.append(ndiff)
    return np.array(rel_position), rel_position_norm


def find_outlets(edges1, edges2):
    """
    Find outlets.

    Find outlet indices given edge node indices.

    Arguments:
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge

    """
    outlets = []
    for e in edges2:
        if e not in edges1:
            outlets.append(e)
    return outlets


def remove_points(idxs_to_delete, idxs_to_replace, edges1, edges2, npoints):
    """
    Remove points.

    Remove points given their indices. This function is useful to find new
    connectivity arrays edges1 and edges2 after deleting nodes.

    Arguments:
        idxs_to_delete: indices of nodes to delete
        idxs_to_replace: indices of nodes that replace the deleted nodes.
                         Must have the same number of components as
                         idxs_to_delete
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        npoints: total number of nodes in the graph

    Returns:
        numpy array with indices of the remaining nodes
        (modified) numpy array containing indices of source nodes for every edge
        (modified) numpy array containing indices of dest nodes for every edge

    """
    npoints_to_delete = len(idxs_to_delete)

    for i in range(npoints_to_delete):
        i1 = np.where(edges1 == idxs_to_delete[i])[0]
        if (len(i1)) != 0:
            edges1[i1] = idxs_to_replace[i]

        i2 = np.where(edges2 == idxs_to_delete[i])[0]
        if (len(i2)) != 0:
            edges2[i2] = idxs_to_replace[i]

    edges_to_delete = np.where(edges1 == edges2)[0]
    edges1 = np.delete(edges1, edges_to_delete)
    edges2 = np.delete(edges2, edges_to_delete)

    sampled_indices = np.delete(np.arange(npoints), idxs_to_delete)
    for i in range(edges1.size):
        edges1[i] = np.where(sampled_indices == edges1[i])[0][0]
        edges2[i] = np.where(sampled_indices == edges2[i])[0][0]

    return sampled_indices, edges1, edges2


def resample_points(points, edges1, edges2, indices, perc_points_to_keep, remove_caps):
    """
    Resample points.

    Select a subset of the points originally contained in the centerline.
    Specifically, this function retains perc_points_to_keep% points deleting
    those corresponding to the smallest edge sizes.

    Arguments:
        points: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        indices: dictionary containing inlet and outlets indices
        perc_points_to_keep (float): percentage of points to keep (in decimals)
        remove_caps (int): number of points to remove at the caps

    Returns:
        numpy array with indices of the remaining nodes
        (modified) n x 3 numpy array of point coordinates
        (modified) numpy array containing indices of source nodes for every edge
        (modified) numpy array containing indices of dest nodes for every edge
        (modified) dictionary containing inlet and outlets indices

    """

    def modify_edges(edges1, edges2, ipoint_to_delete, ipoint_to_replace):
        i1 = np.where(edges1 == ipoint_to_delete)[0]
        if len(i1) != 0:
            edges1[i1] = ipoint_to_replace

        i2 = np.where(np.array(edges2) == ipoint_to_delete)[0]
        if len(i2) != 0:
            edges2[i2] = ipoint_to_replace
        return edges1, edges2

    npoints = points.shape[0]
    npoints_to_keep = int(npoints * perc_points_to_keep)
    ipoints_to_delete = []
    ipoints_to_replace = []

    new_outlets = []
    for ip in range(remove_caps):
        for inlet in indices["inlet"]:
            ipoints_to_delete.append(inlet + ip)
            ipoints_to_replace.append(inlet + remove_caps)
            edges1, edges2 = modify_edges(
                edges1, edges2, inlet + ip, inlet + remove_caps
            )
        for outlet in indices["outlets"]:
            ipoints_to_delete.append(outlet - ip)
            ipoints_to_replace.append(outlet - remove_caps)
            edges1, edges2 = modify_edges(
                edges1, edges2, outlet - ip, outlet - remove_caps
            )

    for outlet in indices["outlets"]:
        new_outlets.append(outlet - remove_caps)

    indices["outlets"] = new_outlets

    for _ in range(npoints - npoints_to_keep):
        diff = np.linalg.norm(points[edges1, :] - points[edges2, :], axis=1)
        # we don't consider the points that we already deleted
        diff[np.where(diff < 1e-13)[0]] = np.inf
        mdiff = np.min(diff)
        mind = np.where(np.abs(diff - mdiff) < 1e-12)[0][0]

        if edges2[mind] not in new_outlets:
            ipoint_to_delete = edges2[mind]
            ipoint_to_replace = edges1[mind]
        else:
            ipoint_to_delete = edges1[mind]
            ipoint_to_replace = edges2[mind]

        edges1, edges2 = modify_edges(
            edges1, edges2, ipoint_to_delete, ipoint_to_replace
        )

        ipoints_to_delete.append(ipoint_to_delete)
        ipoints_to_replace.append(ipoint_to_replace)

    sampled_indices, edges1, edges2 = remove_points(
        ipoints_to_delete, ipoints_to_replace, edges1, edges2, npoints
    )

    points = np.delete(points, ipoints_to_delete, axis=0)

    return sampled_indices, points, edges1, edges2, indices


def dijkstra_algorithm(nodes, edges1, edges2, index):
    """
    Dijkstra's algorithm.

    The algorithm finds the shortest paths from one node to every other node
    in the graph

    Arguments:
        nodes: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        index (int): index of the seed node

    Returns:
        numpy array with n components (n being the total number of nodes)
            containing all shortest path lengths
        numpy array with n components containing the previous nodes explored
            when traversing the graph

    """
    nnodes = nodes.shape[0]
    tovisit = np.arange(0, nnodes)
    dists = np.ones((nnodes)) * np.infty
    prevs = np.ones((nnodes)) * (-1)
    b_edges = np.array([edges1, edges2]).transpose()

    dists[index] = 0
    while len(tovisit) != 0:
        minindex = -1
        minlen = np.infty
        for iinde in range(len(tovisit)):
            if dists[tovisit[iinde]] < minlen:
                minindex = iinde
                minlen = dists[tovisit[iinde]]

        curindex = tovisit[minindex]
        tovisit = np.delete(tovisit, minindex)

        # find neighbors of curindex
        inb = b_edges[np.where(b_edges[:, 0] == curindex)[0], 1]

        for neib in inb:
            if np.where(tovisit == neib)[0].size != 0:
                alt = dists[curindex] + np.linalg.norm(
                    nodes[curindex, :] - nodes[neib, :]
                )
                if alt < dists[neib]:
                    dists[neib] = alt
                    prevs[neib] = curindex
    if np.max(dists) == np.infty:
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=0.5, c="black")
        idx = np.where(dists > 1e30)[0]
        ax.scatter(nodes[idx, 0], nodes[idx, 1], nodes[idx, 2], c="red")
        plt.show()
        raise ValueError(
            "Distance in Dijkstra is infinite for some reason. You can try to adjust resample parameters."
        )
    return dists, prevs


def generate_boundary_edges(points, indices, edges1, edges2):
    """
    Generate boundary edges.

    Generate edges connecting boundary nodes to interior nodes. Every interior
    node is connected to the closest boundary node (in terms of path length).

    Arguments:
        points: n x 3 numpy array of point coordinates
        indices: dictionary containing inlet and outlets indices
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge

    Returns:
        numpy array containing indices of source nodes for every boundary edge
        numpy array containing indices of dest nodes for every boundary edge
        n x 3 numpy array containing (x_j - x_i) / |x_j - x_i|
        n dimensional numpy array containing, for every node, its distance to
            the closest boundary node (in terms of path length)

    """
    npoints = points.shape[0]
    idxs = indices["inlet"] + indices["outlets"]
    bedges1 = []
    bedges2 = []
    rel_positions = []
    dists = []
    types = []
    for index in idxs:
        d, _ = dijkstra_algorithm(points, edges1, edges2, index)
        if index in indices["inlet"]:
            type = 2
        else:
            type = 3
        for ipoint in range(npoints):
            bedges1.append(index)
            bedges2.append(ipoint)
            rp = points[ipoint, :] - points[index, :]
            rel_positions.append(rp)
            if np.linalg.norm(rp) > 1e-12:
                rel_positions[-1] = rel_positions[-1] / np.linalg.norm(rp)
            dists.append(d[ipoint])
            types.append(type)

    # we only keep edges corresponding to the closest boundary node in graph
    # distance to reduce number of edges
    edges_to_delete = []

    for ipoint in range(npoints):
        cur_dists = dists[ipoint::npoints]
        min_dist = np.min(cur_dists)
        minidx = np.where(np.abs(cur_dists - min_dist) < 1e-12)[0][0]
        if min_dist < 1e-12:
            edges_to_delete.append(ipoint + minidx * npoints)
        i = ipoint
        while i < len(dists):
            if i != ipoint + minidx * npoints:
                edges_to_delete.append(i)
            i = i + npoints

    bedges1 = np.delete(np.array(bedges1), edges_to_delete)
    bedges2 = np.delete(np.array(bedges2), edges_to_delete)
    rel_positions = np.delete(np.array(rel_positions), edges_to_delete, axis=0)
    dists = np.delete(np.array(dists), edges_to_delete)
    types = np.delete(np.array(types), edges_to_delete)

    # make edges bidirectional
    bedges1_copy = bedges1.copy()
    bedges1 = np.concatenate((bedges1, bedges2), axis=0)
    bedges2 = np.concatenate((bedges2, bedges1_copy), axis=0)
    rel_positions = np.concatenate((rel_positions, -rel_positions), axis=0)
    dists = np.concatenate((dists, dists))
    types = np.concatenate((types, types))

    return bedges1, bedges2, rel_positions, dists, list(types)


def generate_tangents(points, branch_id):
    """
    Generate tangents.

    Generate tangent vector at every graph node.

    Arguments:
        points: n x 3 numpy array of point coordinates
        branch_id: n-dimensional array containing branch ids

    Returns:
        n x 3 numpy array of normalized tangent vectors

    """
    tangents = np.zeros(points.shape)
    maxbid = int(np.max(branch_id))
    for bid in range(maxbid + 1):
        point_idxs = np.where(branch_id == bid)[0]

        tck, u = scipy.interpolate.splprep(
            [points[point_idxs, 0], points[point_idxs, 1], points[point_idxs, 2]],
            s=0,
            k=np.min((3, len(point_idxs) - 1)),
        )

        x, y, z = scipy.interpolate.splev(u, tck, der=1)
        tangents[point_idxs, 0] = x
        tangents[point_idxs, 1] = y
        tangents[point_idxs, 2] = z

    # make sure tangents are unitary
    tangents = tangents / np.linalg.norm(tangents, axis=0)

    for i in range(tangents.shape[0]):
        tangents[i] = tangents[i] / np.linalg.norm(tangents[i])

    return tangents


def generate_graph(point_data, points, edges1, edges2, add_boundary_edges, rcr_values):
    """
    Generate graph.

    Generate DGL graph out of data obtained from a vtp file.

    Arguments:
        point_data: dictionary containing point data (key: name, value: data)
        points: n x 3 numpy array of point coordinates
        edges1: numpy array containing indices of source nodes for every edge
        edges2: numpy array containing indices of dest nodes for every edge
        add_boundary_edges (bool): decide whether to add boundary edges
        rcr_values: dictionary associating each branch id outlet to values
                    of RCR boundary conditions

    Returns:
        DGL graph
        dictionary containing indices of inlet and outlet nodes
        n x 3 numpy array of point coordinates
        n-dimensional array containin junction ids
        numpy array containing indices of source nodes for every edge
        numpy array containing indices of dist nodes for every edge
    """

    inlet = [0]
    outlets = find_outlets(edges1, edges2)

    indices = {"inlet": inlet, "outlets": outlets}

    bif_id = point_data["BifurcationId"]

    try:
        area = list(gather_array(point_data, "area").values())[0]
    except Exception:
        area = point_data["area"]

    # we manually make the graph bidirected in order to have the relative
    # position of nodes make sense (xj - xi = - (xi - xj)). Otherwise, each edge
    # will have a single feature
    edges1_copy = edges1.copy()
    edges1 = np.concatenate((edges1, edges2))
    edges2 = np.concatenate((edges2, edges1_copy))

    rel_position, distance = generate_edge_features(points, edges1, edges2)

    types, inlet_mask, outlet_mask = generate_types(bif_id, indices)

    # we need to find the closest point in the rcr file, because the
    # id might be different if we used different centerlines for
    # solution and generation of the rcr file
    def find_closest_point_in_rcr_file(point):
        min_d = np.infty
        sid = -1
        for id in rcr_values:
            if type(rcr_values[id]) is dict and "point" in rcr_values[id]:
                diff = np.linalg.norm(point - np.array(rcr_values[id]["point"]))
                if diff < min_d:
                    min_d = diff
                    sid = id
        return sid

    npoints = points.shape[0]
    rcr = np.zeros((npoints, 3))
    for ipoint in range(npoints):
        if outlet_mask[ipoint] == 1:
            if rcr_values["bc_type"] == "RCR":
                id = find_closest_point_in_rcr_file(points[ipoint])
                rcr[ipoint, :] = rcr_values[id]["RCR"]
            elif rcr_values["bc_type"] == "R":
                id = find_closest_point_in_rcr_file(points[ipoint])
                rcr[ipoint, 0] = rcr_values[id]["RP"][0]
            else:
                raise ValueError("Unknown type of boundary conditions!")
    etypes = [0] * edges1.size
    # we set etype to 1 if either of the nodes is a junction
    for iedge in range(edges1.size):
        if types[edges1[iedge], 1] == 1 or types[edges2[iedge], 1] == 1:
            etypes[iedge] = 1

    if add_boundary_edges:
        bedges1, bedges2, brel_position, bdistance, btypes = generate_boundary_edges(
            points, indices, edges1, edges2
        )
        edges1 = np.concatenate((edges1, bedges1))
        edges2 = np.concatenate((edges2, bedges2))
        etypes = etypes + btypes
        distance = np.concatenate((distance, bdistance))
        rel_position = np.concatenate((rel_position, brel_position), axis=0)

    jmasks = {}
    jmasks["inlets"] = np.zeros(bif_id.size)
    jmasks["all"] = np.zeros(bif_id.size)

    graph = dgl.graph((edges1, edges2), idtype=th.int32)

    graph.ndata["x"] = th.tensor(points, dtype=th.float32)
    tangent = th.tensor(point_data["tangent"], dtype=th.float32)
    graph.ndata["tangent"] = th.unsqueeze(tangent, 2)
    graph.ndata["area"] = th.reshape(th.tensor(area, dtype=th.float32), (-1, 1, 1))

    graph.ndata["type"] = th.unsqueeze(types, 2)
    graph.ndata["inlet_mask"] = th.tensor(inlet_mask, dtype=th.int8)
    graph.ndata["outlet_mask"] = th.tensor(outlet_mask, dtype=th.int8)
    graph.ndata["jun_inlet_mask"] = th.tensor(jmasks["inlets"], dtype=th.int8)
    graph.ndata["jun_mask"] = th.tensor(jmasks["all"], dtype=th.int8)
    graph.ndata["branch_mask"] = th.tensor(
        types[:, 0].detach().numpy() == 1, dtype=th.int8
    )
    graph.ndata["branch_id"] = th.tensor(point_data["BranchId"], dtype=th.int8)

    graph.ndata["resistance1"] = th.reshape(
        th.tensor(rcr[:, 0], dtype=th.float32), (-1, 1, 1)
    )
    graph.ndata["capacitance"] = th.reshape(
        th.tensor(rcr[:, 1], dtype=th.float32), (-1, 1, 1)
    )
    graph.ndata["resistance2"] = th.reshape(
        th.tensor(rcr[:, 2], dtype=th.float32), (-1, 1, 1)
    )

    graph.edata["rel_position"] = th.unsqueeze(
        th.tensor(rel_position, dtype=th.float32), 2
    )
    graph.edata["distance"] = th.reshape(
        th.tensor(distance, dtype=th.float32), (-1, 1, 1)
    )
    etypes = th.nn.functional.one_hot(th.tensor(etypes), num_classes=5)
    graph.edata["type"] = th.unsqueeze(etypes, 2)

    return graph
