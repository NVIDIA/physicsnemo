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

import os
import numpy as np
import random
from tqdm import tqdm
import torch as th
from dgl.data.utils import load_graphs as lg
from dgl.data import DGLDataset
import time
import copy


def compute_statistics(graphs, fields, statistics):
    """
    Compute statistics on a list of graphs.

    The computed statistics are: min value, max value, mean, and standard
    deviation.

    Arguments:
        graphs: list of graphs
        fields: dictionary containing field names, divided into node and edge
                fields
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
    Returns:
        dictionary containining statistics (key: statistics name, value: value).
        New fields are appended to the input 'statistics' argument.
    """

    print("Compute statistics")
    for etype in fields:
        for field_name in fields[etype]:
            cur_statistics = {}
            minv = np.infty
            maxv = np.NINF
            Ns = []
            Ms = []
            means = []
            meansqs = []
            for graph_n in tqdm(graphs, desc=field_name, colour="green"):
                graph = graphs[graph_n]
                if etype == "node":
                    d = graph.ndata[field_name]
                elif etype == "edge":
                    d = graph.edata[field_name]
                elif etype == "outlet_node":
                    mask = graph.ndata["outlet_mask"].bool()
                    d = graph.ndata[field_name][mask]

                # number of nodes
                N = d.shape[0]
                # number of times
                M = d.shape[2]
                minv = np.min([minv, th.min(d)])
                maxv = np.max([maxv, th.max(d)])
                mean = float(th.mean(d))
                meansq = float(th.mean(d**2))

                means.append(mean)
                meansqs.append(meansq)
                Ns.append(N)
                Ms.append(M)

            ngraphs = len(graphs)
            MNs = 0
            for i in range(ngraphs):
                MNs = MNs + Ms[i] * Ns[i]

            mean = 0
            meansq = 0
            for i in range(ngraphs):
                coeff = Ms[i] * Ns[i] / MNs
                mean = mean + coeff * means[i]
                meansq = meansq + coeff * meansqs[i]

            cur_statistics["min"] = minv
            cur_statistics["max"] = maxv
            cur_statistics["mean"] = mean
            cur_statistics["stdv"] = np.sqrt(meansq - mean**2)
            statistics[field_name] = cur_statistics

    graph_sts = {"nodes": [], "edges": [], "tsteps": []}

    for graph_n in graphs:
        graph = graphs[graph_n]
        graph_sts["nodes"].append(graph.ndata["x"].shape[0])
        graph_sts["edges"].append(graph.edata["distance"].shape[0])
        graph_sts["tsteps"].append(graph.ndata["pressure"].shape[2])

    for name in graph_sts:
        cur_statistics = {}

        cur_statistics["min"] = int(np.min(graph_sts[name]))
        cur_statistics["max"] = int(np.max(graph_sts[name]))
        cur_statistics["mean"] = np.mean(graph_sts[name])
        cur_statistics["stdv"] = np.std(graph_sts[name])

        statistics[name] = cur_statistics

    return statistics


def load_graphs(input_dir):
    """
    Load all graphs in directory.

    Arguments:
        input_dir (string): input directory path

    Returns:
        list of DGL graphs

    """
    files = os.listdir(input_dir)
    random.seed(10)
    random.shuffle(files)

    graphs = {}
    for file in tqdm(files, desc="Loading graphs", colour="green"):
        if "grph" in file:
            graphs[file] = lg(input_dir + file)[0][0]

    return graphs


def normalize(field, field_name, statistics, norm_dict_label):
    """
    Normalize field.

    Normalize a field using statistics provided as input.

    Arguments:
        field: the field to normalize
        field_name (string): name of field
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'features' or 'labels'
    Returns:
        normalized field

    """
    if statistics["normalization_type"][norm_dict_label] == "min_max":
        delta = statistics[field_name]["max"] - statistics[field_name]["min"]
        if np.abs(delta) > 1e-5:
            field = (field - statistics[field_name]["min"]) / delta
        else:
            field = field * 0
    elif statistics["normalization_type"][norm_dict_label] == "normal":
        delta = statistics[field_name]["stdv"]
        if np.abs(delta) > 1e-5 and not np.isnan(delta):
            field = (field - statistics[field_name]["mean"]) / delta
        else:
            field = field * 0
    elif statistics["normalization_type"][norm_dict_label] == "none":
        pass
    else:
        raise Exception("Normalization type not implemented")
    return field


def normalize_graphs(graphs, fields, statistics, norm_dict_label):
    """
    Normalize all graphs in a list.

    Arguments:
        graphs: list of graphs
        fields: dictionary containing field names, divided into node and edge
                fields
        statistics: dictionary containining statistics
                    (key: statistics name, value: value)
        norm_dict_label (string): 'features' or 'labels'

    """
    print("Normalize graphs")
    for etype in fields:
        for field_name in fields[etype]:
            for graph_n in tqdm(graphs, desc=field_name, colour="green"):
                graph = graphs[graph_n]
                if etype == "node":
                    d = graph.ndata[field_name]
                    graph.ndata[field_name] = normalize(
                        d, field_name, statistics, norm_dict_label
                    )
                elif etype == "edge":
                    d = graph.edata[field_name]
                    graph.edata[field_name] = normalize(
                        d, field_name, statistics, norm_dict_label
                    )
                elif etype == "outlet_node":
                    d = graph.ndata[field_name]
                    graph.ndata[field_name] = normalize(
                        d, field_name, statistics, norm_dict_label
                    )


def add_features(graphs):
    """
    Add features to graphs.

    This function adds node and edge features to all graphs in
    the input list.

    Arguments:
        graphs: list of graphs.
    """
    # pressure and flowrate are always included
    nodes_features = [
        "area",
        "tangent",
        "type",
        "T",
        "dip",
        "sysp",
        "resistance1",
        "capacitance",
        "resistance2",
        "loading",
    ]

    edges_features = ["rel_position", "distance", "type"]

    for graph_n in tqdm(graphs, desc="Add features", colour="green"):
        graph = graphs[graph_n]
        ntimes = graph.ndata["pressure"].shape[2]

        cf = []

        def add_feature(tensor, desired_features, label):
            if label in desired_features:
                cf.append(tensor)

        # graph.ndata['dt'].repeat(1, 1, ntimes)
        add_feature(graph.ndata["area"].repeat(1, 1, ntimes), nodes_features, "area")
        add_feature(
            graph.ndata["tangent"].repeat(1, 1, ntimes), nodes_features, "tangent"
        )
        add_feature(graph.ndata["type"].repeat(1, 1, ntimes), nodes_features, "type")
        add_feature(graph.ndata["T"].repeat(1, 1, ntimes), nodes_features, "T")

        loading = graph.ndata["loading"]

        p = graph.ndata["pressure"].clone()
        q = graph.ndata["flowrate"].clone()

        add_feature(th.ones(p.shape[0], 1, ntimes) * th.min(p), nodes_features, "dip")
        add_feature(th.ones(p.shape[0], 1, ntimes) * th.max(p), nodes_features, "sysp")

        outmask = graph.ndata["outlet_mask"].bool()
        nnodes = outmask.shape[0]

        r1 = th.zeros((nnodes, 1, ntimes), dtype=th.float32)
        c = th.zeros((nnodes, 1, ntimes), dtype=th.float32)
        r2 = th.zeros((nnodes, 1, ntimes), dtype=th.float32)
        r1[outmask, 0, :] = graph.ndata["resistance1"][outmask, 0, :]
        c[outmask, 0, :] = graph.ndata["capacitance"][outmask, 0, :]
        r2[outmask, 0, :] = graph.ndata["resistance2"][outmask, 0, :]
        add_feature(r1, nodes_features, "resistance1")
        add_feature(c, nodes_features, "capacitance")
        add_feature(r2, nodes_features, "resistance2")

        cfeatures = th.cat(cf, axis=1)

        if "loading" in nodes_features:
            loading = graph.ndata["loading"]
            graph.ndata["nfeatures"] = th.cat((p, q, cfeatures, loading), axis=1)
        else:
            graph.ndata["nfeatures"] = th.cat((p, q, cfeatures), axis=1)

        cf = []
        add_feature(graph.edata["rel_position"], edges_features, "rel_position")
        add_feature(graph.edata["distance"], edges_features, "distance")
        add_feature(graph.edata["type"], edges_features, "type")

        graph.edata["efeatures"] = th.cat(cf, axis=1)


def generate_normalized_graphs(input_dir, norm_type, geometries, statistics=None):
    """
    Generate normalized graphs.

    Arguments:
        input_dir: path to input directory
        norm_type: dictionary with keys: features/labels,
                   values: min_max/normal
        statistics: dictionary containing statistics previously computed.
                    Default value -> None.
        geometries: family of geometries to consider: 'healthy',
                    'pathological', 'mixed'

    Return:
        List of normalized graphs
        Dictionary of parameters

    """
    fields_to_normalize = {
        "node": ["area", "pressure", "flowrate", "T"],
        "edge": ["distance"],
        "outlet_node": ["resistance1", "capacitance", "resistance2"],
    }

    docompute_statistics = True
    if statistics != None:
        docompute_statistics = False

    if docompute_statistics:
        statistics = {"normalization_type": norm_type}
    graphs = load_graphs(input_dir)

    if geometries == "mixed":
        pass
    else:
        graphs_to_keep = {}
        if geometries == "healthy":
            list_of_models = [
                "s0090_0001",
                "s0091_0001",
                "s0093_0001",
                "s0094_0001",
                "s0095_0001",
            ]
        elif geometries == "pathological":
            list_of_models = ["s0104_0001", "s0080_0001", "s0140_2001"]
        else:
            raise ValueError("Type of geometry " + geometries + "does not exist")

        for graph in graphs:
            for s in list_of_models:
                if s in graph:
                    graphs_to_keep[graph] = graphs[graph]
                    continue
        graphs = graphs_to_keep

    if docompute_statistics:
        compute_statistics(graphs, fields_to_normalize, statistics)

    normalize_graphs(graphs, fields_to_normalize, statistics, "features")

    params = {"statistics": statistics}
    add_features(graphs)

    return graphs, params


class Bloodflow1DDataset(DGLDataset):
    """
    Class to store and traverse a DGL dataset.

    Attributes:
        graphs: list of graphs in the dataset
        params: dictionary containing parameters of the problem
        times: array containing number of times for each graph in the dataset
        lightgraphs: list of graphs, without edge and node features
        graph_names: n x 2 array (n is the total number of timesteps in the
                     dataset) mapping a graph index (first column) to the
                     timestep index (second column).

    """

    def __init__(self, graphs, params, graph_names):
        """
        Init Dataset.

        Init Dataset with list of graphs, dictionary of parameters, and list of
        graph names.

        Arguments:
            graphs: lift of graphs
            params: dictionary of parameters
            graph_names: list of graph names
            index_map:

        """
        self.graphs = graphs
        self.params = params
        self.times = []
        self.lightgraphs = []
        self.graph_names = graph_names
        super().__init__(name="dataset")

    def create_index_map(self):
        """
        Create index map.

        Index map is a n x 2 array (n is the total number of timesteps in the
        dataset) mapping a graph index (first column) to the timestep index
        (second column).

        """
        i = 0
        offset = 0
        ngraphs = len(self.times)
        stride = self.params["stride"]
        self.index_map = np.zeros((self.total_times - stride * ngraphs, 2))
        for t in self.times:
            # actual time (minus stride)
            at = t - stride
            graph_index = np.ones((at, 1)) * i
            time_index = np.expand_dims(np.arange(0, at), axis=1)
            self.index_map[offset : at + offset, :] = np.concatenate(
                (graph_index, time_index), axis=1
            )
            i = i + 1
            offset = offset + at
        self.index_map = np.array(self.index_map, dtype=int)

    def process(self):
        """
        Process Dataset.

        This function creates lightgraphs, the index map, and collects all times
        from the graphs.

        """
        start = time.time()

        for graph in tqdm(self.graphs, desc="Processing dataset", colour="green"):
            lightgraph = copy.deepcopy(graph)

            node_data = [ndata for ndata in lightgraph.ndata]
            edge_data = [edata for edata in lightgraph.edata]
            for ndata in node_data:
                if "mask" not in ndata:
                    del lightgraph.ndata[ndata]
            for edata in edge_data:
                del lightgraph.edata[edata]

            self.times.append(graph.ndata["nfeatures"].shape[2])
            self.lightgraphs.append(lightgraph)

        self.times = np.array(self.times)
        self.total_times = np.sum(self.times)

        self.create_index_map()

        end = time.time()
        elapsed_time = end - start
        print("\tDataset generated in {:0.2f} s".format(elapsed_time))

    def get_lightgraph(self, i):
        """
        Get ith lightgraph

        Noise is added to node features of the graph (pressure and flowrate).

        Arguments:
            i: index of the graph

        Returns:
            The DGL graph
        """
        indices = self.index_map[i, :]
        igraph = indices[0]
        itime = indices[1]

        features = self.graphs[igraph].ndata["nfeatures"]

        nf = features[:, :, itime].clone()
        nfsize = nf[:, :2].shape

        dt = self.graphs[igraph].ndata["dt"][0]

        # add random noise to pressure and flowrate to account for error
        # injected by the network
        curnoise = np.random.normal(0, self.params["rate_noise"] * dt, nfsize)
        curnoise[self.graphs[igraph].ndata["inlet_mask"].bool(), 1] = 0

        nf[:, :2] = nf[:, :2] + curnoise
        self.lightgraphs[igraph].ndata["nfeatures"] = nf

        ns = features[:, 0:2, itime + 1 : itime + 1 + self.params["stride"]]
        self.lightgraphs[igraph].ndata["next_steps"] = ns

        ef = self.graphs[igraph].edata["efeatures"]
        self.lightgraphs[igraph].edata["efeatures"] = ef.squeeze()

        return self.lightgraphs[igraph]

    def __getitem__(self, i):
        """
        Get ith lightgraph

        Arguments:
            i: index of the lightgraph

        Returns:
            ith lightgraph
        """
        return self.get_lightgraph(i)

    def __len__(self):
        """
        Length of the dataset

        Length of the dataset is the total number of timesteps (minus stride).

        Returns:
            length of the Dataset
        """
        return self.index_map.shape[0]

    def __str__(self):
        """
        Returns graph names.

        Returns:
            graph names
        """
        print("Total number of graphs: {:}".format(self.__len__()))
        return "Dataset = " + ", ".join(self.graph_names)


def train_test_split(graphs, perc):
    """
    Create two list of graphs, a train one and a test one, from a global
    dictionary. Graphs are organized to avoid data leaks (i.e., augmented
    graphs are assigned to the same set as the original one)

    Arguments:
        graphs: dictionary of graphs (key: name, value: DGL graph)
        perc: percentage of graphs in the train set (between 0 and 1)

    Returns:
        list of train graphs
        list of test graphs
    """

    nameset = set()
    for name in graphs:
        simname = name.split(".")[0] + "." + name.split(".")[1]
        nameset.add(simname)

    namelist = list(nameset)
    ntrain = int(perc * len(namelist))

    # this works if every graph is augmented the same number of times
    ncopies = int(len(graphs) / len(namelist))

    trainset = []
    testset = []
    for i, name in enumerate(namelist):
        if i <= ntrain:
            for j in range(ncopies):
                trainset.append(name + ".{:}.grph".format(j))
        else:
            for j in range(ncopies):
                testset.append(name + ".{:}.grph".format(j))

    return trainset, testset


if __name__ == "__main__":
    t_params, args = parse_command_line_arguments()
    norm_type = {"features": "normal", "labels": "normal"}
    graphs, params = generate_normalized_graphs("raw_dataset/graphs/", norm_type)

    graph = graphs[list(graphs)[0]]

    infeat_nodes = graph.ndata["nfeatures"].shape[1]
    infeat_edges = graph.edata["efeatures"].shape[1]
    nout = 2

    nodes_features = [
        "area",
        "tangent",
        "type",
        "T",
        "dip",
        "sysp",
        "resistance1",
        "capacitance",
        "resistance2",
        "loading",
    ]

    edges_features = ["rel_position", "distance", "type"]

    t_params["infeat_nodes"] = infeat_nodes
    t_params["infeat_edges"] = infeat_edges
    t_params["out_size"] = nout
    params["node_features"] = nodes_features
    params["edges_features"] = edges_features

    params.update(t_params)

    trainset, testset = train_test_split(graphs, 0.9)

    train_graphs = [graphs[gname] for gname in trainset]
    traindataset = Bloodflow1DDataset(train_graphs, params, trainset)

    test_graphs = [graphs[gname] for gname in testset]
    traindataset = Bloodflow1DDataset(test_graphs, params, testset)
