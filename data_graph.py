
import random
import time
from pathlib import Path

import dgl
import torch


# ------------------------------------------------------------
# 1. Project and dataset paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "8_rat"

random.seed(8026728)


# ------------------------------------------------------------
# 2. Available circuits and training split
# ------------------------------------------------------------

available_data = [
    'aes_cipher',
    'blabla',
    'des',
    'genericfir',
    'spm',
    'usb',
    'BM64',
    'cic_decimator',
    'synth_ram',
    'zipdiv',
]

train_data_keys = [
    'aes_cipher',
    'blabla',
    'des',
    'genericfir',
    'spm',
    'usb',
]



# ------------------------------------------------------------
# 3. Generate topological ordering
# ------------------------------------------------------------

def gen_topo(g_hetero):
    time_s = time.time()

    na, nb = g_hetero.edges(etype="net_out", form="uv")
    ca, cb = g_hetero.edges(etype="cell_out", form="uv")

    g = dgl.graph(
        (
            torch.cat([na, ca]),
            torch.cat([nb, cb]),
        )
    )

    topo = dgl.topological_nodes_generator(g)
    ret = [t for t in topo]

    time_e = time.time()

    return ret, time_e - time_s


# ------------------------------------------------------------
# 4. Convert heterogeneous graph to homogeneous graph
#    Used by the DeepGCNII baseline
# ------------------------------------------------------------

def gen_homobigraph_with_features(g_hetero):
    na, nb = g_hetero.edges(etype="net_out", form="uv")
    ca, cb = g_hetero.edges(etype="cell_out", form="uv")

    ne = torch.cat(
        [
            torch.tensor(
                [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]],
                dtype=torch.float32,
            ).expand(len(na), 10),
            g_hetero.edges["net_out"].data["ef"],
        ],
        dim=1,
    )

    ce = (
        g_hetero.edges["cell_out"].data["ef"][:, 120:512]
        .reshape(len(ca), 2 * 4, 49)
    )

    ce = torch.cat(
        [
            torch.tensor(
                [[1., 0.]], dtype=torch.float32
            ).expand(len(ca), 2),
            torch.mean(ce, dim=2),
            torch.zeros(
                (len(ca), 2),
                dtype=torch.float32,
            ),
        ],
        dim=1,
    )

    g = dgl.graph(
        (
            torch.cat([na, ca, nb, cb]),
            torch.cat([nb, cb, na, ca]),
        )
    )

    g.ndata["nf"] = g_hetero.ndata["nf"]
    g.ndata["n_atslew"] = g_hetero.ndata["n_atslew"]

    g.edata["ef"] = torch.cat(
        [ne, ce, -ne, -ce],
        dim=0,
    )

    return g


# ------------------------------------------------------------
# 5. Load and preprocess graph datasets
# ------------------------------------------------------------

data = {}

for k in available_data:

    graph_path = DATA_DIR / f"{k}.graph.bin"

    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Graph dataset not found: {graph_path}\n"
            f"Check that the file exists inside {DATA_DIR}."
        )

    # Load graph on CPU.
    graphs, _ = dgl.load_graphs(str(graph_path))
    g = graphs[0]

    # Log-transform net delays.
    g.ndata["n_net_delays_log"] = (
        torch.log(0.0001 + g.ndata["n_net_delays"]) + 7.6
    )

    # Ignore uninitialized stray pins.
    invalid_nodes = torch.abs(g.ndata["n_ats"]) > 1e20

    g.ndata["n_ats"][invalid_nodes] = 0
    g.ndata["n_slews"][invalid_nodes] = 0

    # Combine arrival-time and transformed slew values.
    g.ndata["n_atslew"] = torch.cat(
        [
            g.ndata["n_ats"],
            torch.log(0.0001 + g.ndata["n_slews"]) + 3,
        ],
        dim=1,
    )

    # Ensure cell-edge features and delays use float32.
    g.edges["cell_out"].data["ef"] = (
        g.edges["cell_out"].data["ef"].to(torch.float32)
    )

    g.edges["cell_out"].data["e_cell_delays"] = (
        g.edges["cell_out"].data["e_cell_delays"].to(torch.float32)
    )

    # Generate topological ordering.
    topo, topo_time = gen_topo(g)

    # Store node groups and timing information.
    ts = {
        "input_nodes": (
            (g.ndata["nf"][:, 1] < 0.5)
            .nonzero()
            .flatten()
            .to(torch.int32)
        ),

        "output_nodes": (
            (g.ndata["nf"][:, 1] > 0.5)
            .nonzero()
            .flatten()
            .to(torch.int32)
        ),

        "output_nodes_nonpi": (
            torch.logical_and(
                g.ndata["nf"][:, 1] > 0.5,
                g.ndata["nf"][:, 0] < 0.5,
            )
            .nonzero()
            .flatten()
            .to(torch.int32)
        ),

        "pi_nodes": (
            torch.logical_and(
                g.ndata["nf"][:, 1] > 0.5,
                g.ndata["nf"][:, 0] > 0.5,
            )
            .nonzero()
            .flatten()
            .to(torch.int32)
        ),

        "po_nodes": (
            torch.logical_and(
                g.ndata["nf"][:, 1] < 0.5,
                g.ndata["nf"][:, 0] > 0.5,
            )
            .nonzero()
            .flatten()
            .to(torch.int32)
        ),

        "endpoints": (
            (g.ndata["n_is_timing_endpt"] > 0.5)
            .nonzero()
            .flatten()
            .to(torch.long)
        ),

        "topo": topo,
        "topo_time": topo_time,
    }

    data[k] = (g, ts)


# ------------------------------------------------------------
# 6. Split circuits into training and testing sets
# ------------------------------------------------------------

data_train = {
    k: item
    for k, item in data.items()
    if k in train_data_keys
}

data_test = {
    k: item
    for k, item in data.items()
    if k not in train_data_keys
}


# ------------------------------------------------------------
# 7. Print dataset statistics
# ------------------------------------------------------------

if __name__ == "__main__":

    print(f"Project directory : {PROJECT_DIR}")
    print(f"Graph data folder : {DATA_DIR}")
    print(f"Available circuits: {list(data.keys())}")
    print(f"Training circuits : {list(data_train.keys())}")
    print(f"Testing circuits  : {list(data_test.keys())}")
    print()

    print(
        f"{'NAME':<20}"
        f"{'NODES':>12}"
        f"{'NET EDGES':>14}"
        f"{'CELL EDGES':>14}"
        f"{'TOPO LEVELS':>14}"
        f"{'PO NODES':>12}"
        f"{'ENDPOINTS':>12}"
    )

    for dataset in (data_train, data_test):
        for name, (g, ts) in dataset.items():
            print(
                f"{name:<20}"
                f"{g.num_nodes():>12}"
                f"{g.num_edges(etype='net_out'):>14}"
                f"{g.num_edges(etype='cell_out'):>14}"
                f"{len(ts['topo']):>14}"
                f"{len(ts['po_nodes']):>12}"
                f"{len(ts['endpoints']):>12}"
            )
