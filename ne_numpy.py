import af_reader_py
import time
import networkx as nx
import sys

af_path = "../af_dataset/dataset_af/Large-result_b1.af"
tic = time.perf_counter()
att1, att2, nb_el = af_reader_py.reading_cnf_for_dgl(af_path)
toc = time.perf_counter()
print(toc-tic , " seconds for RUST ")
tic = time.perf_counter()
nxg = nx.DiGraph()
nodes = list([s for s in range(0, nb_el)])
att = list([(s, att2[i]) for i, s in enumerate(att1)])
nxg.add_nodes_from(nodes)
nxg.add_edges_from(att)
print("graph creation : ", time.perf_counter()-tic)

sys.stdout.flush()
#page_rank = (nx.pagerank(nxg))

tic = time.perf_counter()
degree_centrality = list(nx.degree_centrality(nxg).values())
in_degrees = list( s for (i, s) in nxg.in_degree())
out_degrees = list( s for (i, s) in nxg.out_degree())
print("pre",time.perf_counter()-tic)
sys.stdout.flush()
tic = time.perf_counter()
af_reader_py.compute_features(af_path, degree_centrality, in_degrees, out_degrees, 10000, 0.0001)
toc = time.perf_counter()
print("rust : ", toc -tic)