import af_reader_py
import time
import networkx as nx

af_path = "../af_dataset\dataset_af/admbuster_2500000.af"
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
tic = time.perf_counter()
tic3 = time.perf_counter()
page_rank = (nx.pagerank(nxg))
degree_centrality = (nx.degree_centrality(nxg))
in_degrees = nxg.in_degree()
out_degrees = nxg.out_degree()
toc = time.perf_counter()
print(toc -tic)