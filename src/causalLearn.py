import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Edges import Edge
import networkx as nx
import matplotlib.pyplot as plt



def simulate_collider_data(n=1000, seed=42):
    np.random.seed(seed)
    Q = np.random.normal(0, 1, n)
    P = np.random.normal(0, 1, n)
    R = 0.8 * Q + 0.8 * P + np.random.normal(0, 1, n)
    S = 1.5 * R + np.random.normal(0, 1, n)

    data = pd.DataFrame({'Q': Q, 'P': P, 'R': R, 'S': S})
    return data


def run_pc_algorithm(data):
    # Convert to numpy
    data_np = data.to_numpy()

    # Run PC algorithm with Fisher Z test (assuming Gaussian)
    # α is significance threshold for conditional independence
    cg = pc(data_np, alpha=0.01, ci_test=fisherz, labels=list(data.columns))

    return cg


def plot_graph(graph):
    # Print textual summary
    print("\nLearned Causal Graph (edges):")
    
    # transfer all edges to a NetworkX graph for visualization
    nx_graph = nx.DiGraph()
    for edge in graph.G.get_graph_edges():
    
        nx_graph.add_edge(edge.node1, edge.node2)
    nx.draw(nx_graph, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black', arrows=True)
    plt.title("Learned Causal Graph")
    plt.show()
    
    for edge in graph.G.get_graph_edges():
        print(edge)
    
    

def main():
    data = simulate_collider_data()
    cg = run_pc_algorithm(data)
    plot_graph(cg)


if __name__ == "__main__":
    main()

