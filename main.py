import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

graph = nx.Graph()
cities = open('./data/edges.csv')

# Carrega as arestas
for line in cities:
    data = line.split(",")
    data[2] = data[2].replace("\n", "")
    graph.add_edge(data[0], data[1], weight=float(data[2]))

# Carrega a eurística
heuristics = pd.read_csv("./data/heuristics.csv", header=0, index_col=0)


def dfs_paths(graph, start, goal, path=None):
    """Algoritmo de busca em profundidade
    Retorna um generator contendo todos os caminhos encontados na busca"""
    if path is None:
        path = [start]
    if start == goal:
        yield path
    for next in set([i for i in graph.neighbors(start)]) - set(path):
        yield from dfs_paths(graph, next, goal, path + [next])


def bfs_paths(graph, start, goal):
    """Algoritmo de busca em largura
        Retorna um generator contendo todos os caminhos encontados na busca"""
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in set([i for i in graph.neighbors(vertex)]) - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))


def paint_edges(graph, pos, route, color):
    """Função auxiliar que pinta as arestas e os pontos de início e destino no grafo"""
    edges = [(route[n], route[n + 1]) for n in range(len(route) - 1)]
    nx.draw_networkx_edges(graph, pos=pos, edgelist=edges, edge_color=color, width=2.0)
    nx.draw_networkx_nodes(graph, pos, nodelist=[route[0], route[-1]], node_size=120, node_color='blue')


def best_first(graph, start, goal, path=None):
    """Algorítmo de busca best-first, se guia apenas pela menor aresta adjacente"""
    if path is None:
        path = [start]
    if start == goal:
        return path

    prox = node_selector(graph, start, path)
    return best_first(graph, prox, goal, path + [prox])


def node_selector(graph, node, visited=None):
    """Seletor de qual o destino com aresta de menor peso"""
    if not visited:
        visited = [node]
    else:
        if node not in visited:
            visited.append(node)

    adj = set(graph[node]) - set(visited)
    min = list(adj)[0]

    for i in adj:
        if graph[node][i]['weight'] < graph[node][min]['weight']:
            min = i

    return min


def ns_heuristics(graph, node, goal, visited=None):
    """Seletor de destino com menor peso levando em conta herística e peso da aresta.
    A heurística utilizada foi a distância em linha reta entre as cidades"""

    if not visited:
        visited = [node]
    else:
        if node not in visited:
            visited.append(node)

    adj = list(set(graph[node]) - set(visited))
    if goal in adj:
        return goal
    min = adj[0]

    for i in adj[1:]:
        heu_i = heuristics[goal][i]
        if np.isnan(heu_i):
            heu_i = heuristics[i][goal]

        heu_min = heuristics[goal][min]
        if np.isnan(heu_min):
            heu_min = heuristics[min][goal]

        if graph[node][i]['weight'] + heu_i < graph[node][min]['weight'] + heu_min:
            min = i

    return min


def a_star(graph, start, goal, path=None):
    """Algoritmo de busca A*
    Retorna o caminho calculado de acordo com o menor peso da aresta + heurística até o destino"""
    if path is None:
        path = [start]
    if start == goal:
        return path

    prox = ns_heuristics(graph, start, goal, path)
    return a_star(graph, prox, goal, path + [prox])


def get_pos(graph):
    """Carrega o posicionamento das arestas caso o mesmo exista,
    caso contrário, é aplicado o layout spring do NetworkX"""

    try:
        coord = open('./data/coordinates.csv')
        pos = {}
        for line in coord:
            line.replace("\n", "")
            line = line.split(",")
            pos[line[0]] = (float(line[1]), float(line[2]))

    except:
        pos = nx.spring_layout(graph)

    return pos


if __name__ == '__main__':
    while True:

        start = input("Origem: \n")
        if start not in graph.nodes:
            print("{} não contida na KB".format(start))
            continue
        goal = input("Destino: \n")
        if goal not in graph.nodes:
            print("{} não contida na KB".format(goal))
            continue
        alg = int(
            input("Qual algorítmo usar?\n1 - Busca em largura\n2 - Busca em Profundidade\n3 - Best-first\n4 - A*\n"))
        if alg == 1:
            plt.title("Busca em largura\n{} para {}".format(start, goal))
            res = next(bfs_paths(graph, start, goal))
        elif alg == 2:
            plt.title("Busca em profundidade\n{} para {}".format(start, goal))
            res = next(dfs_paths(graph, start, goal))
        elif alg == 3:
            plt.title("Busca em \"melhor primeiro\"\n{} para {}".format(start, goal))
            res = best_first(graph, start, goal)
        elif alg == 4:
            plt.title("Busca em A*\n{} para {}".format(start, goal))
            res = a_star(graph, start, goal)
        else:
            break

        pos = get_pos(graph)
        nx.draw_networkx_nodes(graph, pos, node_size=40, node_color='lime')
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        paint_edges(graph, pos, res, 'cyan')
        plt.show()
