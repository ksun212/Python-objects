
from typing import Union
from collections.abc import Mapping

class FilterAdjacency(Mapping):  # edgedict
    def __init__(self, d, NODE_OK, EDGE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK
        self.EDGE_OK = EDGE_OK

    def __len__(self):
        return sum(1 for n in self)

    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, node):
        if node in self._atlas and self.NODE_OK(node):

            def new_node_ok(nbr):
                return self.NODE_OK(nbr) and self.EDGE_OK(node, nbr)

            return FilterAtlas(self._atlas[node], new_node_ok)
        raise KeyError(f"Key {node} not found")

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self._atlas!r}, {self.NODE_OK!r}, {self.EDGE_OK!r})"


class FilterMultiInner(FilterAdjacency):  # muliedge_seconddict
    def __iter__(self):
        try:  # check that NODE_OK has attr 'nodes'
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            my_nodes = (n for n in self.NODE_OK.nodes if n in self._atlas)
        else:
            my_nodes = (n for n in self._atlas if self.NODE_OK(n))
        for n in my_nodes:
            some_keys_ok = False
            for key in self._atlas[n]:
                if self.EDGE_OK(n, key):
                    some_keys_ok = True
                    break
            if some_keys_ok is True:
                yield n

    def __getitem__(self, nbr):
        if nbr in self._atlas and self.NODE_OK(nbr):

            def new_node_ok(key):
                return self.EDGE_OK(nbr, key)

            return FilterAtlas(self._atlas[nbr], new_node_ok)
        raise KeyError(f"Key {nbr} not found")


class FilterMultiAdjacency(FilterAdjacency):  # multiedgedict
    def __getitem__(self, node):
        if node in self._atlas and self.NODE_OK(node):

            def edge_ok(nbr, key):
                return self.NODE_OK(nbr) and self.EDGE_OK(node, nbr, key)

            return FilterMultiInner(self._atlas[node], self.NODE_OK, edge_ok)
        raise KeyError(f"Key {node} not found")


class Graph:
    _adj: Union[FilterMultiAdjacency, dict] = _CachedPropertyResetterAdj()
    _node = _CachedPropertyResetterNode()

    node_dict_factory = dict
    node_attr_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict
    graph_attr_dict_factory = dict

    
    def __init__(self, incoming_graph_data=None, **attr):
        self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
        self._node = self.node_dict_factory()  # empty node attribute dict
        self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict
        # attempt to load graph with data
        if incoming_graph_data is not None:
            convert.to_networkx_graph(incoming_graph_data, create_using=self)
        # load graph attributes (must be after convert)
        self.graph.update(attr)


    def clear(self):
        self._adj.clear()
        self._node.clear()
        self.graph.clear()