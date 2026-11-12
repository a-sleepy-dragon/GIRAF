from .graph_algorithm import GraphSelectionAlgorithm


class ExtendAlgorithm(GraphSelectionAlgorithm):
    def calculate_best_indexes(self, queries):
        candidates = self._start(queries, "extend")
        return self._greedy(candidates, queries, "extend")
