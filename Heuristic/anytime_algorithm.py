from .graph_algorithm import GraphSelectionAlgorithm


class AnytimeAlgorithm(GraphSelectionAlgorithm):
    def calculate_best_indexes(self, queries):
        candidates = self._start(queries, "anytime")
        return self._greedy(candidates, queries, "anytime")
