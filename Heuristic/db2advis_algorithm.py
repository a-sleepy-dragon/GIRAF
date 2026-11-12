from .graph_algorithm import GraphSelectionAlgorithm


class DB2AdvisAlgorithm(GraphSelectionAlgorithm):
    def calculate_best_indexes(self, queries):
        candidates = self._start(queries, "db2advis")
        baseline = self._time(queries)
        print(f"  [db2advis] try {len(candidates)} candidates one by one", flush=True)
        scored = []
        for n, candidate in enumerate(candidates):
            gain, size, _ = self._trial(candidate, [], queries, baseline, 7000 + n)
            if size > 0:
                scored.append((gain / size, candidate, size))
        scored.sort(reverse=True, key=lambda x: x[0])
        chosen, used = [], 0.0
        for _, candidate, size in scored:
            if len(chosen) >= self.max_indexes or used + size > self.budget_mb:
                continue
            chosen.append(candidate); used += size
        return chosen
