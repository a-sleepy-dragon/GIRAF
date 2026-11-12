import itertools

from .graph_algorithm import GraphSelectionAlgorithm, estimate_index_storage


class AutoAdminAlgorithm(GraphSelectionAlgorithm):
    def calculate_best_indexes(self, queries):
        candidates = self._start(queries, "auto_admin")
        limit = min(int(self.parameters.get("max_indexes_naive", 2)), len(candidates))
        baseline = self._time(queries)
        print(f"  [auto_admin] enumerate combos (size 1..{limit}), {len(candidates)} candidates", flush=True)
        best = (sum(baseline.values()), [])
        for size in range(1, limit + 1):
            for combo in itertools.combinations(candidates, size):
                space = sum(float(estimate_index_storage(x.label, list(x.props), self.driver, x.on) or 0)
                            for x in combo)
                if space > self.budget_mb:
                    continue
                after = self._measure_configuration(combo, queries, 8000 + size)
                cost = sum(after.values())
                if cost < best[0]:
                    best = (cost, list(combo))
        return best[1]
