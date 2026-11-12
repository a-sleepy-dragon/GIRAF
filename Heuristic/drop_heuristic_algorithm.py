from .graph_algorithm import GraphSelectionAlgorithm, estimate_index_storage


class DropHeuristicAlgorithm(GraphSelectionAlgorithm):
    def calculate_best_indexes(self, queries):
        candidates = self._start(queries, "drop")
        chosen = candidates[:]
        sizes = {c: float(estimate_index_storage(c.label, list(c.props), self.driver, c.on) or 0)
                 for c in chosen}
        used = sum(sizes.values())
        while chosen and (len(chosen) > self.max_indexes or used > self.budget_mb):
            print(f"  [drop] {len(chosen)} left, measure which to drop", flush=True)
            base = self._time(queries)
            trials = [(self._trial_drop(i, chosen, queries), i) for i in chosen]
            _, removed = min(trials, key=lambda x: x[0])
            chosen.remove(removed)
            used -= sizes[removed]
        return chosen
