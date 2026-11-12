from .graph_algorithm import GraphSelectionAlgorithm, estimate_index_storage


class RelaxationAlgorithm(GraphSelectionAlgorithm):
    def calculate_best_indexes(self, queries):
        candidates = self._start(queries, "relaxation")
        chosen = list(candidates)
        sizes = {x: float(estimate_index_storage(x.label, list(x.props), self.driver, x.on) or 0)
                 for x in chosen}
        used = sum(sizes.values())
        print(f"  [relaxation] start from all {len(chosen)}, remove until budget", flush=True)
        while chosen and (used > self.budget_mb or len(chosen) > self.max_indexes):
            print(f"  [relaxation] {len(chosen)} left, measure which to drop", flush=True)
            base = self._measure_configuration(chosen, queries, 9000)
            trials = []
            for candidate in chosen:
                remaining = [x for x in chosen if x != candidate]
                after = self._measure_configuration(remaining, queries, 9001)
                loss = sum(after.values()) - sum(base.values())
                trials.append((loss / max(sizes[candidate], 1e-9), candidate))
            _, removed = min(trials, key=lambda x: x[0])
            chosen.remove(removed); used -= sizes[removed]
        return chosen
