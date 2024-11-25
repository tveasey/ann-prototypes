import numpy as np

from tqdm.auto import tqdm

cost_of_copy = 0.05
cost_of_insert = 1.0

def merge_cost(n1: int, n2: list) -> float:
    n_s = np.sum(n2)
    return cost_of_copy * (n1 + n_s) + cost_of_insert * n_s

class BaselineMergePolicy:
    def __init__(self, merge_count: int = 10):
        self.n_ = 0
        self.f_ = 0.0
        self.segments_ = []
        self.fractions_ = []
        self.max_merge_count_ = merge_count

    def should_promote(self) -> bool:
        return self.n_ > 0

    def add(self, n: int, f: float) -> float:
        self.segments_.append(n)
        self.fractions_.append(f)
        return self.flush() if len(self.segments_) == self.max_merge_count_ else 0.0

    def flush(self) -> tuple[float, float, float]:
        count = len(self.segments_)
        if count == 0:
            return 0.0
        max = np.argmax([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.segments_[0], self.segments_[max] = self.segments_[max], self.segments_[0]
        self.fractions_[0], self.fractions_[max] = self.fractions_[max], self.fractions_[0]
        cost = merge_cost(
            int((1-self.fractions_[0]) * self.segments_[0]),
            [int((1-self.fractions_[j]) * self.segments_[j]) for j in range(1, count)]
        )
        self.n_ = sum([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.segments_ = []
        self.fractions_ = []
        return cost

class TieredBaselineMergePolicy:
    def __init__(self, merge_count: int = 10):
        self.max_merge_count_ = merge_count
        self.tiers_ = [BaselineMergePolicy(merge_count)]

    def add(self, n: int, f: float) -> float:
        cost = self.tiers_[0].add(n, f)
        n_tiers = len(self.tiers_)
        for i in range(n_tiers):
            if self.tiers_[i].should_promote():
                if i+1 == len(self.tiers_):
                    self.tiers_.append(BaselineMergePolicy(self.max_merge_count_))
                cost += self.tiers_[i+1].add(self.tiers_[i].n_, self.tiers_[i].f_)
                self.tiers_[i] = BaselineMergePolicy(self.max_merge_count_)
        return cost
    
    def merge_all(self) -> float:
        # The optimal way to do this is to merge all tiered graphs to the largest segment
        # but this makes the cost rather sensitive to initial conditions. Instead we just
        # do our normal tiered policy but relax the number of segments in the flush. This
        # is a better estimate of the average cost over all initial conditions.
        #
        # Note optimal strategy.
        # n = []
        # f = []
        # for i in range(len(self.tiers_)):
        #     if self.tiers_[i].n_ > 0:
        #         n.append(self.tiers_[i].n_)
        #         f.append(self.tiers_[i].f_)
        #     n.extend(self.tiers_[i].segments_)
        #     f.extend(self.tiers_[i].fractions_)
        #     self.tiers_[i] = BaselineMergePolicy(self.max_merge_count_, self.max_flush_count_)
        # self.tiers_[-1].n_ = sum([int((1-f[i]) * n[i]) for i in range(len(n))])
        # max = np.argmax([int((1-f[i]) * n[i]) for i in range(len(n))])
        # n[0], n[max] = n[max], n[0]
        # f[0], f[max] = f[max], f[0]
        # cost = merge_cost(
        #     int((1-f[max]) * n[max]),
        #     [int((1-f[i]) * n[i]) for i in range(len(n)) if i != max]
        # )
        cost = 0.0
        for i in range(len(self.tiers_) - 1):
            cost += self.tiers_[i].flush()
            cost += self.tiers_[i+1].add(self.tiers_[i].n_, self.tiers_[i].f_)
            self.tiers_[i] = BaselineMergePolicy(self.max_merge_count_)
        cost += self.tiers_[-1].flush()
        return cost

class MergeToLargestPolicy:
    def __init__(self, merge_count: int = 10, flush_count: int = 7):
        self.n_ = 0
        self.f_ = 0.0
        self.flush_count_ = 0
        self.segments_ = []
        self.fractions_ = []
        self.max_merge_count_ = merge_count
        self.max_flush_count_ = flush_count

    def should_promote(self) -> bool:
        return self.flush_count_ > self.max_flush_count_

    def add(self, n: int, f: float) -> float:
        self.segments_.append(n)
        self.fractions_.append(f)
        return self.flush() if len(self.segments_) == self.max_merge_count_ else 0.0

    def flush(self) -> float:
        count = len(self.segments_)
        if count == 0:
            return 0.0
        if self.n_ == 0:
            max = np.argmax([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
            self.segments_[0], self.segments_[max] = self.segments_[max], self.segments_[0]
            self.fractions_[0], self.fractions_[max] = self.fractions_[max], self.fractions_[0]
            cost = merge_cost(
                int((1-self.fractions_[0]) * self.segments_[0]),
                [int((1-self.fractions_[j]) * self.segments_[j]) for j in range(1, count)]
            )
        else:
            cost = merge_cost(
                self.n_,
                [int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)]
            )
        self.n_ += sum([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.segments_ = []
        self.fractions_ = []
        self.flush_count_ += 1
        return cost

class TieredMergeToLargestPolicy:
    def __init__(self, merge_count: int = 10, flush_count: int = 6):
        self.tiers_ = [MergeToLargestPolicy(merge_count, flush_count)]
        self.max_merge_count_ = merge_count
        self.max_flush_count_ = flush_count

    def add(self, n: int, f: float) -> float:
        cost = self.tiers_[0].add(n, f)
        n_tiers = len(self.tiers_)
        for i in range(n_tiers):
            if self.tiers_[i].should_promote():
                if i+1 == len(self.tiers_):
                    self.tiers_.append(
                        MergeToLargestPolicy(self.max_merge_count_, self.max_flush_count_)
                    )
                cost += self.tiers_[i+1].add(self.tiers_[i].n_, self.tiers_[i].f_)
                self.tiers_[i] = MergeToLargestPolicy(self.max_merge_count_, self.max_flush_count_)
        return cost

    def merge_all(self) -> float:
        cost = 0.0
        for i in range(len(self.tiers_) - 1):
            cost += self.tiers_[i].flush()
            cost += self.tiers_[i+1].add(self.tiers_[i].n_, self.tiers_[i].f_)
            self.tiers_[i] = MergeToLargestPolicy(self.max_merge_count_, self.max_flush_count_)
        cost += self.tiers_[-1].flush()
        return cost

def merge(policy, n: list, f: list) -> list:
    total_cost = 0.0
    for ni, fi in zip(n, f):
        total_cost += policy.add(ni, fi)
    total_cost += policy.merge_all()
    #print("Docs merged:", policy.tiers_[-1].n_)
    return total_cost

def trial():
    np.random.seed(42)

    flush_counts = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30]
    blc = []
    tlc = [[] for _ in flush_counts]
    for _ in tqdm(range(5), desc="Trials"):
        count = np.random.randint(400000, 600000)
        # Initial segment sizes.
        n = [int(np.random.normal(100, 4)) for _ in range(count)]
        # The delete fractions.
        f = [fi for fi in np.random.uniform(0.0, 0.1, count)]

        blc = merge(TieredBaselineMergePolicy(), n.copy(), f.copy())
        for i, flush_count in enumerate(flush_counts):
            tlc[i] = merge(TieredMergeToLargestPolicy(flush_count=flush_count), n.copy(), f.copy())

    print(f"Lucene baseline cost: {np.average(blc):.2f}")
    for i, flush_count in enumerate(flush_counts):
        print(f"Tiered merge to largest flush count {flush_count}: {np.average(np.array(tlc[i]) / blc):.2f}")

    print([np.average(np.array(tlc[i]) / np.array(blc)) for i in range(len(flush_counts))])

if __name__ == "__main__":
    trial()
