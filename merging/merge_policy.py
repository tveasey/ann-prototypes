import math
import numpy as np

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

cost_of_copy = 0.5
cost_of_insert = 1.0
max_delete_fraction = 0.1

def merge_cost(n1: int, n2: list) -> float:
    n_s = np.sum(n2)
    cost = cost_of_copy * (n1 + n_s)
    graph_size = n1
    for n in n2:
        cost += cost_of_insert * n * math.log(graph_size)
        graph_size += n
    return cost

class BaselineMergePolicy:
    def __init__(self, merge_count: int = 10):
        self.n_ = 0
        self.f_ = 0.0
        self.segments_ = []
        self.fractions_ = []
        self.max_merge_count_ = merge_count

    def should_promote(self) -> bool:
        return self.n_ > 0

    def add(self, n: int, f: float) -> tuple[float, float]:
        self.segments_.append(n)
        self.fractions_.append(f)
        if len(self.segments_) == self.max_merge_count_:
            return self.flush()
        return 0.0, 0.0

    def flush(self) -> tuple[float, float]:
        count = len(self.segments_)
        if count == 0:
            return 0.0, 0.0
        max = np.argmax([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.segments_[0], self.segments_[max] = self.segments_[max], self.segments_[0]
        self.fractions_[0], self.fractions_[max] = self.fractions_[max], self.fractions_[0]
        cost = merge_cost(
            int((1-self.fractions_[0]) * self.segments_[0]),
            [int((1-self.fractions_[i]) * self.segments_[i]) for i in range(1, count)]
        )
        wrote = sum([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.n_ = sum([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.f_ = np.random.uniform(0.0, max_delete_fraction)
        self.segments_ = []
        self.fractions_ = []
        return cost, wrote

class TieredBaselineMergePolicy:
    def __init__(self, merge_count: int = 10):
        self.max_merge_count_ = merge_count
        self.tiers_ = [BaselineMergePolicy(merge_count)]
        self.total_cost_ = 0.0
        self.amplification_ = 0
        self.total_docs_ = 0

    def cost(self) -> float:
        return self.total_cost_

    def amplification(self) -> float:
        return self.amplification_ / self.total_docs_

    def num_segments(self) -> int:
        n = 0
        for tier in self.tiers_:
            n += len(tier.segments_)
        return n
    
    def query_cost(self) -> float:
        cost = 0.0
        for tier in self.tiers_:
            cost += np.sum(np.log([n for n in tier.segments_]))
        return cost / math.log(self.total_docs_)

    def add(self, n: int, f: float) -> None:
        cost_i, wrote_i = self.tiers_[0].add(n, f)
        self.total_cost_ += cost_i
        self.amplification_ += wrote_i
        self.total_docs_ += n
        n_tiers = len(self.tiers_)
        for i in range(n_tiers):
            if self.tiers_[i].should_promote():
                if i+1 == len(self.tiers_):
                    self.tiers_.append(BaselineMergePolicy(self.max_merge_count_))
                cost_i, wrote_i = self.tiers_[i+1].add(self.tiers_[i].n_, self.tiers_[i].f_)
                self.total_cost_ += cost_i
                self.amplification_ += wrote_i
                self.tiers_[i] = BaselineMergePolicy(self.max_merge_count_)
    
    def merge_all(self) -> None:
        n = []
        f = []
        for i in range(len(self.tiers_)):
            if self.tiers_[i].n_ > 0:
                n.append(self.tiers_[i].n_)
                f.append(self.tiers_[i].f_)
            n.extend(self.tiers_[i].segments_)
            f.extend(self.tiers_[i].fractions_)
            self.tiers_[i] = BaselineMergePolicy(self.max_merge_count_, self.max_flush_count_)
        self.tiers_[-1].n_ = sum([int((1-f[i]) * n[i]) for i in range(len(n))])
        max = np.argmax([int((1-f[i]) * n[i]) for i in range(len(n))])
        n[0], n[max] = n[max], n[0]
        f[0], f[max] = f[max], f[0]
        self.total_cost_ += merge_cost(
            int((1-f[max]) * n[max]),
            [int((1-f[i]) * n[i]) for i in range(len(n)) if i != max]
        )
        self.amplification_ += sum([int((1-f[i]) * n[i]) for i in range(len(n))])

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

    def add(self, n: int, f: float) -> tuple[float, float]:
        self.segments_.append(n)
        self.fractions_.append(f)
        if len(self.segments_) == self.max_merge_count_:
            return self.flush()
        return 0.0, 0.0

    def flush(self) -> tuple[float, float]:
        count = len(self.segments_)
        if count == 0:
            return 0.0, 0.0
        if self.n_ == 0:
            max = np.argmax([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
            self.segments_[0], self.segments_[max] = self.segments_[max], self.segments_[0]
            self.fractions_[0], self.fractions_[max] = self.fractions_[max], self.fractions_[0]
            cost = merge_cost(
                int((1-self.fractions_[0]) * self.segments_[0]),
                [int((1-self.fractions_[i]) * self.segments_[i]) for i in range(1, count)]
            )
        else:
            cost = merge_cost(
                self.n_,
                [int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)]
            )
        wrote = self.n_ + sum([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.n_ += sum([int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count)])
        self.f_ = np.random.uniform(0.0, max_delete_fraction)
        self.segments_ = []
        self.fractions_ = []
        self.flush_count_ += 1
        return cost, wrote

class TieredMergeToLargestPolicy:
    def __init__(self, merge_count: int = 10, flush_count: int = 8):
        self.tiers_ = [MergeToLargestPolicy(merge_count, flush_count)]
        self.max_merge_count_ = merge_count
        self.max_flush_count_ = flush_count
        self.total_cost_ = 0.0
        self.amplification_ = 0
        self.total_docs_ = 0

    def cost(self) -> float:
        return self.total_cost_

    def amplification(self) -> float:
        return self.amplification_ / self.total_docs_
    
    def num_segments(self) -> int:
        n = 0
        for tier in self.tiers_:
            n += 1 if tier.n_ > 0 else 0
            n += len(tier.segments_)
        return n
    
    def query_cost(self) -> float:
        cost = 0.0
        for tier in self.tiers_:
            cost += math.log(tier.n_) if tier.n_ > 0 else 0.0
            cost += np.sum(np.log([n for n in tier.segments_]))
        return cost / math.log(self.total_docs_)

    def add(self, n: int, f: float) -> None:
        cost_i, wrote_i = self.tiers_[0].add(n, f)
        self.total_cost_ += cost_i
        self.amplification_ += wrote_i
        self.total_docs_ += n
        n_tiers = len(self.tiers_)
        for i in range(n_tiers):
            if self.tiers_[i].should_promote():
                if i+1 == len(self.tiers_):
                    self.tiers_.append(
                        MergeToLargestPolicy(self.max_merge_count_, self.max_flush_count_)
                    )
                cost_i, wrote_i = self.tiers_[i+1].add(self.tiers_[i].n_, self.tiers_[i].f_)
                self.total_cost_ += cost_i
                self.amplification_ += wrote_i
                self.tiers_[i] = MergeToLargestPolicy(self.max_merge_count_, self.max_flush_count_)

    def merge_all(self) -> None:
        n = []
        f = []
        for i in range(len(self.tiers_)):
            if self.tiers_[i].n_ > 0:
                n.append(self.tiers_[i].n_)
                f.append(self.tiers_[i].f_)
            n.extend(self.tiers_[i].segments_)
            f.extend(self.tiers_[i].fractions_)
            self.tiers_[i] = BaselineMergePolicy(self.max_merge_count_, self.max_flush_count_)
        self.tiers_[-1].n_ = sum([int((1-f[i]) * n[i]) for i in range(len(n))])
        max = np.argmax([int((1-f[i]) * n[i]) for i in range(len(n))])
        n[0], n[max] = n[max], n[0]
        f[0], f[max] = f[max], f[0]
        self.total_cost_ += merge_cost(
            int((1-f[max]) * n[max]),
            [int((1-f[i]) * n[i]) for i in range(len(n)) if i != max]
        )
        self.amplification_ += sum([int((1-f[i]) * n[i]) for i in range(len(n))])

def merge(policy, n: list, f: list):
    for ni, fi in zip(n, f):
        policy.add(ni, fi)
    # Simulate force merge to a single segment.
    #policy.merge_all()
    return policy

def trial():
    np.random.seed(42)

    flush_counts = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30]
    blc = []
    bla = []
    bls = []
    blq = []
    tlc = [[] for _ in flush_counts]
    tla = [[] for _ in flush_counts]
    tls = [[] for _ in flush_counts]
    tlq = [[] for _ in flush_counts]

    for _ in tqdm(range(500), desc="Trials"):
        count = np.random.randint(5000, 150000)
        # Initial segment sizes.
        n = [int(100 + np.random.exponential(100.0)) for _ in range(count)]
        # The delete fractions.
        f = [fi for fi in np.random.uniform(0.0, max_delete_fraction, count)]

        baseline = merge(TieredBaselineMergePolicy(), n.copy(), f.copy())
        blc.append(baseline.cost())
        bla.append(baseline.amplification())
        bls.append(baseline.num_segments())
        blq.append(baseline.query_cost())

        for i, flush_count in enumerate(flush_counts):
            largest = merge(TieredMergeToLargestPolicy(flush_count=flush_count), n.copy(), f.copy())
            tlc[i].append(largest.cost())
            tla[i].append(largest.amplification())
            tls[i].append(largest.num_segments())
            tlq[i].append(largest.query_cost())

    print("Lucene baseline")
    print(f"  Avg. cost: {np.average(blc):.2f}",
          f"amplification: {np.average(bla):.2f}",
          f"segments: {np.average(bls):.2f}",
          f"query cost: {np.average(blq):.2f}")

    print("Tiered merge to largest")
    cost_vs_baseline = []
    amplification_vs_baseline = []
    segments_vs_baseline = []
    query_cost_vs_baseline = []
    for i, flush_count in enumerate(flush_counts):
        cost_vs_baseline.append(np.average(np.array(tlc[i]) / np.array(blc)))
        amplification_vs_baseline.append(np.average(np.array(tla[i]) / np.array(bla)))
        segments_vs_baseline.append(np.average(np.array(tls[i]) / np.array(bls)))
        query_cost_vs_baseline.append(np.average(np.array(tlq[i]) / np.array(blq)))
        print(f"  flush count {flush_count} avg cost: {cost_vs_baseline[-1]:.2f}",
              f"amplification: {amplification_vs_baseline[-1]:.2f}",
              f"segments: {segments_vs_baseline[-1]:.2f}",
              f"query cost: {query_cost_vs_baseline[-1]:.2f}")

    plt.plot(flush_counts, cost_vs_baseline, label="Cost")
    plt.plot(flush_counts, amplification_vs_baseline, label="Amplification")
    plt.plot(flush_counts, segments_vs_baseline, label="Segments")
    plt.plot(flush_counts, query_cost_vs_baseline, label="Query cost")
    plt.xlabel("Flush count")
    plt.ylabel("Ratio")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    trial()
