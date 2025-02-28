import math
import numpy as np

from tqdm.auto import tqdm

cost_of_copy = 0.5
cost_of_insert = 1.0
max_delete_fraction = 0.01

def merge_cost(n1: int, n2: list) -> float:
    n_s = sum(n2)
    cost = cost_of_copy * (n1 + n_s)
    graph_size = n1
    for n in n2:
        cost += cost_of_insert * n * math.log(graph_size)
        graph_size += n
    return cost


class LuceneBaselineTieredMergePolicy:
    """
    The Lucene tiered merge policy.
    """

    def __init__(self,
                 num_segments_per_tier: int = 10,
                 max_merge_at_once: int = 10,
                 floor_segment_size: int = 1000):
        self.num_segments_per_tier_ = num_segments_per_tier
        self.max_merge_at_once_ = max_merge_at_once
        self.floor_segment_size_ = floor_segment_size
        self.segments_ = []
        self.fractions_ = []
        self.total_cost_ = 0.0
        self.amplification_ = 0
        self.num_merges_ = 0
        self.total_docs_ = 0

    def cost(self) -> float:
        return self.total_cost_

    def amplification(self) -> float:
        return self.amplification_ / self.total_docs_

    def num_segments(self) -> int:
        return len(self.segments_)

    def num_merges(self) -> int:
        return self.num_merges_

    def query_cost(self) -> float:
        return sum(np.log([n for n in self.segments_])) / math.log(self.total_docs_)

    def segments(self) -> list[int]:
        return self.segments_

    def add(self, n: int, f: float) -> None:
        self.segments_.append(n)
        self.fractions_.append(f)
        self.total_docs_ += int((1 - f) * n)
        self.flush()

    def flush(self) -> tuple[float, float]:
        while self.should_merge():
            self.segments_.sort(key=lambda x: -x)
            a, b = self.find_merge()
            n = sum(int((1-self.fractions_[i]) * self.segments_[i]) for i in range(a, b))
            self.total_cost_ += merge_cost(
                int((1-self.fractions_[a]) * self.segments_[a]),
                [int((1-self.fractions_[i]) * self.segments_[i]) for i in range(a+1, b)]
            )
            self.amplification_ += n
            self.num_merges_ += 1
            del self.segments_[a:b]
            del self.fractions_[a:b]
            self.segments_.append(n)
            self.fractions_.append(
                np.random.uniform(0.0, max_delete_fraction) if max_delete_fraction > 0.0 else 0.0
            )

    def should_merge(self):
        # Compute the number of segments we expect to have in the index based on
        # floor_segment_size and num_segments_per_tier, given the total index document
        # count.
        remaining_index_size = self.total_docs_

        allowed_number_of_segments = 0
        num_segments_per_tier = min(self.num_segments_per_tier_, self.max_merge_at_once_)
        tier_size = self.floor_segment_size_
        while remaining_index_size > 0:
            num_segments_on_tier = min(math.ceil(remaining_index_size / tier_size),
                                       num_segments_per_tier)
            remaining_index_size -= num_segments_on_tier * tier_size
            allowed_number_of_segments += num_segments_on_tier
            tier_size *= num_segments_per_tier
        allowed_number_of_segments = max(allowed_number_of_segments, num_segments_per_tier)

        # if we have less segments than this threshold, we don't merge and accumulate more
        # segments, which may enable us to find less costly merges later on.
        return len(self.segments_) > allowed_number_of_segments

    def find_merge(self) -> tuple[int, int]:
        # Find merges that cost the least while achieving the most reclaim.
        a = np.argmin([self.merge_cost(i) for i in range(len(self.segments_) - 1)])
        b = min(a + self.max_merge_at_once_, len(self.segments_))
        return a, b

    def merge_cost(self, i: int) -> float:
        # bias towards merges of same-size segments:
        #    merge_score = first_segment_size / total_merge_size
        # now bias towards the smaller merges:
        #    merge_score *= total_merge_size^0.05
        # now bias towards reclaiming deletes:
        #    merge_score *= (1 - del_ratio)^2
        a = i
        b = min(i+self.max_merge_at_once_, len(self.segments_))
        total_merge_size = sum(self.segments_[a:b])
        first_segment_size = self.segments_[a]
        del_ratio = np.mean(self.fractions_[a:b])
        merge_cost = first_segment_size / total_merge_size
        merge_cost *= total_merge_size**0.05
        merge_cost *= (1.0 - del_ratio)**2
        return merge_cost


class LuceneLogDocMergePolicy:
    """
    A simple merge policy that gathers merge_count segments then merges merge_count - 1
    into the largest one.
    """

    def __init__(self, merge_count: int = 10, num_merges: int = 0):
        self.n_ = 0
        self.f_ = 0.0
        self.num_merges_ = num_merges
        self.segments_ = []
        self.fractions_ = []
        self.max_merge_count_ = merge_count

    def num_merges(self) -> int:
        return self.num_merges_

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
        wrote = sum(int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count))
        self.n_ = sum(int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count))
        self.f_ = np.random.uniform(0.0, max_delete_fraction) if max_delete_fraction > 0.0 else 0.0
        self.num_merges_ += 1
        self.segments_ = []
        self.fractions_ = []
        return cost, wrote

class LuceneTieredLogDocMergePolicy:
    """
    Is a tiered version of the baseline merge policy.

    For each tier, it gathers merge_count segments then merges merge_count - 1
    into the largest one. Then, it promotes the resulting largest segment to the
    next tier.

    In total we get log10(n) tiers and write amplification.
    """

    def __init__(self, merge_count: int = 10):
        self.max_merge_count_ = merge_count
        self.tiers_ = [LuceneLogDocMergePolicy(merge_count)]
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

    def num_merges(self) -> int:
        return sum(tier.num_merges() for tier in self.tiers_)

    def query_cost(self) -> float:
        cost = 0.0
        for tier in self.tiers_:
            cost += sum(np.log([n for n in tier.segments_]))
        return cost / math.log(self.total_docs_)

    def segments(self) -> list[int]:
        segments = []
        for tier in self.tiers_:
            segments.extend(tier.segments_)
        return segments

    def add(self, n: int, f: float) -> None:
        cost_i, wrote_i = self.tiers_[0].add(n, f)
        self.total_cost_ += cost_i
        self.amplification_ += wrote_i
        self.total_docs_ += int((1 - f) * n)
        n_tiers = len(self.tiers_)
        for i in range(n_tiers):
            if self.tiers_[i].should_promote():
                if i+1 == len(self.tiers_):
                    self.tiers_.append(LuceneLogDocMergePolicy(self.max_merge_count_))
                cost_i, wrote_i = self.tiers_[i+1].add(self.tiers_[i].n_, self.tiers_[i].f_)
                self.total_cost_ += cost_i
                self.amplification_ += wrote_i
                self.tiers_[i] = LuceneLogDocMergePolicy(
                    self.max_merge_count_, self.tiers_[i].num_merges()
                )

    def merge_all(self) -> None:
        n = []
        f = []
        for i in range(len(self.tiers_)):
            if self.tiers_[i].n_ > 0:
                n.append(self.tiers_[i].n_)
                f.append(self.tiers_[i].f_)
            n.extend(self.tiers_[i].segments_)
            f.extend(self.tiers_[i].fractions_)
            self.tiers_[i] = LuceneLogDocMergePolicy(
                self.max_merge_count_, self.tiers_[i].num_merges()
            )
        self.tiers_[-1].n_ = sum(int((1-f[i]) * n[i]) for i in range(len(n)))
        max = np.argmax([int((1-f[i]) * n[i]) for i in range(len(n))])
        n[0], n[max] = n[max], n[0]
        f[0], f[max] = f[max], f[0]
        self.total_cost_ += merge_cost(
            int((1-f[max]) * n[max]),
            [int((1-f[i]) * n[i]) for i in range(len(n)) if i != max]
        )
        self.amplification_ += sum([int((1-f[i]) * n[i]) for i in range(len(n))])


class MergeToLargestPolicy:
    def __init__(self, merge_count: int = 10, flush_count: int = 7, num_merges: int = 0):
        self.n_ = 0
        self.f_ = 0.0
        self.num_merges_ = num_merges
        self.flush_count_ = 0
        self.segments_ = []
        self.fractions_ = []
        self.max_merge_count_ = merge_count
        self.max_flush_count_ = flush_count

    def num_merges(self) -> int:
        return self.num_merges_

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
        wrote = self.n_ + sum(int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count))
        self.n_ += sum(int((1-self.fractions_[i]) * self.segments_[i]) for i in range(count))
        self.f_ = np.random.uniform(0.0, max_delete_fraction) if max_delete_fraction > 0.0 else 0.0
        self.num_merges_ += 1
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

    def num_merges(self) -> int:
        return sum(tier.num_merges() for tier in self.tiers_)

    def query_cost(self) -> float:
        cost = 0.0
        for tier in self.tiers_:
            cost += math.log(tier.n_) if tier.n_ > 0 else 0.0
            cost += sum(np.log([n for n in tier.segments_]))
        return cost / math.log(self.total_docs_)

    def segments(self) -> list[int]:
        segments = []
        for tier in self.tiers_:
            if tier.n_ > 0:
                segments.append(tier.n_)
            segments.extend(tier.segments_)
        return segments

    def add(self, n: int, f: float) -> None:
        cost_i, wrote_i = self.tiers_[0].add(n, f)
        self.total_cost_ += cost_i
        self.amplification_ += wrote_i
        self.total_docs_ += int((1 - f) * n)
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
                self.tiers_[i] = MergeToLargestPolicy(
                    self.tiers_[i].max_merge_count_,
                    self.tiers_[i].max_flush_count_,
                    self.tiers_[i].num_merges()
                )

    def merge_all(self) -> None:
        n = []
        f = []
        for i in range(len(self.tiers_)):
            if self.tiers_[i].n_ > 0:
                n.append(self.tiers_[i].n_)
                f.append(self.tiers_[i].f_)
            n.extend(self.tiers_[i].segments_)
            f.extend(self.tiers_[i].fractions_)
            self.tiers_[i] = MergeToLargestPolicy(
                self.max_merge_count_,
                self.max_flush_count_,
                self.tiers_[i].num_merges()
            )
        self.tiers_[-1].n_ = sum(int((1-f[i]) * n[i]) for i in range(len(n)))
        max = np.argmax([int((1-f[i]) * n[i]) for i in range(len(n))])
        n[0], n[max] = n[max], n[0]
        f[0], f[max] = f[max], f[0]
        self.total_cost_ += merge_cost(
            int((1-f[max]) * n[max]),
            [int((1-f[i]) * n[i]) for i in range(len(n)) if i != max]
        )
        self.amplification_ += sum(int((1-f[i]) * n[i]) for i in range(len(n)))

def merge(policy, n: list, f: list):
    for ni, fi in zip(n, f):
        policy.add(ni, fi)
    # Simulate force merge to a single segment.
    #policy.merge_all()
    return policy

def trial():
    np.random.seed(42)

    flush_counts = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    segments_per_tier = [2, 3, 4, 5, 6, 7]

    ldc = []
    lda = []
    lds = []
    ldm = []
    ldq = []
    stc = [[] for _ in segments_per_tier]
    sta = [[] for _ in segments_per_tier]
    sts = [[] for _ in segments_per_tier]
    stm = [[] for _ in segments_per_tier]
    stq = [[] for _ in segments_per_tier]
    tlc = [[] for _ in flush_counts]
    tla = [[] for _ in flush_counts]
    tls = [[] for _ in flush_counts]
    tlm = [[] for _ in flush_counts]
    tlq = [[] for _ in flush_counts]

    for _ in tqdm(range(100), desc="Trials"):

        # Random number of initial segments.
        count = np.random.randint(5000, 150000)

        # Initial segment sizes.
        n = [int(100 + np.random.exponential(100.0)) for _ in range(count)]
        # The delete fractions.
        f = [fi for fi in np.random.uniform(0.0, max_delete_fraction, count) if max_delete_fraction > 0.0]

        logdoc = merge(LuceneTieredLogDocMergePolicy(), n.copy(), f.copy())
        ldc.append(logdoc.cost())
        lda.append(logdoc.amplification())
        lds.append(logdoc.num_segments())
        ldm.append(logdoc.num_merges())
        ldq.append(logdoc.query_cost())

        for i, segments in enumerate(segments_per_tier):
            baseline = merge(LuceneBaselineTieredMergePolicy(segments), n.copy(), f.copy())
            stc[i].append(baseline.cost())
            sta[i].append(baseline.amplification())
            sts[i].append(baseline.num_segments())
            stm[i].append(baseline.num_merges())
            stq[i].append(baseline.query_cost())

        for i, flush_count in enumerate(flush_counts):
            largest = merge(TieredMergeToLargestPolicy(flush_count=flush_count), n.copy(), f.copy())
            tlc[i].append(largest.cost())
            tla[i].append(largest.amplification())
            tls[i].append(largest.num_segments())
            tlm[i].append(largest.num_merges())
            tlq[i].append(largest.query_cost())

    print("Lucene log doc")
    print(f"  Avg. cost: {np.average(ldc) / 1e6:.2f},",
          f"amplification: {np.average(lda):.2f},",
          f"segments: {np.average(lds):.2f},",
          f"merges: {np.average(ldm):.2f},",
          f"query cost: {np.average(ldq):.2f}")

    print("Lucene baseline")
    for i, segments in enumerate(segments_per_tier):
        print(f"  segments per tier {segments} avg. cost: {np.average(stc[i]) / 1e6:.2f},",
              f"amplification: {np.average(sta[i]):.2f},",
              f"segments: {np.average(sts[i]):.2f},",
              f"merges: {np.average(stm[i]):.2f},",
              f"query cost: {np.average(stq[i]):.2f}")

    print("Tiered merge to largest")
    for i, flush_count in enumerate(flush_counts):
        print(f"  flush count {flush_count} avg cost: {np.average(np.array(tlc[i]))/1e6:.2f},",
              f"amplification: {np.average(np.array(tla[i])):.2f},",
              f"segments: {np.average(np.array(tls[i])):.2f},",
              f"merges: {np.average(np.array(tlm[i])):.2f},",
              f"query cost: {np.average(np.array(tlq[i])):.2f}")

if __name__ == "__main__":
    trial()
