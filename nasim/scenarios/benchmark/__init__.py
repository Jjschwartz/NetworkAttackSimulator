import os.path as osp

from .generated import AVAIL_GEN_BENCHMARKS

BENCHMARK_DIR = osp.dirname(osp.abspath(__file__))

AVAIL_STATIC_BENCHMARKS = {
    "multi-site": {
        "file": osp.join(BENCHMARK_DIR, "multi-site.yaml"),
        "max_score": 17},
    "single-site": {
        "file": osp.join(BENCHMARK_DIR, "single-site.yaml"),
        "max_score": 17},
    "small-linear-two": {
        "file": osp.join(BENCHMARK_DIR, "small-linear-two.yaml"),
        "max_score": 17},
    "small": {
        "file": osp.join(BENCHMARK_DIR, "small.yaml"),
        "max_score": 16},
    "standard": {
        "file": osp.join(BENCHMARK_DIR, "standard.yaml"),
        "max_score": 17},
    "tiny": {
        "file": osp.join(BENCHMARK_DIR, "tiny.yaml"),
        "max_score": 17}
}

AVAIL_BENCHMARKS = list(AVAIL_STATIC_BENCHMARKS.keys()) \
                    + list(AVAIL_GEN_BENCHMARKS.keys())
