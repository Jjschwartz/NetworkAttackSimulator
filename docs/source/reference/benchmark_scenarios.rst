.. _`benchmark_scenarios`:

Benchmark Scenarios
===================

There are a number of existing scenarios that come with NASim. They cover a range of complexities and sizes and are intended to be used to help with benchmarking algorithms. Additionally, there are two flavours of existing scenarios: **static** and **generated**.

.. note:: For full list of benchmark scenarios see :ref:`all_benchmark_scenarios`.


Static Scenarios
----------------

These are scenarios that will be exactly the same every time they are loaded. They are defined in .yaml files in the `nasim/scenarios/benchmark/` directory.

Currently available are (from smallest to largest):

- `tiny` - 3 hosts, 3 subnets, 1 exploits
- `small` - 8 hosts, 4 subnets, 3 exploits
- `small-linear-two` - 8 hosts, 6 subnets, 3 exploits
- `standard` - 16 hosts, 5 subnets, 5 exploits
- `single-site` - 16 hosts, 1 subnet, 5 exploits
- `multi-site` - 16 hosts, 6 subnets, 5 exploits


Generated Scenarios
-------------------

These are scenarios that are generated from parameters and will change in certain ways depending on the random seed.

Currently available are (from smallest to largest):

-


.. _'all_benchmark_scenarios`:

All benchmark scenarios
-----------------------
