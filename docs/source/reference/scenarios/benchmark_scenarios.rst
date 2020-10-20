.. _benchmark_scenarios:

Benchmark Scenarios
===================

There are a number of existing scenarios that come with NASim. They cover a range of complexities and sizes and are intended to be used to help with benchmarking algorithms. Additionally, there are two flavours of existing scenarios: **static** and **generated**.

.. note:: For full list of benchmark scenarios see :ref:`all_benchmark_scenarios`.

**Static** scenarios are predefined and will be exactly the same every time they are loaded. They are defined in .yaml files in the `nasim/scenarios/benchmark/` directory.

**Generated** are scenario generated using the :ref:`scenario_generator` based on some parameters. While certain features of the each scenario will remain constant between generations (e.g. number of hosts, services, exploits), other features may change (e.g. specific host configurations, firewall settings, exploit probabilities) depending on the random seed.


.. _all_benchmark_scenarios:

All benchmark scenarios
-----------------------

The following table provides details of each benchmark scenario currently available in NASim.

.. csv-table:: NASim Benchmark scenarios
   :file: benchmark_scenarios_table.csv
   :header-rows: 1


The number of actions is calculated as *Hosts X (Exploits + PrivEscs + 4)*. The +4 is for the 4 scans available for each host (OSScan, ServiceScan, ProcessScan, and SubnetScan).

The number of states is calculated as *Hosts X 2^(3 + OS + Services) X 3 *. Here the first 3 comes from the *compromised*, *reachable* and *discovered* features of the state and the base of 2 is due to all state features being boolean (present/absent). The second 3 comes from the number of possible access levels possible on a host.

The table below provides mean steps to reach the goal and reward (+/- stdev) for a uniform random agent, with scores averaged over 100 runs.

.. csv-table:: NASim Benchmark scenarios Agent scores
   :file: benchmark_scenarios_agent_scores.csv
   :header-rows: 2


Notes on the scenarios
----------------------

The *tiny*, *small*, *medium*, *large*, and *huge* (and their generated versions) are all based on the network scenarios first used by:

- `Sarraute, Carlos, Olivier Buffet, and Jörg Hoffmann. "POMDPs make better hackers: Accounting for uncertainty in penetration testing." Twenty-Sixth AAAI Conference on Artificial Intelligence. 2012. <https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPaper/4996>`_
- `Speicher, Patrick, et al. "Towards Automated Network Mitigation Analysis (extended)." arXiv preprint arXiv:1705.05088 (2017). <https://arxiv.org/abs/1705.05088>`_

The *pocp-1-gen* and *pocp-2-gen* scenarios are based on the work by:

- `Shmaryahu, D., Shani, G., Hoffmann, J., & Steinmetz, M. (2018, June). Simulated penetration testing as contingent planning. In Twenty-Eighth International Conference on Automated Planning and Scheduling. <https://www.aaai.org/ocs/index.php/ICAPS/ICAPS18/paper/viewPaper/17766>`_

The other scenarios were made up by author after looking at some random google images of network layouts, and playing around with different interesting network topologies.
