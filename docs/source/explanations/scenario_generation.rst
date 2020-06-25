.. _scenario_generation_explanation:

Scenario Generation Explanation
===============================

Generating the scenarios involves a number of design decisions that strongly determine the form of the network being generated. This document aims to explain some of the more technical details of generating the scenarios when using the :ref:`scenario_generator` class.

The scenario generator is based heavily on prior work, specifically:

- `Sarraute, Carlos, Olivier Buffet, and Jörg Hoffmann. "POMDPs make better hackers: Accounting for uncertainty in penetration testing." Twenty-Sixth AAAI Conference on Artificial Intelligence. 2012. <https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPaper/4996>`_
- `Speicher, Patrick, et al. "Towards Automated Network Mitigation Analysis (extended)." arXiv preprint arXiv:1705.05088 (2017). <https://arxiv.org/abs/1705.05088>`_

Network Topology
----------------

Description to come. Till then we recommend reading the papers linked above, especially the appendix of Speicher et al (2017).

.. _correlated_configurations:

Correlated Configurations
-------------------------

When generating a scenario with ``uniform=False`` the scenario will be generated with host configurations being correlated. This means that rather than the OS and services it is running being chosen uniformly at random from the available OSs and services, they are chosen randomly with increased probability given to OSs and services that are being run by other hosts whose configuration was generated earlier.


Specifically, the distribution of configurations of each host in the network are generated using a Nested Dirichlet Process, so that across the network hosts will have corelated configurations (i.e. certain services/configurations will be more common across hosts on the network). The correlation can be controlled using three parameters: ``alpha_H``, ``alpha_V``, and ``lambda_V``.

``alpha_H`` and ``alpha_V`` control the degree of correlation, with lower values leading to greater correlation.

``lambda_V`` controls the average number of services running per host, with higher values will mean more services (so more vulnerable) hosts on average.

All three parameters must have a positive value, with the defaults being ``alpha_H=2.0``, ``alpha_V=2.0``, and ``lambda_V=1.0``, which tends to generate networks with fairly correlated configurations where hosts have only a single vulnerability on average.


.. _generated_exploit_probs:

Generated Exploit Probabilities
-------------------------------

Success probabilities of each exploit are determined based on the value of the ``exploit_probs`` argument, as follows:

- ``exploit_probs=None`` - probabilities generated randomly from uniform distribution over the interval (0, 1).
- ``exploit_probs=float`` - probability of each exploit is set to the float value, which must be a valid probability.
- ``exploit_probs=list[float]`` - probability of each exploit is set to corresponding float value in list. This requires that the length of the list matches the number of exploits as specified by the ``num_exploits`` argument.
- ``exploit_probs="mixed"`` - probabilities chosen from a set distribution which is based on the `CVSS attack complexity <https://www.first.org/cvss/v2/guide>`_ distribution of `top 10 vulnerabilities in 2017 <https://go.recordedfuture.com/hubfs/reports/cta-2018-0327.pdf>`_. Specifically, exploit probabilities are chosen from [0.3, 0.6, 0.9] which correspond to high, medium and low attack complexity, respectively, with probabilities [0.2, 0.4, 0.4].

For deterministic exploits set ``exploit_probs=1.0``.


Firewall
--------

The firewall restricts which services can be communicated with between hosts on different subnets. This is mostly done by selecting services at random to block between each subnet, with some contraints.

Firstly, there exists no firewall between subnets in the user zone. So communication between hosts on different user subnets is allowed for all services.

Secondly, the number of services blocked is controlled by the ``restrictiveness`` parameter. This controls the number of services to block between zones (i.e. between the internet, DMZ, sensitive, and user zones).

Thirdly, to ensure that the goal can be reached, traffic from at least one service running on each subnet will be allowed between each zone. This may mean more services will be allowed than restrictiveness parameter.
