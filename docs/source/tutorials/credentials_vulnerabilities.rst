.. _`cred_vul_tute`:

Credentials and Vulnerabilities
===============================

The following additions are optional and do not need to be defined in costum scenario.yaml files. Nevertheless, internal structures had to be modified which are 'visible' even if not manually defined.


Idea
----

To better reflect real world complexity, credentials and vulnerabilities are introduced to NASim.

These modifications are implemented with the idea to train RL-agents in more realistic enviroments.


Credentials
-----------

Credentials are implemented as single digits from **1** to **9**, each describing a full credential pair of, theoretically, a username and password. **0** is in meaning equivilant to ``None``.


Host
^^^^

Hosts can now be further described by ``credentials_needed: int`` and ``credentials_tofind: int``.

.. code-block:: yaml

    host_configurations:
        (1, 0):
            os: linux
            services: [ssh]
            processes: [tomcat]
            firewall:
                (3, 0): [ssh]
            value: 0
            credentials_needed: 0
            credentials_tofind: 12

``credentials_needed: int`` describes the credentials needed to perform a succesful exploit. If 0 or not defined no credentials are needed.

``credentials_tofind: int`` describes the credentials which can be found by wiretapping from a specific host.
This is a concatination of integer, meaning `123` is equal to its permutations for example `213`.


PrivelegeEscalation
^^^^^^^^^^^^^^^^^^^

Under the pretext of post-exploitation, **PrivelegeEscalation()** can be further extended with ``credentials_tofind: int``. 

.. code-block:: yaml

    privilege_escalation:
        pe_gain_Credentails_from_Shadow:
            process: shadow
            os: Rasbian
            prob: 1.0
            cost: 3
            access: root
            credentials_tofind: 3

``credentials_tofind: int`` describes the credentials which are found by exploiting / attacking a process on the specific host.


Wiretapping
^^^^^^^^^^^


**Wiretapping()** is a new action with the goal to 'listen' from a targeted node. A compromised host can tap to reveal the credentials hold in ``cred_tofind: int``.

The cost ``wiretapping_cost: int`` can be set in the scenario.yaml.



Vulnerabilities
---------------

A vulnerability is a weakness or flaw on a host which can be exploited.

It is implemented as a ``string`` and with its introduction both **Host** and **Exploit** get additional options and a new action **VulScan()** is implemented.


Host
^^^^

.. code-block:: yaml

    host_configurations:
        (1, 0):
            os: Rasbian
            services: [openssh, samba] 
            vul: [CVE20072447]
            processes: [shadow]
            credentials_needed: 0
            credentials_tofind: 21

``vul: list[str]`` describes the existing vulnerabilities on a host.


Exploit
^^^^^^^

.. code-block:: yaml

    e_samba:
        service: samba
        os: None
        vul: CVE20072447
        prob: 1.0
        cost: 3
        access: root

``vul: str`` describes the exploited vulnerability. This `vul` has to exist on the targeted host for the exploit to succeed.
Can further be *None* or not defined if not needed.


VulScan
^^^^^^^

Similiar to **OSScan()** it is an action performed on a host returning its vulnerabilites ``vul: list[str]``.

The cost ``vul_scan_cost: int`` can be set in the scenario.yaml.


Scenarios
^^^^^^^^^

Some scenarios are added featuring credentials and vulnerabilities, namely *tiny-wire*, *tiny-post*, *tiny-cred*, *small-wire* and *small-post*.

In addition, some scenarios from previous work got an equivalant *.yaml* with credentials and vulnerabilities specifically defined. (names ending on *-cwp*) 


Information
-----------

If no credentials or vulnerabilities are used, following properties are "visible":

- **HostVector** will still hold ``cred_tofind = 0`` and ``cred_found = 0``.

- **Exploit** will need ``cred_needed = 0`` and ``vul = None``

- **Host** will have ``vul = []``, ``cred_tofind = 0`` and ``cred_needed = 0``


These modifications are based on the Bachelor's Thesis of Norman Becker

Contributions by Georg Blum @Class1G 

 **Evaluation of Reinforcement Learning for Autonomous Penetration Testing using A3C, Q-learning and DQN** Norman Becker et al. *2024*
https://arxiv.org/abs/2407.15656

-- Copyright © Norman Becker --