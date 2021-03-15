.. _`creating_scenarios_tute`:

Creating Custom Scenarios
=========================

With NASim it is possible to use custom scenarios defined in a valid YAML file. In this tutorial we will cover how to create and run you own custom scenario.

.. _'defining_custom_yaml':

Defining a custom scenario using YAML
-------------------------------------

Before we dive into writing a new custom YAML scenario it is worth having a look at some examples. NASim comes with a number of benchmark YAML scenarios which can be found in the ``nasim/scenarios/benchmark`` directory (or view on github `here <https://github.com/Jjschwartz/NetworkAttackSimulator/tree/master/nasim/scenarios/benchmark>`_). For this tutorial we will be using the ``tiny.yaml`` scenario as an example.

A custom scenarios in NASim requires definining components: the network and the pen-tester.


Defining the network
^^^^^^^^^^^^^^^^^^^^

The network is defined by the following sections:

   1. **subnets**: size of each subnet in network
   2. **topology**: an adjacency matrix defining which subnets are connected
   3. **os**: names of available operating systems on network
   4. **services**: names of available services on network
   5. **processes**: names of available processes on network
   6. **hosts**: a dictionary of hosts on the network and their configurations
   7. **firewall**: definition of the subnet firewalls


Subnets
"""""""

This property defines the number of subnets on the network and the size of each. It is simply defined as an ordered list of integers. The address of the first subnet in the list is *1*, the second subnet is *2*, and so on. The address of *0* is reserved for the "internet" subnet (see topology section below). For example, the ``tiny`` network contains 3 subnets all of size 1:

.. code-block:: yaml

   subnets: [1, 1, 1]

   # or alternatively

   subnets:
     - 1
     - 1
     - 1


Topology
""""""""

The topology is defined by an adjacency matrix with a row and column for every subnet in the network along with an additional row and column designating the "internet" subnet, i.e. connection to outside of the network. The first row and column is reserved for the "internet" subnet. A connection between subnets is indicated with a ``1`` while not connection is indicated with a ``0``. Note that we assume that connections are symmetric and that a subnet is connected with itself.

For the ``tiny`` network, subnet *1* is a public subnet so is connected to the internet, indicated by a ``1`` in row 1, column 2 and row 2, column 1. Subnet *1* is also connected with subnets *2* and *3*, indicated by ``1`` in relevant cells, meanwhile subnets *2* and *3* are private and not connected directly to the internet, indicated by the ``0`` values.

.. code-block:: yaml

   topology: [[ 1, 1, 0, 0],
              [ 1, 1, 1, 1],
              [ 0, 1, 1, 1],
              [ 0, 1, 1, 1]]



OS, services, processes
"""""""""""""""""""""""

Similar to how we defined the subnet list, the **os**, **services** and **processes** are defined by a simple list. The names of any of the items in each list can be anything, but note that they will be used for validating the host configurations, exploits, etc, so just need to match-up with those values as desired.

Continuing our example, the ``tiny`` scenario includes one OS: *linux*, one service: *ssh*, and one process: *tomcat*:

.. code-block:: yaml

   os:
     - linux
   services:
     - ssh
   processes:
     - tomcat


Host Configurations
"""""""""""""""""""

The host configuration section is a mapping from host address to their configuration, where the address is a ``(subnet number, host number)`` tuple and the configuration must include the hosts OS, services running, processes running, and optional host firewall settings.

There are a few things to note when defining a host:

   1. The number of hosts defined for each subnet needs to match the size of each subnet
   2. Host addresses within a subnet must start from ``0`` and count up from there (i.e. three hosts in subnet *1* would have addresses ``(1, 0)``, ``(1, 1)``, and ``(1, 2)``)
   3. The names of any OS, service, and process must match values provided in the **os**, **services** and **processes** sections of the YAML file.
   4. Each host must have an OS and at least one service running. It is okay for hosts to have no processes running (which can be indicated using an empty list ``[]``).

**Host firewalls** are defined as a mapping from host address to the list of services to deny from that host. Host addresses must be a valid address of a host in the network and any services must also match services defined in the services section. Finally, if a host address is not part of the firewall then it is assumed all traffic is allowed from that host, at the host level (it may still be blocked by subnet firewall).

**Host Value** is the optional value the agent will recieve when compromising the host. Unlike for the *sensitive_hosts* section this value can be negative as well as zero and positive. This makes it possible to set additional host specific rewards or penalties, for example setting a negative reward for a 'honeypot' host on the network. A couple of things to note:

  1. Host value is optional and will default to 0.
  2. For any *sensitive hosts* the value must either not be specified or it must match the value specified in the *sensitive_hosts* section of the file.
  3. Same as for *sensitive hosts*,  agent will only recieve the value as a reward when they compromise the host.

Here is the example host configurations section for the ``tiny`` scenario, where a host firewall and is defined only for host ``(1, 0)`` and the host ``(1, 0)`` has a value of ``0`` (noting we could leave value unspecified in this case for the same result, we include it here as an example):

.. code-block:: yaml

   host_configurations:
     (1, 0):
       os: linux
       services: [ssh]
       processes: [tomcat]
       # which services to deny between individual hosts
       firewall:
         (3, 0): [ssh]
       value: 0
     (2, 0):
       os: linux
       services: [ssh]
       processes: [tomcat]
       firewall:
         (1, 0): [ssh]
     (3, 0):
       os: linux
       services: [ssh]
       processes: [tomcat]


Firewall
""""""""

The final section for defining the network is the firewall, which is defined as a mapping from ``(subnet number, subnet number)`` tuples to list of services to allow. Some things to note about defining firewalls:

   1. A firewall rule can only be defined between subnets that are connected in the topology adjacency matrix.
   2. Each rule defines which services are allowed in a single direction, from the first subnet in the tuple to the second subnet in the tuple (i.e. (source subnet, destination subnet))
   3. An empty list means all traffic will be blocked from source to destination

Here is the firewall definition for the ``tiny`` scenario where SSH traffic is allowed between all subnets, except from subnet 1 to 0 and from 1 to 2.

.. code-block:: yaml

    # two rows for each connection between subnets as defined by topology
    # one for each direction of connection
    # lists which services to allow
    firewall:
      (0, 1): [ssh]
      (1, 0): []
      (1, 2): []
      (2, 1): [ssh]
      (1, 3): [ssh]
      (3, 1): [ssh]
      (2, 3): [ssh]
      (3, 2): [ssh]


And with that we have covered everything needed to define the scenario's network. Next up is defining the pen-tester.


Defining the pen-tester
^^^^^^^^^^^^^^^^^^^^^^^

The pen-tester is defined by these sections:

   1. **sensitive_hosts**: a dictionary containing the address of sensitive/target hosts and their value
   2. **exploits**: a dictionary of exploits
   3. **privilege_escalation**: a dictionary of privilege escalation actions
   4. **os_scan_cost**: cost of using OS scan
   5. **service_scan_cost**: cost of using service scan
   6. **process_scan_cost**: cost of using process scan
   7. **subnet_scan_cost**: cost of using subnet scan
   8. **step_limit**: the maximum number of actions pen-tester can perform in a single episode


Sensitive hosts
"""""""""""""""

This section specifies the addresses and values of the target hosts in the network. When the pen-tester gains root access on these hosts they will recieve the specified value as a reward. The *sensitive_hosts* section is a dictionary where the entries are address, value pairs. Where the address is a ``(subnet number, host number)`` tuple and the value is a non-negative float or integer.

In the ``tiny`` scenario the pen-tester is aiming to get root access on the hosts ``(2, 0)`` and ``(3, 0)``, both of which have a value of 100:

.. code-block:: yaml

    sensitive_hosts:
      (2, 0): 100
      (3, 0): 100


Exploits
""""""""

The exploits section is a dictionary which maps exploit names to exploit definitions. Every scenario requires at least on exploit. An exploit definition is a dictionary which must include the following entries:

  1. **service**: the name of the service the exploit targets.

     - Note, the value must match the name of a service defined in the **services** section of the network definition.

  2. **os**: the name of the operating system the exploit targets or ``none`` if the exploit works on all OSs.

     - If the value is not ``none`` it must match the name of an OS defined in the **os** section of the network definition

  3. **prob**: the probability that the exploit succeeds given all preconditions are met (i.e. target host is discovered and reachable, and the host is running targete service and OS)
  4. **cost**: the cost of performing the action. This should be a non-negative int or float and can represent the cost of the action in any sense desired (financial, time, traffic generated, etc)
  5. **access**: the resulting access the pen-tester will get on the target host if the exploit succeeds. This can be either *user* or *root*.


The name of the exploits can be anything you desire, so long as they are immutable and hashable (i.e. strings, ints, tuples) and unique.

The ``tiny`` example scenario has only a single exploit ``e_ssh`` which targets the SSH service running on linux hosts, has a cost of 1 and results in user level access:

.. code-block:: yaml

    exploits:
      e_ssh:
        service: ssh
        os: linux
        prob: 0.8
        cost: 1
        access: user


Privilege Escalation
"""""""""""""""""""""

Similar to the exploits section, the privilege escalation section is a dictionary which maps privilege escalation action names to their definitions. A privilege escalation action definition is a dictionary which must include the following entries:

  1. **process**: the name of the process the action targets.

     - The value must match the name of a process defined in the **processes** section of the network definition.

  2. **os**: the name of the operating system the action targets or ``none`` if the exploit works on all OSs.

     - If the value is not ``none`` it must match the name of an OS defined in the **os** section of the network definition.

  3. **prob**: the probability that the action succeeds given all preconditions are met (i.e. pen-tester has access to target host, and the host is running target process and OS)
  4. **cost**: the cost of performing the action. This should be a non-negative int or float and can represent the cost of the action in any sense desired (financial, time, traffic generated, etc)
  5. **access**: the resulting access the pen-tester will get on the target host if the action succeeds. This can be either *user* or *root*.

Similar to  exploits, the name of each privilege exploit action can be anything you desire, so long as they are immutable and hashable (i.e. strings, ints, tuples) and unique.

.. note:: It is not required that a scenario has any privilege escalation actions defined. In this case define the privilege escalation section to be empty: ``privilege_escalation: {}``.

          Note however that you will need to make sure that it is possible to get root access on the sensitive hosts via using only exploits, otherwise the pen-tester will never be able to reach the goal.

The ``tiny`` example scenario has a single privilege escalation action ``pe_tomcat`` which targets the tomcat process running on linux hosts, has a cost of 1 and results in root level access:

.. code-block:: yaml

    privilege_escalation:
      pe_tomcat:
        process: tomcat
        os: linux
        prob: 1.0
        cost: 1
        access: root


Scan costs
""""""""""

Each scan must have an associated non-negative cost associated with it. This cost can represent whatever you wish and will be factored in to the reward the agent recieves each time a scan is performed.

Scan costs are easy to define, requiring only a non-negative float or integer value. You must specify the cost of all scans. Here, in the example ``tiny`` scenario, we define a cost of 1 for all scans:

.. code-block:: yaml

    service_scan_cost: 1
    os_scan_cost: 1
    subnet_scan_cost: 1
    process_scan_cost: 1


Step limit
""""""""""

The step limit defines the maximum number of steps (i.e. actions) the pen-tester has to reach the goal within a single episode. During simulation once the step limit is reached the episode is considered done, with the agent having failed to reach the goal.

Defining the step limit is easy since it requires only a positive integer value. For example, here we define a step limit of 1000 for the ``tiny`` scenario:

.. code-block:: yaml

    step_limit: 1000



With that we have everything we need to define a custom scenario. Running the scenario is even easier!


.. _'running_custom_yaml':

Running a custom YAML scenario
------------------------------

To create a ``NASimEnv`` from a custom YAML scenario file we use the ``nasim.load()`` function:

.. code-block:: python

   import nasim
   env = nasim.load('path/to/custom/scenario.yaml`)


The load function also takes some additional parameters to control the observation mode and observation and action spaces for the environment, see :ref:`nasim_init` for reference and :ref:`env_params` for explanation.

If there are any issues with the format of your file you should recieve some, hopefully, helpful error messages when attempting to load it. Once the environment is loaded successfully you can interact with it as per normal (see :ref:`env_tute` for more details).
