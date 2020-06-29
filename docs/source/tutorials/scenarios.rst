.. _`scenarios_tute`:

Understanding Scenarios
=======================

A scenario in NASim defines all the necessary properties for creating a network environment. Each scenario definition can be broken down into two components: the network configuration and the pen-tester.

Network Configuration
---------------------

The network configuration is defined by a the following properties:

- *subnets*: the number and size of the subnets in the network.
- *topology*: how the different subnets in the network are connected
- *host configurations*: what OS and services are running on each host in the network
- *firewall*: which communication is prevented between subnets


Pen-Tester
----------

The pen-tester is defined by:

- *exploits*: the set of exploits available to the pen-tester
- *scan costs*: the cost of performing each type of scan (service, OS, and subnet)
- *sensitive hosts*: the target hosts on the network and their value

Example Scenario
----------------

To illustrate these properties here we show an example scenario, where the aim of the pen-tester is to excract sensitive documents from the server in the sensitive subnet and one of the user hosts.

The figure below shows the the layout of our example network.

.. image:: example_network.png
  :width: 700

From the figure we can see that this network has the following properties:

- *subnets*: three subnets: DMZ with a single server, Sensitive with a single server and User with three user machines.
- *topology*: Only the DMZ is connected to the internet, while all subnets in network are interconnected.
- *host configurations*: The OS and services running on each host are shown next to each host (e.g. the server in the DMZ subnet is running Linux and http and ssh services).
- *firewall*: The arrows above and below the firwalls indicate which services can be communicated with in each direction between subnets and between the DMZ subnet and the internet (e.g. the internet can communicate with http services running on hosts in the DMZ, while the firewall blocks no communication from the DMZ to the internet).

Next we need to define our pen-tester, which we specify based on the scenario we wish to simulate.

- *exploits*: for this scenario the pen-tester has access to three exploits

  1. *ssh_exploit*: which exploits the ssh service running on windows machine, has a cost of 2 and a probability of working of 0.6.
  2. *ftp_exploit*: which exploits the ftp service running on a linux machine, has a cost of 1 and a probability of working of 0.9.
  3. *http_exploit*: which exploits the http service running on any OS, has a cost of 3 and a probability of working of 1.0.

- *scan costs*: here we need to specify the cost of each type of scan

  1. *service_scan*: 1
  2. *os_scan*: 2
  3. *subnet_scan*: 1

- *sensitive hosts*: here we have two target hosts

  1. *(2, 0), 1000* : the server (host 0) running on sensitive subnet (subnet 2), which has a value of 1000.
  2. *(3, 2), 1000* : the last user machine (host 2) running on user subnet (subnet 3), which has a value of 1000.

And with that our scenario is fully defined and we have everything we need to run an attack simulation.
