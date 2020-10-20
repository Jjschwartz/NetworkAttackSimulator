"""This script will output description statistics of all benchmark
scenarios.

It will output a table to stdout (and optionally to a .csv file) which
contains the following headers:

- Name : the scenarios name
- Type : static or generated
- Subnets : the number of subnets
- Hosts : the number of hosts
- OS : the number of OS
- Services : the number of services
- Processes : the number of processes
- Exploits : the number of exploits
- PrivEsc : the number of priviledge escalation actions
- Actions : the total number of actions available to agent
- States : the total number of states
- Step limit : the step limit for the scenario

Usage
-----

$ python describe_scenarios.py [-o --output filename.csv]

"""
import prettytable

from nasim.scenarios import make_benchmark_scenario
from nasim.scenarios.benchmark import AVAIL_BENCHMARKS


def describe_scenarios(output=None):
    rows = []
    headers = None
    for name in AVAIL_BENCHMARKS:
        scenario = make_benchmark_scenario(name, seed=0)
        des = scenario.get_description()
        if headers is None:
            headers = list(des.keys())

        if des["States"] > 1e8:
            des["States"] = f"{des['States']:.2E}"

        rows.append([str(des[h]) for h in headers])

    table = prettytable.PrettyTable(headers)
    for row in rows:
        table.add_row(row)

    print(table)

    if output is not None:
        print(f"\nSaving to {output}")
        with open(output, "w") as fout:
            fout.write(",".join(headers) + "\n")
            for row in rows:
                fout.write(",".join(row) + "\n")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="File name to output as CSV too")
    args = parser.parse_args()

    describe_scenarios(args.output)
