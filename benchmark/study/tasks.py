"""Render the dependency table in session.org. Pipe --dot to Graphviz for SVG."""

import argparse
from dataclasses import dataclass
from graphlib import TopologicalSorter
import json
from pathlib import Path


@dataclass(frozen=True)
class Task:
    id: str
    work: str
    depends_on: tuple[str, ...]
    status: str


def read_graph(path):
    tasks = {}
    for line in path.read_text().splitlines():
        if not line.startswith("| "):
            continue
        key, work, deps, status = (cell.strip() for cell in line.strip("|").split("|"))
        if key == "ID":
            continue
        if key in tasks:
            raise ValueError(f"Repeated task {key}")
        tasks[key] = Task(key, work, tuple(d.strip() for d in deps.split(",") if d.strip()), status)
    for task in tasks.values():
        missing = set(task.depends_on) - tasks.keys()
        if missing:
            raise ValueError(f"{task.id} has unknown prerequisites {missing}")
    order = TopologicalSorter({task.id: task.depends_on for task in tasks.values()}).static_order()
    return [tasks[key] for key in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot", action="store_true")
    parser.add_argument("--log", type=Path, default=Path(__file__).with_name("session.org"))
    args = parser.parse_args()
    tasks = read_graph(args.log)
    if args.dot:
        print('digraph research { rankdir=LR; node [shape=box, style=rounded, fontname="sans"];')
        for task in tasks:
            label = json.dumps(f"{task.id}. {task.work}\n{task.status}")
            print(f'  "{task.id}" [label={label}];')
            for dep in task.depends_on:
                print(f'  "{dep}" -> "{task.id}";')
        print("}")
    else:
        done = {task.id for task in tasks if task.status == "done"}
        for task in tasks:
            pending = ", ".join(dep for dep in task.depends_on if dep not in done)
            print(f"{task.id}  {task.status:<10}  {task.work}" + (f" [needs {pending}]" if pending else ""))


if __name__ == "__main__":
    main()
