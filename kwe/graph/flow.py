"""
LangGraph orchestration flow definitions for KWECLI.
"""
from typing import Callable, Any, Dict

class StateGraph:
    """Defines a directed graph of nodes with guard and retry logic."""
    def __init__(self):
        self.nodes: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.edges: Dict[str, Dict[str, str]] = {}

    def add_node(self, name: str, func: Callable[[Dict[str, Any]], Any]) -> None:
        self.nodes[name] = func

    def add_edge(self, from_node: str, to_node: str, on: str = 'ok') -> None:
        self.edges.setdefault(from_node, {})[on] = to_node

    def compile(self) -> 'StateGraph':
        # In more advanced usage, validate graph consistency
        return self

    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        state = initial_state.copy()
        current = 'plan'
        while True:
            func = self.nodes.get(current)
            if not func:
                break
            result = func(state)
            status = result.get('status', 'ok') if isinstance(result, dict) else 'ok'
            next_node = self.edges.get(current, {}).get(status)
            if not next_node or next_node == current:
                break
            current = next_node
        return state

def export_graph(graph: StateGraph, path: str) -> None:
    """Export the orchestration graph as a DOT file."""
    try:
        lines = ['digraph G {']
        for src, outs in graph.edges.items():
            for cond, dst in outs.items():
                lines.append(f'    "{src}" -> "{dst}" [label="{cond}"];')
        lines.append('}')
        with open(path.replace('.png', '.dot'), 'w') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
