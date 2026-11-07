"""Separate logical Target calls from the extra work when creating CUDA graphs."""


def initialization(runtime,tree):
    linear_events=runtime.get('graph_events',[])
    tree_events=(tree or {}).get('graph_events',[])
    # Both unchanged runtimes perform two warmup evaluations and one capture.
    evaluations=3
    nodes=evaluations*(sum(x['query_tokens'] for x in linear_events)+sum(x['node_count'] for x in tree_events))
    seconds=sum(sum(x.get(k,0.) for k in ('prepare_ms','warmup_ms','capture_ms')) for x in linear_events+tree_events)/1000
    return dict(graph_count=len(linear_events)+len(tree_events),
        additional_target_trunk_evaluations=evaluations*(len(linear_events)+len(tree_events)),
        additional_target_input_nodes=nodes,recorded_graph_initialization_seconds=seconds,
        scope='Two warmup evaluations and one capture per saved graph event; replay calls are recorded separately')
