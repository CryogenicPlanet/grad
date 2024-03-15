// GRAPH CODE

use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::process::Command;

use petgraph::Graph;

use crate::grad;
use grad::Value;

use std::io::Write;

fn trace(root: Value) -> (HashSet<Value>, HashSet<(Value, Value)>) {
    let mut nodes: HashSet<Value> = HashSet::new();
    let mut edges: HashSet<(Value, Value)> = HashSet::new();

    fn build(v: &Value, nodes: &mut HashSet<Value>, edges: &mut HashSet<(Value, Value)>) {
        if !nodes.contains(v) {
            nodes.insert(v.clone());
            if let Some(children) = &v.get_children() {
                for child in children {
                    edges.insert((child.clone(), v.clone()));
                    build(child, nodes, edges); // This will need to be adjusted to work with closures
                }
            }
        }
    }

    build(&root, &mut nodes, &mut edges);

    return (nodes, edges);
}

pub fn draw_dots(root: Value) {
    let mut deps = Graph::<&str, &str>::new();

    let (nodes, edges) = trace(root);

    // let mut labels: Vec<String> = Vec::new();

    let mut node_lookup_input: HashMap<Value, NodeIndex> = HashMap::new();
    let mut node_lookup_output: HashMap<Value, NodeIndex> = HashMap::new();

    for node in &nodes {
        let node_label = format!(
            "d {:.2} | grad {:.2} ",
            // node.label,
            node.get_number_val(),
            node.get_grad_val()
        );
        let node_label_str = Box::leak(node_label.into_boxed_str());
        let node_elm = deps.add_node(node_label_str);
        node_lookup_input.insert(node.clone(), node_elm);

        if node.get_op() != "" {
            let op_label = format!("{}", node.get_op());
            let op_label_str = Box::leak(op_label.into_boxed_str());
            let op_elm = deps.add_node(op_label_str);

            deps.extend_with_edges(&[(op_elm, node_elm)]);

            node_lookup_output.insert(node.clone(), op_elm);
        }
    }

    for edge in &edges {
        let start_node = node_lookup_input.get(&edge.0).unwrap();
        let end_node = node_lookup_output.get(&edge.1).unwrap();
        deps.extend_with_edges(&[(*start_node, *end_node)]);
    }

    let dot = Dot::with_config(&deps, &[Config::EdgeNoLabel]);
    let mut file = File::create("graph.dot").unwrap();
    writeln!(file, "{:?}", dot).unwrap();

    Command::new("dot")
        .args(&["-Tpng", "graph.dot", "-o", "graph.png"])
        .output()
        .expect("Failed to execute command");

    Command::new("open")
        .arg("graph.png")
        .output()
        .expect("Failed to open graph.png");
}
