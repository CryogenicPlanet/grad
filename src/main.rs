use core::num;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::ops::{Add, Div, Mul, Sub};

fn main() {
    // // inputs
    // let x1 = Value::leaf(1.0, "x1".to_string());
    // let x2 = Value::leaf(2.0, "x2".to_string());

    // // weights
    // let w1 = Value::leaf(-3.0, "w1".to_string());
    // let w2 = Value::leaf(-4.0, "w2".to_string());

    // // biases
    // let b = Value::leaf(8.0, "b".to_string());

    // let x1w1 = &x1 * &w1;
    // let x2w2 = &x2 * &w2;

    // let mut n = &(&(x1w1) + &(x2w2)) + &b;

    // n = n.set_label("n".to_string());

    // // let o = n.tanh();

    // let e = &(&n * 2.0).exp();
    // let o = &(e - 1.0) / &(e + 1.0);

    // // let a = Value::leaf(-2.0, "a".to_string());
    // // let b = Value::leaf(3.0, "b".to_string());
    // // let d = &a * &b;
    // // let d = d.set_label("d".to_string());
    // // let e = &a + &b;
    // // let e = e.set_label("e".to_string());
    // // let f = &d * &e;
    // // let f = f.set_label("f".to_string());
    // // f.backward();

    // // o.maybe_backwards();

    // o.backward();

    let n = MLP::new(3, vec![4, 4, 1]);

    // println!("params {:?}", n.parameters().len());

    let params = n.parameters();

    let x: Vec<Value> = vec![
        Value::leaf(4.0, "a".to_string()),
        Value::leaf(3.0, "b".to_string()),
        Value::leaf(-6.0, "c".to_string()),
    ];

    // let y = n.call(x);
    // println!("value y: {:?}", y);

    let xs = vec![
        vec![Value::n(2.0), Value::n(3.0), Value::n(-1.0)],
        vec![Value::n(3.0), Value::n(-1.0), Value::n(0.5)],
        vec![Value::n(0.5), Value::n(1.0), Value::n(1.01)],
        vec![Value::n(1.0), Value::n(1.0), Value::n(-1.01)],
    ];

    let ys = vec![Value::n(1.0), Value::n(-1.0), Value::n(-1.0), Value::n(1.0)];

    fn predictions(n: &MLP, xs: &Vec<Vec<Value>>) -> Vec<Value> {
        let ypred: Vec<Value> = xs
            .iter()
            .flat_map(|x| {
                let result = n.call(x.to_vec());
                if result.len() == 1 {
                    result.into_iter()
                } else {
                    vec![].into_iter()
                }
            })
            .collect();

        ypred
    }

    fn prediction_loss(n: &MLP, xs: &Vec<Vec<Value>>, ys: &Vec<Value>) -> Value {
        let ypred = predictions(n, xs);
        let loss = generate_loss(ypred, ys.clone());

        loss.zero_grad();

        loss.backward();
        // println!("Backward pass done");

        loss
    }

    let mut loss_val = prediction_loss(&n, &xs, &ys).get_number_val();

    println!("loss {:?}", loss_val);

    for _ in 0..100 {
        if loss_val < 0.05 {
            break;
        }

        for p in &params {
            let old_number = p.get_number_val();
            let new_number = old_number + (-0.05 * p.get_grad_val());

            p.number.replace(new_number);
        }

        loss_val = prediction_loss(&n, &xs, &ys).get_number_val();

        println!("new loss {:?}", loss_val);
    }

    println!("new predictions {:?}", predictions(&n, &xs))

    // draw_dots(loss);
}

fn generate_loss(y_out: Vec<Value>, y_gt: Vec<Value>) -> Value {
    y_out
        .iter()
        .zip(y_gt.iter())
        .map(|(y_out, y_gt)| (y_out - y_gt).pow(2.0))
        .reduce(|a, b| &a + &b)
        .unwrap_or(Value::n(0.0))
}

fn topological_sort<'a>(root: &'a Value) -> Vec<&'a Value> {
    fn visit<'a>(root: &'a Value, visited: &mut HashSet<String>, sorted: &mut Vec<&'a Value>) {
        // println!("visiting {}", root.label);
        if !visited.contains(&root.uuid) {
            if visited.insert(root.uuid.clone()) {
                if let Some(children) = &root._prev {
                    for child in children {
                        visit(child, visited, sorted);
                    }
                }
                sorted.push(root);
            }
        }
    }

    let mut visited = HashSet::new();
    let mut sorted = Vec::new();
    visit(root, &mut visited, &mut sorted);
    sorted
}

#[derive(Clone)]
struct Value {
    uuid: String,
    number: Rc<RefCell<f64>>,
    label: String,
    grad: Rc<RefCell<f64>>,
    _prev: Option<Vec<Value>>,
    _op: String,
    _backwards: Option<Rc<RefCell<Box<dyn FnMut()>>>>, // Adjusted to use Rc for cloning and made optional
}

impl Value {
    fn get_grad_val(&self) -> f64 {
        self.grad.borrow().clone()
    }

    fn get_number_val(&self) -> f64 {
        self.number.borrow().clone()
    }

    fn tanh(self) -> Value {
        // let self_clone = self.clone(); // Clone self to use for backward pass

        let t = self.get_number_val().tanh();

        let mut out = Value::new_with_children(
            t,
            Some(vec![self.clone()]),
            "tanh".to_string(),
            format!("tanh({})", self.label),
        );

        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let old_self_grad = *self_grad_clone.borrow();

            let old_out_grad = *out_grad_clone.borrow();

            let new_grad = old_self_grad + (1.0 - t.powi(2)) * old_out_grad;

            self.grad.replace(new_grad);

            // println!("tanh backward pass {} {}", self.grad, out.grad);
        }) as Box<dyn FnMut()>));

        out = out.set_backwards_pass(backward_pass);

        out
    }

    fn exp(&self) -> Value {
        let e = self.get_number_val().exp();

        let mut out = Value::new_with_children(
            e,
            Some(vec![self.clone()]),
            "exp".to_string(),
            format!("exp({})", self.label),
        );

        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let old_self_grad = *self_grad_clone.borrow();

            let old_out_grad = *out_grad_clone.borrow();

            // dx/dy of e^x is e^x
            let new_grad = old_self_grad + e * old_out_grad;

            self_grad_clone.replace(new_grad);

            // println!("tanh backward pass {} {}", self.grad, out.grad);
        }) as Box<dyn FnMut()>));

        out = out.set_backwards_pass(backward_pass);

        out
    }

    fn pow(&self, other: f64) -> Value {
        let number = self.get_number_val();

        let data = number.powf(other);
        let mut out = Value::new_with_children(
            data,
            Some(vec![self.clone()]),
            "pow".to_string(),
            format!("pow({})", self.label),
        );

        let val_of_derivate = other * number.powf(other - 1.0);

        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let old_self_grad = *self_grad_clone.borrow();

            let old_out_grad = *out_grad_clone.borrow();

            // dx/dy of e^x is e^x
            let new_grad = old_self_grad + val_of_derivate * old_out_grad;

            self_grad_clone.replace(new_grad);
        }) as Box<dyn FnMut()>));

        out = out.set_backwards_pass(backward_pass);

        out
    }

    fn maybe_backwards(&self) {
        match &self._backwards {
            Some(f) => {
                // println!("back propagating for {}", self.label);
                (*f.borrow_mut())();

                // if let Some(prev) = &self._prev {
                //     for p in prev {
                //         p.maybe_backwards();
                //     }
                // }
            }
            None => (),
        }
    }

    fn backward(&self) {
        self.set_grad(1.0);

        let sorted = topological_sort(self);

        for node in sorted.iter().rev() {
            node.maybe_backwards();
        }
    }

    // The zero grad will take the same node as root and just set all of gradients to zero before a backwards pass
    fn zero_grad(&self) {
        let sorted = topological_sort(self);

        for node in sorted.iter().rev() {
            let zero_grad = 0.0;
            node.grad.replace(zero_grad);
        }
    }

    fn set_backwards_pass(mut self, backward_pass: Rc<RefCell<Box<dyn FnMut()>>>) -> Self {
        self._backwards = Some(backward_pass);
        self
    }

    fn set_grad(&self, grad: f64) {
        self.grad.replace(grad);
    }

    fn set_label(mut self, label: String) -> Self {
        self.label = label;
        self
    }

    fn leaf(number: f64, label: String) -> Self {
        Self::new_with_children(number, None, "".to_string(), label)
    }

    fn n(number: f64) -> Self {
        Self::new_with_children(number, None, "".to_string(), "".to_string())
    }

    fn new_with_children(
        number: f64,
        children: Option<Vec<Value>>,
        op: String,
        label: String,
    ) -> Self {
        Value {
            uuid: Uuid::new_v4().to_string(),
            number: Rc::new(RefCell::new(number)),
            grad: Rc::new(RefCell::new(0.0)),
            _prev: children,
            _op: op,
            label,
            _backwards: None,
        }
    }
}

struct Neuron {
    nin: i32,
    w: Vec<Value>,
    b: Value,
}

use rand::distributions::{Distribution, Uniform};

impl Neuron {
    fn new(nin: i32) -> Self {
        let uniform = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let bias = Value::leaf(uniform.sample(&mut rng), "b".to_string());

        let mut w: Vec<Value> = Vec::new();

        for i in 0..nin {
            let wi = Value::leaf(uniform.sample(&mut rng), format!("w{}", i));
            // println!("weight val {}", wi.number);
            w.push(wi);
        }

        Neuron { nin, w, b: bias }
    }

    fn call(&self, x: Vec<Value>) -> Value {
        assert_eq!(x.len(), self.w.len(), "x and w must be the same size.");

        let zipped = x.iter().zip(self.w.iter());

        let sum = zipped.map(|(x, w)| x * w).reduce(|a, b| &a + &b);

        match sum {
            Some(sum) => (&sum + &self.b).tanh(),
            None => panic!("something went wrong in the sum func"),
        }
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::with_capacity(self.w.len() + 1);
        for weight in &self.w {
            params.push(weight);
        }
        params.push(&self.b);
        params
    }
}
//  impl

struct Layer {
    neurons: Vec<Neuron>,
    nin: i32,
    nout: i32,
}

impl Layer {
    fn new(nin: i32, nout: i32) -> Self {
        let mut neurons: Vec<Neuron> = Vec::new();

        for _ in 0..nout {
            let n = Neuron::new(nin);
            neurons.push(n)
        }

        Layer { neurons, nin, nout }
    }

    fn call(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x.clone())).collect()
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        for n in &self.neurons {
            params.extend(n.parameters());
        }
        params
    }
}

struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn new(nin: i32, nount: Vec<i32>) -> Self {
        let mut sizes = vec![nin];
        sizes.extend(nount);

        let mut layers: Vec<Layer> = Vec::new();

        for i in 0..sizes.len() - 1 {
            // println!("Input size: {}, Output size: {}", sizes[i], sizes[i + 1]);
            let n = Layer::new(sizes[i], sizes[i + 1]);
            layers.push(n);
        }

        MLP { layers }
    }

    fn call(&self, x: Vec<Value>) -> Vec<Value> {
        let mut out = x;

        for layer in self.layers.iter() {
            out = layer.call(out.clone());
        }

        out
    }

    fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        for l in &self.layers {
            params.extend(l.parameters());
        }
        params
    }
}

impl Div<&Value> for &Value {
    type Output = Value;
    fn div(self, rhs: &Value) -> Value {
        let exp = rhs.pow(-1.0);
        self * &exp
    }
}

// Implementing the Add trait for Value using references to avoid moving ownership.
impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        let a = self;
        let b = other;

        let out = Value::new_with_children(
            a.get_number_val() + b.get_number_val(),
            Some(vec![a.clone(), b.clone()]),
            "+".to_string(),
            format!("{}+{}", a.label, b.label),
        );

        let a_grad_clone = Rc::clone(&a.grad);
        let b_grad_clone = Rc::clone(&b.grad);
        let out_grad_clone = Rc::clone(&out.grad);

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let old_out_grad = *out_grad_clone.borrow();

            let old_a_grad = *a_grad_clone.borrow();
            let new_a_grad = old_a_grad + 1.0 * old_out_grad;
            a_grad_clone.replace(new_a_grad);

            let old_b_grad = *b_grad_clone.borrow();
            let new_b_grad = old_b_grad + 1.0 * old_out_grad;
            b_grad_clone.replace(new_b_grad);
        }) as Box<dyn FnMut()>));

        out.set_backwards_pass(backward_pass)
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, rhs: &Value) -> Value {
        self + &(rhs * -1.0)
    }
}

impl Sub<f64> for &Value {
    type Output = Value;

    fn sub(self, rhs: f64) -> Value {
        self + (rhs * -1.0)
    }
}

impl Add<f64> for &Value {
    type Output = Value;

    fn add(self, other: f64) -> Value {
        let b = Value::leaf(other, format!("{}", other));

        self + &b
    }
}

// Implementing the Mul trait for Value using references to avoid moving ownership.
impl Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        let a = self;
        let b = other;

        let a_number = a.get_number_val();
        let b_number = b.get_number_val();

        let out = Value::new_with_children(
            a_number * b_number,
            Some(vec![a.clone(), b.clone()]),
            "*".to_string(),
            format!("{}*{}", a.label, b.label),
        );

        let a_grad_clone = Rc::clone(&self.grad);
        let b_grad_clone = Rc::clone(&other.grad);
        let out_grad_clone = Rc::clone(&out.grad);

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let old_out_grad = *out_grad_clone.borrow();

            let old_a_grad = *a_grad_clone.borrow();
            let new_a_grad = old_a_grad + b_number * old_out_grad; // Use captured value
            a_grad_clone.replace(new_a_grad);

            let old_b_grad = *b_grad_clone.borrow();
            let new_b_grad = old_b_grad + a_number * old_out_grad; // Use captured value
            b_grad_clone.replace(new_b_grad);
        }) as Box<dyn FnMut()>));

        out.set_backwards_pass(backward_pass)
    }
}

impl Mul<f64> for &Value {
    type Output = Value;

    fn mul(self, other: f64) -> Value {
        let b = Value::leaf(other, format!("{}", other));
        self * &b
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // Define your tolerance for floating-point comparison
        let tolerance = 1e-6;
        (self.get_number_val() - other.get_number_val()).abs() < tolerance && self._op == other._op
        // Assuming _prev comparison is not needed or should be handled differently
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Discretize or convert the float to a stable format for hashing
        let discretized = (self.get_number_val() * 1e6).round() as i64;
        discretized.hash(state);
        self._op.hash(state);
        // Handle _prev hashing if needed, possibly by hashing the length or similar
    }
}

use std::rc::Rc;
use std::{fmt, vec};

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("number", &self.number)
            // .field("_prev", &self._prev)
            // .field("_op", &self._op)
            .finish()
    }
}

// GRAPH CODE

fn trace(root: Value) -> (HashSet<Value>, HashSet<(Value, Value)>) {
    let mut nodes: HashSet<Value> = HashSet::new();
    let mut edges: HashSet<(Value, Value)> = HashSet::new();

    fn build(v: &Value, nodes: &mut HashSet<Value>, edges: &mut HashSet<(Value, Value)>) {
        if !nodes.contains(v) {
            nodes.insert(v.clone());
            if let Some(children) = &v._prev {
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

use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use std::process::Command;
use uuid::Uuid;

use petgraph::Graph;

fn draw_dots(root: Value) {
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

        if node._op != "" {
            let op_label = format!("{}", node._op);
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
