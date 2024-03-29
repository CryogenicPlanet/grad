pub mod utils {
    use std::collections::HashSet;

    use crate::scalar::{Value, MLP};

    pub fn topological_sort<'a>(root: &'a Value) -> Vec<&'a Value> {
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

    pub fn predictions(n: &MLP, xs: &Vec<Vec<Value>>) -> Vec<Value> {
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

    pub fn prediction_loss<F>(n: &MLP, xs: &Vec<Vec<Value>>, loss_cb: F) -> Value
    where
        F: Fn(Vec<Value>) -> Value,
    {
        let ypred = predictions(n, xs);
        let loss = loss_cb(ypred);

        loss.zero_grad();

        loss.backward();

        loss
    }

    pub fn gradient_descent(n: &MLP, lr: f64) {
        for p in &n.parameters() {
            let old_number = p.get_number_val();
            let new_number = old_number - (lr * p.get_grad_val());

            p.number.replace(new_number);
        }
    }

    pub fn save_weights(n: &MLP, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let weights = n.parameters();
        for w in weights {
            let weight = w.get_number_val();
            file.write_all(&weight.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }

    pub fn load_weights(n: &MLP, path: &str) -> std::io::Result<()> {
        let mut file = File::open(path)?;
        let weights = n.parameters();
        for w in weights {
            let mut weight_bytes = [0u8; 8]; // Create a buffer for 8 bytes
            file.read_exact(&mut weight_bytes)?; // Read 8 bytes into the buffer
            let weight = f64::from_le_bytes(weight_bytes); // Convert the bytes back into an f64
            w.number.replace(weight); // Replace the current weight with the loaded weight
        }
        file.flush()?;
        Ok(())
    }

    use serde_json;
    use std::fs::File;
    use std::io::{self, BufReader, Read, Write};
    use std::path::Path;

    pub fn read_xy_from_json<P: AsRef<Path>>(path: P) -> io::Result<(Vec<Vec<Value>>, Vec<Value>)> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let json: serde_json::Value = serde_json::from_reader(reader)?;

        let xs = json["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| {
                x.as_array()
                    .unwrap()
                    .iter()
                    .map(|n| Value::n(n.as_f64().unwrap()))
                    .collect()
            })
            .collect();

        let y_gt = json["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| Value::n(n.as_f64().unwrap()))
            .collect();

        Ok((xs, y_gt))
    }
}

use rand::distributions::{Distribution, Uniform};
use uuid::Uuid;

#[derive(Clone)]
pub struct Value {
    uuid: String,
    pub number: Rc<RefCell<f64>>,
    label: String,
    grad: Rc<RefCell<f64>>,
    _prev: Option<Vec<Value>>,
    _op: String,
    _backwards: Option<Rc<RefCell<Box<dyn FnMut()>>>>, // Adjusted to use Rc for cloning and made optional
}

impl Value {
    pub fn get_grad_val(&self) -> f64 {
        self.grad.borrow().clone()
    }

    pub fn get_number_val(&self) -> f64 {
        self.number.borrow().clone()
    }

    pub fn get_children(&self) -> Option<Vec<Value>> {
        self._prev.clone()
    }

    pub fn get_op(&self) -> String {
        self._op.clone()
    }

    pub fn relu(self) -> Value {
        let new_number = if self.get_number_val() < 0.0 {
            0.0
        } else {
            self.get_number_val()
        };

        let mut out = Value::new_with_children(
            new_number,
            Some(vec![self.clone()]),
            "relu".to_string(),
            format!("relu({})", self.label),
        );

        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let old_self_grad = *self_grad_clone.borrow();

            let old_out_grad = *out_grad_clone.borrow();

            let new_grad =
                old_self_grad + (if new_number > 0.0 { 1.0 } else { 0.0 }) * old_out_grad;

            self_grad_clone.replace(new_grad);
        }) as Box<dyn FnMut()>));

        out = out.set_backwards_pass(backward_pass);

        out
    }

    pub fn tanh(self) -> Value {
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

    pub fn exp(&self) -> Value {
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

    pub fn pow(&self, other: f64) -> Value {
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

    pub fn backward(&self) {
        self.set_grad(1.0);

        let sorted = utils::topological_sort(self);

        for node in sorted.iter().rev() {
            node.maybe_backwards();
        }
    }

    // The zero grad will take the same node as root and just set all of gradients to zero before a backwards pass
    pub fn zero_grad(&self) {
        let sorted = utils::topological_sort(self);

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

    pub fn set_label(mut self, label: String) -> Self {
        self.label = label;
        self
    }

    pub fn leaf(number: f64, label: String) -> Self {
        Self::new_with_children(number, None, "".to_string(), label)
    }

    pub fn n(number: f64) -> Self {
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

pub struct Neuron {
    pub nin: i32,
    pub w: Vec<Value>,
    pub b: Value,
}

impl Neuron {
    pub fn new(nin: i32) -> Self {
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

    pub fn call(&self, x: Vec<Value>) -> Value {
        assert_eq!(x.len(), self.w.len(), "x and w must be the same size.");

        let zipped = x.iter().zip(self.w.iter());

        let sum = zipped.map(|(x, w)| x * w).reduce(|a, b| &a + &b);

        match sum {
            Some(sum) => (&sum + &self.b).tanh(),
            None => panic!("something went wrong in the sum func"),
        }
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::with_capacity(self.w.len() + 1);
        for weight in &self.w {
            params.push(weight);
        }
        params.push(&self.b);
        params
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub nin: i32,
    pub nout: i32,
}

impl Layer {
    pub fn new(nin: i32, nout: i32) -> Self {
        let mut neurons: Vec<Neuron> = Vec::new();

        for _ in 0..nout {
            let n = Neuron::new(nin);
            neurons.push(n)
        }

        Layer { neurons, nin, nout }
    }

    pub fn call(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x.clone())).collect()
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();
        for n in &self.neurons {
            params.extend(n.parameters());
        }
        params
    }
}

pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: i32, nount: Vec<i32>) -> Self {
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

    pub fn call(&self, x: Vec<Value>) -> Vec<Value> {
        let mut out = x;

        for layer in self.layers.iter() {
            out = layer.call(out.clone());
        }

        out
    }

    pub fn parameters(&self) -> Vec<&Value> {
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

use std::hash::{Hash, Hasher};

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Discretize or convert the float to a stable format for hashing
        let discretized = (self.get_number_val() * 1e6).round() as i64;
        discretized.hash(state);
        self._op.hash(state);
        // Handle _prev hashing if needed, possibly by hashing the length or similar
    }
}

use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
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
