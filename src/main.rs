mod grad;
mod graph;

use grad::utils::{
    gradient_descent, load_weights, prediction_loss, predictions, read_xy_from_json,
};
use grad::{Value, MLP};

use crate::grad::utils::save_weights;
use crate::graph::draw_dots;
// use graph::draw_dots;

/*
fn main() {
    // inputs
    let x1 = Value::leaf(1.0, "x1".to_string());
    let x2 = Value::leaf(2.0, "x2".to_string());

    // weights
    let w1 = Value::leaf(-3.0, "w1".to_string());
    let w2 = Value::leaf(-4.0, "w2".to_string());

    // biases
    let b = Value::leaf(8.0, "b".to_string());

    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;

    let mut n = &(&(x1w1) + &(x2w2)) + &b;

    n = n.set_label("n".to_string());

    // let o = n.tanh();

    let e = &(&n * 2.0).exp();
    let o = &(e - 1.0) / &(e + 1.0);

    // let a = Value::leaf(-2.0, "a".to_string());
    // let b = Value::leaf(3.0, "b".to_string());
    // let d = &a * &b;
    // let d = d.set_label("d".to_string());
    // let e = &a + &b;
    // let e = e.set_label("e".to_string());
    // let f = &d * &e;
    // let f = f.set_label("f".to_string());
    // f.backward();

    // o.maybe_backwards();

    o.backward();
}

*/

fn main() {
    // basic_nn();
    // draw_dots(loss);

    moon_mlp();

    // let model = MLP::new(2, vec![16, 16, 1]);

    // let _ = load_weights(&model, "weights.bin");

    // let (xs, ys) = read_xy_from_json("moonData.json").unwrap();

    // let _ = generate_plot_data(&model, xs.clone(), ys.clone());

    // for (gt, pred) in ys.iter().zip(predictions(&model, &xs).iter()).take(5) {
    //     println!(
    //         "gt {:?} vs prediction {:?}",
    //         gt.get_number_val(),
    //         pred.get_number_val()
    //     );
    // }
}

fn basic_nn() {
    let n = MLP::new(3, vec![4, 4, 1]);

    let params = n.parameters();

    let xs = vec![
        vec![Value::n(2.0), Value::n(3.0), Value::n(-1.0)],
        vec![Value::n(3.0), Value::n(-1.0), Value::n(0.5)],
        vec![Value::n(0.5), Value::n(1.0), Value::n(1.01)],
        vec![Value::n(1.0), Value::n(1.0), Value::n(-1.01)],
    ];

    fn generate_loss(y_out: Vec<Value>, y_gt: Vec<Value>) -> Value {
        y_out
            .iter()
            .zip(y_gt.iter())
            .map(|(y_out, y_gt)| (y_out - y_gt).pow(2.0))
            .reduce(|a, b| &a + &b)
            .unwrap_or(Value::n(0.0))
    }

    let ys = vec![Value::n(1.0), Value::n(-1.0), Value::n(-1.0), Value::n(1.0)];

    let mut loss_val =
        prediction_loss(&n, &xs, |pred| generate_loss(pred, ys.clone())).get_number_val();

    println!("loss {:?}", loss_val);

    for _ in 0..100 {
        if loss_val < 0.025 {
            break;
        }

        for p in &params {
            let old_number = p.get_number_val();
            let new_number = old_number + (-0.05 * p.get_grad_val());

            p.number.replace(new_number);
        }

        loss_val =
            prediction_loss(&n, &xs, |pred| generate_loss(pred, ys.clone())).get_number_val();

        println!("new loss {:?}", loss_val);
    }

    let pred = predictions(&n, &xs);

    println!("new predictions {:?}", pred);

    let loss = generate_loss(pred, ys.clone());

    draw_dots(loss);
}

fn moon_mlp() {
    let (xs, ys) = read_xy_from_json("moonData.json").unwrap();

    let n = MLP::new(2, vec![16, 16, 1]);

    println!("params {:?}", n.parameters().len());

    fn generate_loss(y_out: Vec<Value>, y_gt: Vec<Value>, n: &MLP) -> Value {
        let losses = y_out
            .iter()
            .zip(y_gt.iter())
            .map(|(ypred, yi)| {
                let margin_loss = (((yi * -1.0).borrow() * ypred).borrow() + 1.0).relu();
                margin_loss
            })
            .collect::<Vec<Value>>();

        let sum_loss = losses
            .iter()
            .map(|a| a * 1.0)
            .reduce(|a, b| a.borrow() + b.borrow())
            .unwrap();

        let data_loss = sum_loss.borrow() * (1.0 / losses.len() as f64);

        let reg_loss = &(n
            .parameters()
            .iter()
            .map(|p| *p * *p)
            .reduce(|a, b| &a + &b)
            .unwrap())
            * 1e-4;

        let total_loss = &data_loss + &reg_loss;

        let accuracy: Vec<bool> = y_gt
            .iter()
            .zip(y_out.iter())
            .map(|(yi, ypred)| (yi.get_number_val() > 0.0) == (ypred.get_number_val() > 0.0))
            .collect();

        let accuracy_sum = accuracy
            .iter()
            .map(|a| if a.clone() { 1.0 } else { 0.0 })
            .sum::<f64>();

        println!("accuracy {:?}", accuracy_sum / accuracy.len() as f64);

        total_loss
    }

    let mut loss_val =
        prediction_loss(&n, &xs, |pred| generate_loss(pred, ys.clone(), &n)).get_number_val();

    println!("loss {:?}", loss_val);

    for i in 0..100 {
        let learning_rate = 1.0 - 0.9 * (i as f64) / 100.0;

        gradient_descent(&n, learning_rate);

        loss_val =
            prediction_loss(&n, &xs, |pred| generate_loss(pred, ys.clone(), &n)).get_number_val();

        println!("new loss {:?} in index {}", loss_val, i);
    }

    for (gt, pred) in ys.iter().zip(predictions(&n, &xs).iter()).take(5) {
        println!(
            "gt {:?} vs prediction {:?}",
            gt.get_number_val(),
            pred.get_number_val()
        );
    }

    let _ = generate_plot_data(&n, xs, ys);

    let _ = save_weights(&n, "weights.bin");
}

use serde_json::json;
use std::borrow::Borrow;
use std::fs::File;
use std::io::prelude::*;

// Assuming `model` is a function that takes a slice of `Value` and returns a `Value`
// Assuming `Value` struct has a `data` field of type f64 for simplicity
fn generate_plot_data(model: &MLP, xs: Vec<Vec<Value>>, y: Vec<Value>) -> std::io::Result<()> {
    let h = 0.25;
    let (x_min, x_max) = (
        xs.iter()
            .map(|v| v[0].get_number_val())
            .fold(f64::INFINITY, f64::min)
            - 1.0,
        xs.iter()
            .map(|v| v[0].get_number_val())
            .fold(f64::NEG_INFINITY, f64::max)
            + 1.0,
    );
    let (y_min, y_max) = (
        xs.iter()
            .map(|v| v[1].get_number_val())
            .fold(f64::INFINITY, f64::min)
            - 1.0,
        xs.iter()
            .map(|v| v[1].get_number_val())
            .fold(f64::NEG_INFINITY, f64::max)
            + 1.0,
    );
    let mut xx = Vec::new();
    let mut yy = Vec::new();
    let mut inputs = Vec::new();

    let x_steps = ((x_max - x_min) / h).round() as usize;
    let y_steps = ((y_max - y_min) / h).round() as usize;

    for i in 0..x_steps {
        let mut xx_row = Vec::new();
        let mut yy_row = Vec::new();
        for j in 0..y_steps {
            let x = x_min + i as f64 * h;
            let y = y_min + j as f64 * h;
            xx_row.push(x);
            yy_row.push(y);
            inputs.push(vec![Value::n(x), Value::n(y)]);
        }
        xx.push(xx_row);
        yy.push(yy_row);
    }

    let scores: Vec<Value> = predictions(model, &inputs);

    // println!("scores {:?}", scores);

    let z: Vec<f64> = scores
        .iter()
        .map(|s| if s.get_number_val() > 0.0 { 1.0 } else { 0.0 })
        .collect();

    // println!("z {:?}", z);

    let xs_serial: Vec<Vec<f64>> = xs
        .iter()
        .map(|v| v.iter().map(|val| val.get_number_val()).collect())
        .collect();
    let y_serializable: Vec<f64> = y.iter().map(|val| val.get_number_val()).collect();

    // Assuming z is a flat Vec<f64> with the correct total number of elements
    let mut z_reshaped = vec![vec![0.0; y_steps]; x_steps];

    for (i, chunk) in z.chunks(y_steps).enumerate() {
        z_reshaped[i] = chunk.to_vec();
    }

    let plot_data = json!({
        "xx": xx,
        "yy": yy,
        "Z": z_reshaped,
        "X": xs_serial,
        "y": y_serializable
    });

    let mut file = File::create("plot_data.json")?;
    file.write_all(plot_data.to_string().as_bytes())?;

    Ok(())
}
