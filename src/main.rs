#![feature(associated_type_defaults)]
#[macro_use]
extern crate lazy_static;

mod graph;
// mod old_tensor;
mod nn;
mod scalar;
mod tensor;

use scalar::utils::{gradient_descent, prediction_loss, predictions, read_xy_from_json};
use scalar::{Value, MLP};
use tensor::{get_raw_matrix, ArrayType, Matrix, Tensor};
// use tensor::Vector;

use crate::scalar::utils::save_weights;

use serde_json::json;
use std::borrow::Borrow;
use std::fs::File;
use std::io::prelude::*;
use std::vec;

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
    // moon_mlp();

    let start = std::time::Instant::now();
    // basic_tensor();
    moon_mlp_tensor();
    println!("Tensor test duration: {:?}", start.elapsed());

    // let start = std::time::Instant::now();
    // basic_nn();
    // println!("Basic NN test duration: {:?}", start.elapsed());
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

    // let mut loss_val =
    //     prediction_loss(&n, &xs, |pred| generate_loss(pred, ys.clone())).get_number_val();

    // println!("loss {:?}", loss_val);

    // for _ in 0..100 {
    //     if loss_val < 0.025 {
    //         break;
    //     }

    //     for p in &params {
    //         let old_number = p.get_number_val();
    //         let new_number = old_number + (-0.05 * p.get_grad_val());

    //         p.number.replace(new_number);
    //     }

    //     loss_val =
    //         prediction_loss(&n, &xs, |pred| generate_loss(pred, ys.clone())).get_number_val();

    //     println!("new loss {:?}", loss_val);
    // }

    let pred = predictions(&n, &xs);

    println!("new predictions {:?}", pred);

    // let loss = generate_loss(pred, ys.clone());

    // graph::scalar::draw_dots(loss);
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
                // mean  v n
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

    let _ = generate_plot_data(&n, xs, ys);

    let _ = save_weights(&n, "weights.bin");
}

fn basic_tensor() {
    // let a = Vector::new(vec![1.0, 2.0, 3.0]);
    // let b = Vector::new(vec![4.0, 5.0, 6.0]);
    // let c = &a * &b;

    // let d = &c * &a;

    // println!("a: {:?}", a.get_number());
    // println!("b: {:?}", b.get_number());
    // println!("c: {:?}", c.get_number());

    // d.backward();
    // println!("d grad: {:?}", d.get_grad());
    // println!("c grad: {:?}", c.get_grad());

    // println!("a grad: {:?}", a.get_grad());
    // println!("b grad: {:?}", b.get_grad());

    // graph::tensor::draw_dots(d);

    // let a: Tensor<Matrix> = Matrix::new_with_label(
    //     vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    //     "a".to_string(),
    // );

    // // std::thread::sleep(std::time::Duration::from_secs(10));
    // let b = Matrix::new_with_label(
    //     vec![
    //         vec![7.0, 8.0, 9.0],
    //         vec![10.0, 11.0, 12.0],
    //         vec![13.0, 14.0, 15.0],
    //     ],
    //     "b".to_string(),
    // );

    // let d = Matrix::new_with_label(
    //     vec![vec![2.0, 2.0, 2.0], vec![2.0, 2.0, 2.0]],
    //     "d".to_string(),
    // );

    // // std::thread::sleep(std::time::Duration::from_secs(10));
    // let c = &a + &d;

    // // c.backward();

    // // TODO
    // let e = (&c * &b).tanh();

    // // println!("a {:?}", a.get_numbers());
    // // println!("b {:?}", b.get_numbers());
    // // println!("c: {:?}", c.get_numbers());

    // // e.set_grad(1.0);
    // // println!("e grad {:?}", e.get_grad());
    // // e.backward();
    // // println!("e grad {:?}", e.get_grad());
    // // println!("d grad {:?}", d.get_grad());

    // e.backward();
    // println!("c grad: {:?}", c.get_grad());

    // println!("a grad: {:?}", a.get_grad());
    // println!("b grad: {:?}", b.get_grad());

    // draw_dots(e)

    // let neuron = nn::Neuron::new(4, 3);

    let mlp = nn::MLP::new(&[3, 4, 4, 1]);

    let x = Matrix::new(vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.01],
        vec![1.0, 1.0, -1.01],
    ]);

    // let ys = Matrix::new(vec![vec![-1.0, -1.0]]);
    let ys = Matrix::new(vec![vec![1.0], vec![-1.0], vec![-1.0], vec![1.0]]);

    // // let y = neuron.call(x);

    let mut loss = nn::prediction_loss(&mlp, &x, |pred| (pred - &ys).pow(2.0));

    println!("first loss {:?}", loss.get_numbers());

    // mlp.print_weights();

    for i in 0..100 {
        mlp.descend(-0.05);
        // println!("DESCENT -------");

        // mlp.print_weights();

        loss = nn::prediction_loss(&mlp, &x, |pred| (pred - &ys).pow(2.0));
        println!("loss {} {:?}", i, loss.get_numbers());
    }

    let y = mlp.call(&x);

    println!(
        "final preds {:?} shape {:?} {:?}",
        y.get_numbers(),
        y.size().unwrap(),
        ys.size().unwrap()
    );

    // let loss = (&y - &ys).pow(2.0);

    println!("loss {:?}", loss.get_numbers());

    // y.backward();
    // loss.backward();

    graph::tensor::draw_dots(loss);

    // println!("")
    // mlp.descend(-0.1);
}

fn moon_mlp_tensor() {
    let (xs, ys) = tensor::utils::read_xy_from_json("moonData.json").unwrap();

    let n = nn::MLP::new(&[2, 16, 16, 1]);

    fn gen_loss(y_out: &Tensor<Matrix>, y_gt: Tensor<Matrix>) -> Tensor<Matrix> {
        println!("gen_loss");
        (y_out - &y_gt).pow(2.0)
    }

    println!("yt {:?}", ys.get_numbers());

    let y = n.call(&xs);

    println!("initial preds  {:?}", y.get_numbers());

    // println!("shape {:?} y_gt {:?}", y.size(), ys.size());

    let mut loss = nn::prediction_loss(&n, &xs, |pred| gen_loss(pred, ys.clone()));

    for i in 0..30 {
        n.descend(-0.1);

        loss = nn::prediction_loss(&n, &xs, |pred| gen_loss(pred, ys.clone()));
        println!("loss {} {:?}", i, loss.get_numbers());
    }

    let y_final = n.call(&xs);

    println!("final preds  {:?}", y_final.get_numbers());

    // println!("loss {:?}", loss.get_numbers());
}
