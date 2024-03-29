use std::vec;

use crate::tensor::{get_matrix_from_buffer, get_raw_matrix, gpu, Matrix, Tensor};
use metal::{CommandBufferRef, ComputeCommandEncoderRef, Device, MTLSize};
use rand::distributions::{Distribution, Uniform};

pub struct Neuron {
    pub layer_size: i32,
    pub neurons_per_layer: i32,
    pub w: Tensor<Matrix>,
    pub b: Tensor<Matrix>,
}

pub fn prediction_loss<F>(n: &MLP, x: &Tensor<Matrix>, loss_cb: F) -> Tensor<Matrix>
where
    F: Fn(&Tensor<Matrix>) -> Tensor<Matrix>,
{
    let ypred = n.call(x);
    let loss = loss_cb(&ypred);

    loss.zero_grad();

    println!("backwards");
    let start = std::time::Instant::now();
    loss.backward();
    let duration = start.elapsed();
    println!("Backwards took: {:?}", duration);

    loss
}

impl Neuron {
    pub fn new(layer_size: i32, neurons_per_layer: i32) -> Self {
        let uniform = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();

        let bias = Matrix::new(vec![vec![1.0f32; neurons_per_layer as usize]]);

        let mut weights: Vec<Vec<f32>> = Vec::new();

        for _ in 0..neurons_per_layer {
            let w = vec![uniform.sample(&mut rng); layer_size as usize];
            weights.push(w);
        }

        Neuron {
            layer_size,
            neurons_per_layer,
            w: Matrix::new(weights),
            b: bias,
        }
    }

    pub fn call(&self, x: Tensor<Matrix>) -> Tensor<Matrix> {
        let start = std::time::Instant::now();

        println!("calling neuron");

        let mut xw = &(&x * &self.w) + &self.b;
        xw = xw.tanh();

        let duration = start.elapsed();
        println!("Neuron call took: {:?}", duration);

        xw
    }

    pub fn descend(&self, learning_rate: f32) {
        let weights = &self.w;
        let bias = &self.b;

        let device = weights.device.clone();
        let command_queue = weights.command_queue.clone();

        let ltr_pipeline = gpu::gpu_pipeline("apply_learning_rate", &device);

        let (weights_encoder, weights_command_buffer) =
            gpu::gpu_encoder(ltr_pipeline.clone(), &command_queue);

        println!("descending weights");
        self.setup_encoder_for_learning_rate(
            weights_encoder,
            weights_command_buffer,
            weights,
            learning_rate,
            &device,
        );

        // Setup for bias
        let (bias_encoder, bias_command_buffer) = gpu::gpu_encoder(ltr_pipeline, &command_queue);
        println!("descending bias");
        self.setup_encoder_for_learning_rate(
            bias_encoder,
            bias_command_buffer,
            bias,
            learning_rate,
            &device,
        );
    }

    pub fn print_weights(&self) -> String {
        let weights = &self.w;
        let bias = &self.b;

        format!(
            "weights {:?}\n\tbias {:?}",
            weights.get_numbers(),
            bias.get_numbers()
        )
    }

    fn setup_encoder_for_learning_rate(
        self: &Self,
        encoder: &ComputeCommandEncoderRef,
        command_buffer: &CommandBufferRef,
        tensor: &Tensor<Matrix>,
        learning_rate: f32,
        device: &Device,
    ) {
        let ltr = [learning_rate];
        let ltr_buffer = device.new_buffer_with_data(
            unsafe { std::mem::transmute(ltr.as_ptr()) },
            (std::mem::size_of::<[f32; 1]>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_buffer = device.new_buffer(
            tensor.number.borrow().length() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // println!(
        //     "\ttensor grad buffer {:?}\n\tnumber buffer {:?}\n\tltr buffer {:?}",
        //     get_raw_matrix(&tensor.grad, tensor.size().unwrap()),
        //     get_raw_matrix(&tensor.number, tensor.size().unwrap()),
        //     get_matrix_from_buffer(&ltr_buffer, (1, 1))
        // );

        encoder.set_buffer(0, Some(&tensor.number.borrow()), 0);
        encoder.set_buffer(1, Some(&tensor.grad.borrow()), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);
        encoder.set_buffer(3, Some(&ltr_buffer), 0);

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        let num_threadgroups = MTLSize {
            width: (tensor.number.borrow().length() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        gpu::gpu_end(num_threadgroups, threads_per_group, encoder, command_buffer);

        // println!(
        //     "\tresult buffer {:?}\n\n ---- \n",
        //     get_matrix_from_buffer(&result_buffer, tensor.size().unwrap())
        // );

        tensor.number.replace(result_buffer);
    }
}

pub struct MLP {
    layers: Vec<Neuron>,
}

impl MLP {
    pub fn new(layer_sizes: &[i32]) -> Self {
        let num_layers = layer_sizes.len();
        let mut layers = Vec::with_capacity(num_layers - 1);

        for i in 0..num_layers - 1 {
            let nin = layer_sizes[i];
            let nout = layer_sizes[i + 1];
            let layer = Neuron::new(nout, nin);
            layers.push(layer);
        }

        MLP { layers }
    }

    pub fn call(&self, x: &Tensor<Matrix>) -> Tensor<Matrix> {
        let start = std::time::Instant::now();
        let mut input = x.clone();
        for layer in &self.layers {
            input = layer.call(input);
        }
        let duration = start.elapsed();
        println!("MLP call took: {:?}", duration);
        input
    }

    pub fn print_weights(&self) {
        println!("MLP weights");
        for layer in &self.layers {
            let weights = layer.print_weights();

            println!(
                "Layer weights {}: \n\t{}\n----",
                layer.neurons_per_layer, weights
            );
        }
    }

    pub fn descend(&self, learning_rate: f32) {
        println!("descending");

        for layer in &self.layers {
            layer.descend(learning_rate);
        }
    }
}
