use metal::*;
use std::fmt;
use std::{cell::RefCell, fs, rc::Rc, sync::Arc};
use uuid::Uuid;

pub mod utils {
    use std::{
        collections::HashSet,
        fs::File,
        io::{self, BufReader},
        path::Path,
    };

    use super::{Matrix, Tensor, TensorType};

    pub fn topological_sort<'a, T: TensorType>(root: &'a Tensor<T>) -> Vec<&'a Tensor<T>> {
        fn visit<'a, T: TensorType>(
            root: &'a Tensor<T>,
            visited: &mut HashSet<String>,
            sorted: &mut Vec<&'a Tensor<T>>,
        ) {
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
    pub fn read_xy_from_json<P: AsRef<Path>>(
        path: P,
    ) -> io::Result<(Tensor<Matrix>, Tensor<Matrix>)> {
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
                    .map(|n| n.as_f64().unwrap() as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let y_gt = json["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| vec![n.as_f64().unwrap() as f32])
            .collect::<Vec<Vec<f32>>>();

        Ok((
            Tensor::new_matrix(xs, Some("xs".to_string())),
            Tensor::new_matrix(y_gt, Some("ys".to_string())),
        ))
    }
}

lazy_static! {
    static ref DEFAULT_DEVICE: Arc<Device> =
        Arc::new(Device::system_default().expect("Failed to create Metal device"));
    static ref DEFAULT_COMMAND_QUEUE: Arc<CommandQueue> =
        Arc::new(DEFAULT_DEVICE.new_command_queue());
}

pub enum ArrayType {
    Vector(Vec<f32>),
    Matrix(Vec<Vec<f32>>),
}

fn get_scalar_buffer(device: &Device, vec: &[f32]) -> Buffer {
    let data = vec.as_ptr() as *const _ as *const std::ffi::c_void;
    let length = vec.len() * std::mem::size_of::<f32>();
    device.new_buffer_with_data(
        data,
        length as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

pub mod gpu {
    use std::fs;

    use metal::{
        CommandBufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
        MTLSize,
    };

    pub fn gpu_pipeline(name: &str, device: &Device) -> ComputePipelineState {
        let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
            .expect("Failed to read Metal shader file");

        let function = device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function(name, None)
            .expect(&format!("Failed to get '{}' function from library", name));
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create compute pipeline state");

        return pipeline_state;
    }

    pub fn gpu_encoder<'a>(
        pipeline_state: ComputePipelineState,
        command_queue: &'a CommandQueue,
    ) -> (&'a ComputeCommandEncoderRef, &'a CommandBufferRef) {
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_state);
        (encoder, command_buffer)
    }

    pub fn gpu_setup<'a>(
        name: &'a str,
        device: &'a Device,
        command_queue: &'a CommandQueue,
    ) -> (&'a ComputeCommandEncoderRef, &'a CommandBufferRef) {
        let pipeline_state = gpu_pipeline(name, device);

        return gpu_encoder(pipeline_state, command_queue);
    }

    pub fn gpu_end(
        num_threadgroups: MTLSize,
        threads_per_group: MTLSize,
        encoder: &ComputeCommandEncoderRef,
        command_buffer: &CommandBufferRef,
    ) {
        encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    pub fn gpu_trace() -> metal::CaptureScope {
        let capture_scope = metal::CaptureManager::shared()
            .new_capture_scope_with_device(&metal::Device::system_default().unwrap());

        let capture_descriptor = metal::CaptureDescriptor::new();
        capture_descriptor.set_capture_scope(&capture_scope);
        capture_descriptor.set_output_url(std::path::Path::new(
            "/Users/cryogenic/general/learning/grad/framecapture.gputrace",
        ));
        capture_descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        let _ = metal::CaptureManager::shared().start_capture(&capture_descriptor);

        return capture_scope;
    }
}

impl fmt::Debug for ArrayType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayType::Vector(v) => f
                .debug_struct("Vector")
                .field("size", &v.len())
                .field("values", &v)
                .finish(),
            ArrayType::Matrix(m) => f
                .debug_struct("Matrix")
                .field("size", &m.len())
                .field("values", &m)
                .finish(),
        }
    }
}

impl From<Vec<f32>> for ArrayType {
    fn from(item: Vec<f32>) -> Self {
        ArrayType::Vector(item)
    }
}
impl From<Vec<Vec<f32>>> for ArrayType {
    fn from(item: Vec<Vec<f32>>) -> Self {
        ArrayType::Matrix(item)
    }
}

#[derive(Clone, Copy)]
pub struct Matrix {
    size: (usize, usize),
}

impl Matrix {
    pub fn new(vec: Vec<Vec<f32>>) -> Tensor<Matrix> {
        Tensor::new_matrix(vec, None)
    }

    pub fn new_with_label(vec: Vec<Vec<f32>>, label: String) -> Tensor<Matrix> {
        Tensor::new_matrix(vec, Some(label))
    }
}

#[derive(Clone, Copy)]
pub struct Vector {
    size: usize,
}

impl Vector {
    pub fn new(vec: Vec<f32>) -> Tensor<Vector> {
        Tensor::new_vector(vec)
    }
}

#[derive(Clone)]
pub enum TensorKind {
    Vector(Vector),
    Matrix(Matrix),
}

pub trait TensorType: Clone {
    type Kind: Into<TensorKind> + Clone;
}

impl TensorType for Matrix {
    type Kind = Matrix;
}

impl TensorType for Vector {
    type Kind = Vector;
}

impl From<Matrix> for TensorKind {
    fn from(item: Matrix) -> Self {
        TensorKind::Matrix(item)
    }
}

impl From<Vector> for TensorKind {
    fn from(item: Vector) -> Self {
        TensorKind::Vector(item)
    }
}

// type TensorOrScalar = Tensor<Vector> | Tensor<Matrix> | Scalar;

#[derive(Clone)]
pub struct Tensor<T: TensorType> {
    uuid: String,
    kind: T::Kind,
    pub label: String,
    buffer_size: usize,

    grad_buffer_size: Rc<RefCell<(usize, usize)>>,

    pub number: Rc<RefCell<Buffer>>, // GPU buffer for numbers
    pub grad: Rc<RefCell<Buffer>>,   // GPU buffer for gradients
    _prev: Option<Vec<Tensor<T>>>,
    _op: String,
    _backwards: Option<Rc<RefCell<Box<dyn FnMut()>>>>, // Adjusted to use Rc for cloning and made optional

    pub device: Arc<Device>,              // Metal device reference
    pub command_queue: Arc<CommandQueue>, // C
}

pub fn get_raw_matrix(buffer: &Rc<RefCell<Buffer>>, size: (usize, usize)) -> Vec<Vec<f32>> {
    let contents = buffer.borrow().contents() as *const f32;
    let mut data = Vec::with_capacity(size.0);
    for i in 0..size.0 {
        let row_base = i * size.1;
        let row = (0..size.1)
            .map(|j| unsafe { *contents.offset(row_base as isize + j as isize) })
            .collect::<Vec<f32>>();
        data.push(row);
    }

    data
}

pub fn get_matrix_from_buffer(buffer: &Buffer, size: (usize, usize)) -> Vec<Vec<f32>> {
    let contents = buffer.contents() as *const f32;
    let mut data = Vec::with_capacity(size.0);
    for i in 0..size.0 {
        let row_base = i * size.1;
        let row = (0..size.1)
            .map(|j| unsafe { *contents.offset(row_base as isize + j as isize) })
            .collect::<Vec<f32>>();
        data.push(row);
    }

    data
}

impl<T: TensorType> Tensor<T> {
    fn get_raw_val(&self, buffer: &Rc<RefCell<Buffer>>) -> ArrayType {
        let k = &self.kind.clone().into();

        match k {
            TensorKind::Vector(v) => {
                let contents = buffer.borrow().contents();
                let data =
                    unsafe { std::slice::from_raw_parts(contents as *const f32, v.size).to_vec() };
                ArrayType::from(data)
            }
            TensorKind::Matrix(m) => ArrayType::from(get_raw_matrix(buffer, m.size)),
        }
    }

    pub fn get_children(&self) -> Option<Vec<Tensor<T>>> {
        self._prev.clone()
    }

    pub fn get_op(&self) -> String {
        self._op.clone()
    }

    pub fn get_numbers(&self) -> ArrayType {
        self.get_raw_val(&self.number)
    }

    pub fn get_grad(&self) -> ArrayType {
        let size = self.grad_buffer_size.borrow();

        ArrayType::from(get_raw_matrix(&self.grad, *size))
    }

    fn set_backward(&mut self, backward: Rc<RefCell<Box<dyn FnMut()>>>) {
        self._backwards = Some(backward);
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

        println!("sorted {:?}", sorted);

        for node in sorted.iter().rev() {
            node.maybe_backwards();
        }
    }

    pub fn zero_grad(&self) {
        let sorted = utils::topological_sort(self);

        for node in sorted.iter().rev() {
            node.set_grad(0.0);
        }
    }

    pub fn set_grad(&self, grad: f32) {
        let grad_value = [grad];

        let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
            .expect("Failed to read Metal shader file");

        let set_grad_func = self
            .device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function("set_grad", None)
            .expect("Failed to get 'matrix_add' function from library");

        let set_grad_pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&set_grad_func)
            .expect("Failed to create compute pipeline state");

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&set_grad_pipeline_state);

        let grad_value_buffer = self.device.new_buffer_with_data(
            grad_value.as_ptr() as *const _,
            (grad_value.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_buffer_size = self.grad.borrow().length();
        let result_buffer = self.device.new_buffer_with_data(
            self.grad.borrow().contents(),
            result_buffer_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&result_buffer), 0);
        encoder.set_buffer(1, Some(&grad_value_buffer), 0);
        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        let num_threadgroups = MTLSize {
            width: (self.buffer_size as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.grad.replace(result_buffer);

        let s = match self.kind.clone().into() {
            TensorKind::Matrix(s) => s.size,
            _ => (0, 0), // Assuming a default or error case should be handled here
        };

        self.grad_buffer_size.replace(s);

        // println!(
        //     "set grad {:?} {:?}",
        //     get_raw_matrix(&self.grad, s),
        //     self.grad_buffer_size.borrow()
        // );
    }
}

impl Tensor<Vector> {
    fn size(&self) -> Result<usize, String> {
        let k = &self.kind.clone().into();

        if let TensorKind::Vector(v) = k {
            return Ok(v.size);
        }

        Err("Invalid tensor type".to_string())
    }

    fn new_vector(vec: Vec<f32>) -> Self {
        Self::new_f32(
            DEFAULT_DEVICE.clone(),
            vec,
            None,
            "".to_string(),
            "".to_string(),
        )
    }

    fn new_f32(
        device: Arc<Device>,
        numbers: Vec<f32>,
        children: Option<Vec<Tensor<Vector>>>,
        op: String,
        label: String,
    ) -> Self {
        let size = numbers.len();

        let number_buffer = device.new_buffer_with_data(
            unsafe { std::mem::transmute(numbers.as_ptr()) },
            (numbers.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let grad_buffer = device.new_buffer_with_data(
            unsafe { std::mem::transmute(vec![0f32; size].as_ptr()) },
            (numbers.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Tensor {
            uuid: Uuid::new_v4().to_string(),
            kind: Vector { size },
            buffer_size: size,
            number: Rc::new(RefCell::new(number_buffer)),
            grad: Rc::new(RefCell::new(grad_buffer)),
            grad_buffer_size: Rc::new(RefCell::new((0, 0))),
            device,
            command_queue: DEFAULT_COMMAND_QUEUE.clone(),
            _prev: children,
            _op: op,
            label,
            _backwards: None,
        }
    }

    fn new_vector_with_children(
        device: Arc<Device>,
        result_buffer: Buffer,
        children: Option<Vec<Tensor<Vector>>>,
        op: String,
        label: String,
        command_queue: Arc<CommandQueue>,
        size: usize,
    ) -> Self {
        let grad_buffer_size: usize = size as usize * std::mem::size_of::<f32>();
        let grad_buffer = device.new_buffer(
            grad_buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Tensor {
            uuid: Uuid::new_v4().to_string(),
            kind: Vector { size },
            buffer_size: size,
            number: Rc::new(RefCell::new(result_buffer)),
            grad: Rc::new(RefCell::new(grad_buffer)),
            grad_buffer_size: Rc::new(RefCell::new((0, 0))),
            label,
            _prev: children,
            _op: op,
            _backwards: None,
            device,
            command_queue,
        }
    }
}

impl Tensor<Matrix> {
    pub fn size(&self) -> Result<(usize, usize), String> {
        let k = &self.kind.clone().into();

        if let TensorKind::Matrix(v) = k {
            return Ok((v.size.0, v.size.1));
        }

        Err("Invalid tensor type".to_string())
    }

    pub fn get_preds_matrix(&self) -> Vec<Vec<f32>> {
        let numbers = self.get_numbers();

        match numbers {
            ArrayType::Matrix(numbers) => numbers,
            _ => panic!("Invalid tensor type"),
        }
    }

    pub fn pow(self, other: f32) -> Self {
        let command_queue = self.command_queue.clone();

        let (encoder, command_buffer) = gpu::gpu_setup("matrix_pow", &self.device, &command_queue);

        let result_buffer_size = u32size(self.size().unwrap());

        let result_buffer = self.device.new_buffer(
            result_buffer_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let exponent_buffer = get_scalar_buffer(&self.device, &[other]);

        encoder.set_buffer(0, Some(&self.number.borrow()), 0);
        encoder.set_buffer(1, Some(&exponent_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: (self.number.borrow().length() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        gpu_end(
            num_threadgroups,
            threads_per_group,
            &encoder,
            &command_buffer,
        );

        exponent_buffer.set_purgeable_state(MTLPurgeableState::Empty);

        let mut out = Tensor::new_with_children(
            self.device.clone(),
            result_buffer,
            Some(vec![self.clone()]),
            "pow".to_string(),
            format!("pow({})", self.label),
            self.command_queue.clone(),
            self.size().unwrap(),
        );

        let pow_backward_pipeline_state =
            gpu_pipeline("matrix_pow_backwards", &self.device.clone());

        let command_queue = self.command_queue.clone();
        let device: Arc<Device> = self.device.clone();

        let out_grad_clone = Rc::clone(&out.grad);
        let self_numbers_clone = Rc::clone(&self.number);
        let self_grad_clone = Rc::clone(&self.grad);

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let (encoder, command_buffer) =
                gpu_encoder(pow_backward_pipeline_state.clone(), &command_queue);

            let result_grad_buffer = device.new_buffer(
                u32size(self.size().unwrap()),
                metal::MTLResourceOptions::StorageModeShared,
            );

            let exponent_buffer = get_scalar_buffer(&device, &[other]);

            encoder.set_buffer(0, Some(&out_grad_clone.borrow()), 0);
            encoder.set_buffer(1, Some(&self_numbers_clone.borrow()), 0);
            encoder.set_buffer(2, Some(&exponent_buffer), 0);
            encoder.set_buffer(3, Some(&result_grad_buffer), 0);
            let threads_per_group = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };

            let num_threadgroups = MTLSize {
                width: ((self.size().unwrap().0 * self.size().unwrap().1) as u64 + 63) / 64,
                height: 1,
                depth: 1,
            };

            gpu_end(
                num_threadgroups,
                threads_per_group,
                &encoder,
                &command_buffer,
            );

            self_grad_clone
                .borrow()
                .set_purgeable_state(MTLPurgeableState::Empty);
            exponent_buffer.set_purgeable_state(MTLPurgeableState::Empty);

            self_grad_clone.replace(result_grad_buffer);
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);
        out
    }

    pub fn relu(self) -> Tensor<Matrix> {
        let command_queue = self.command_queue.clone();

        let (encoder, command_buffer) = gpu_setup("matrix_relu", &self.device, &command_queue);

        let result_buffer_size = u32size(self.size().unwrap());

        let result_buffer = self.device.new_buffer(
            result_buffer_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number.borrow()), 0);
        encoder.set_buffer(1, Some(&result_buffer), 0);

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        let num_threadgroups = MTLSize {
            width: (self.number.borrow().length() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        gpu_end(
            num_threadgroups,
            threads_per_group,
            &encoder,
            &command_buffer,
        );

        let mut out = Tensor::new_with_children(
            self.device.clone(),
            result_buffer,
            Some(vec![self.clone()]),
            "relu".to_string(),
            format!("relu({})", self.label),
            self.command_queue.clone(),
            self.size().unwrap(),
        );

        let relu_backward_pipeline_state =
            gpu_pipeline("matrix_relu_backwards", &self.device.clone());

        let command_queue = self.command_queue.clone();

        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);

        let device = self.device.clone();

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let (encoder, command_buffer) =
                gpu_encoder(relu_backward_pipeline_state.clone(), &command_queue);

            let result_grad_buffer = device.new_buffer(
                // TODO: this is wrong
                u32size(self.size().unwrap()),
                metal::MTLResourceOptions::StorageModeShared,
            );

            encoder.set_buffer(0, Some(&out_grad_clone.borrow()), 0);
            encoder.set_buffer(1, Some(&self_grad_clone.borrow()), 0);
            encoder.set_buffer(2, Some(&result_grad_buffer), 0);

            gpu_end(
                num_threadgroups,
                threads_per_group,
                &encoder,
                &command_buffer,
            );

            self_grad_clone.replace(result_grad_buffer);
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);
        out
    }

    pub fn tanh(self) -> Self {
        let device = self.device.clone();
        let command_queue = self.command_queue.clone();

        let (encoder, command_buffer) = gpu_setup("matrix_tanh", &device, &command_queue);

        let result_buffer = self.device.new_buffer(
            self.number.borrow().length() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number.borrow()), 0);
        encoder.set_buffer(1, Some(&result_buffer), 0);

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        let num_threadgroups = MTLSize {
            width: (self.number.borrow().length() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        gpu_end(
            num_threadgroups,
            threads_per_group,
            &encoder,
            &command_buffer,
        );

        let mut out = Tensor::new_with_children(
            device,
            result_buffer,
            Some(vec![self.clone()]),
            "tanh".to_string(),
            format!("tanh({})", self.label),
            command_queue,
            (self.size().unwrap().0, self.size().unwrap().1),
        );

        let tanh_backward_pipeline_state =
            gpu_pipeline("matrix_tanh_backwards", &self.device.clone());

        let command_queue = self.command_queue.clone();

        let out_grad_clone = Rc::clone(&out.grad);
        let out_number_clone = Rc::clone(&out.number);
        let self_grad_clone = Rc::clone(&self.grad);
        let grad_buffer_size = Rc::clone(&self.grad_buffer_size);
        let size_clone = self.size().unwrap().clone();

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let (encoder, command_buffer) =
                gpu_encoder(tanh_backward_pipeline_state.clone(), &command_queue);

            let result_buffer = self.device.new_buffer(
                (size_clone.0 * size_clone.1 * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            encoder.set_buffer(0, Some(&out_number_clone.borrow()), 0);
            encoder.set_buffer(1, Some(&self_grad_clone.borrow()), 0);
            encoder.set_buffer(2, Some(&out_grad_clone.borrow()), 0); // The result is stored back in self_grad
            encoder.set_buffer(3, Some(&result_buffer), 0);

            gpu_end(
                num_threadgroups,
                threads_per_group,
                &encoder,
                &command_buffer,
            );

            self_grad_clone
                .borrow()
                .set_purgeable_state(MTLPurgeableState::Empty);

            self_grad_clone.replace(result_buffer);

            grad_buffer_size.replace(size_clone);
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);
        out
    }

    fn new_matrix(vec: Vec<Vec<f32>>, label: Option<String>) -> Self {
        Self::new_f32(
            DEFAULT_DEVICE.clone(),
            vec,
            None,
            "".to_string(),
            label.unwrap_or("".to_string()),
        )
    }

    fn new_f32(
        device: Arc<Device>,
        numbers: Vec<Vec<f32>>,
        children: Option<Vec<Tensor<Matrix>>>,
        op: String,
        label: String,
    ) -> Self {
        let size = (numbers.len(), numbers[0].len());

        let buffer_size = size.0 as usize * size.1 as usize * std::mem::size_of::<f32>();

        let numbers_flat: Vec<f32> = numbers.into_iter().flatten().collect();

        // Calculate the byte size of the flat numbers vector.
        let buffer_size_bytes = (numbers_flat.len() * std::mem::size_of::<f32>()) as u64;

        // println!("buffer size bytes {:?} vs buffer_size", buffer_size_bytes, bu);
        // Create the number buffer directly from the slice, without using transmute.
        let number_buffer = device.new_buffer_with_data(
            numbers_flat.as_ptr() as *const std::ffi::c_void, // Cast to *const c_void as expected by new_buffer_with_data
            buffer_size_bytes,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create the gradient buffer. Assuming the gradient buffer should be the same size as the number buffer.
        // Calculate the size of the gradient buffer in terms of f32 elements.

        // Create a vector of zeros for initializing the gradient buffer.
        let grad_zeros = vec![0f32; buffer_size];

        // Create the gradient buffer with the zeros vector.
        let grad_buffer = device.new_buffer_with_data(
            grad_zeros.as_ptr() as *const std::ffi::c_void, // Cast to *const c_void
            (buffer_size * std::mem::size_of::<f32>()) as u64, // Calculate the byte size
            metal::MTLResourceOptions::StorageModeShared,
        );

        Tensor {
            uuid: Uuid::new_v4().to_string(),
            kind: Matrix { size },
            buffer_size,
            number: Rc::new(RefCell::new(number_buffer)),
            grad: Rc::new(RefCell::new(grad_buffer)),
            grad_buffer_size: Rc::new(RefCell::new((0, 0))),
            device,
            command_queue: DEFAULT_COMMAND_QUEUE.clone(),
            _prev: children,
            _op: op,
            label,
            _backwards: None,
        }
    }

    fn new_with_children(
        device: Arc<Device>,
        result_buffer: Buffer,
        children: Option<Vec<Tensor<Matrix>>>,
        op: String,
        label: String,
        command_queue: Arc<CommandQueue>,
        size: (usize, usize),
    ) -> Self {
        let grad_buffer_size: usize = size.0 * size.1 * std::mem::size_of::<f32>();
        let grad_buffer = device.new_buffer(
            grad_buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Tensor {
            uuid: Uuid::new_v4().to_string(),
            kind: Matrix { size },
            number: Rc::new(RefCell::new(result_buffer)),
            grad: Rc::new(RefCell::new(grad_buffer)),
            grad_buffer_size: Rc::new(RefCell::new((0, 0))),
            label,
            buffer_size: grad_buffer_size,
            _prev: children,
            _op: op,
            _backwards: None,
            device,
            command_queue,
        }
    }
}

use std::ops::{Add, Mul, Sub};

/*
// impl Add<&Tensor<Vector>> for &Tensor<Vector> {
//     type Output = Tensor<Vector>;

//     fn add(self, rhs: &Tensor<Vector>) -> Tensor<Vector> {
//         let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
//             .expect("Failed to read Metal shader file");

//         let add_function = self
//             .device
//             .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
//             .expect("Failed to create library from source")
//             .get_function("vector_add", None)
//             .expect("Failed to get 'vector_add' function from library");

//         let add_pipeline_state = self
//             .device
//             .new_compute_pipeline_state_with_function(&add_function)
//             .expect("Failed to create compute pipeline state");

//         let command_buffer = self.command_queue.new_command_buffer();
//         let encoder = command_buffer.new_compute_command_encoder();
//         encoder.set_compute_pipeline_state(&add_pipeline_state);

//         // Create the result buffer before using it
//         let result_buffer = self.device.new_buffer(
//             self.number.length() as u64,
//             metal::MTLResourceOptions::StorageModeShared,
//         );

//         encoder.set_buffer(0, Some(&self.number), 0);
//         encoder.set_buffer(1, Some(&rhs.number), 0);
//         encoder.set_buffer(2, Some(&result_buffer), 0); // Store the result in the result buffer

//         let threads_per_group = MTLSize {
//             width: 64,
//             height: 1,
//             depth: 1,
//         };
//         let num_threadgroups = MTLSize {
//             width: (self.number.length() / std::mem::size_of::<f32>() as u64 + 63) / 64,
//             height: 1,
//             depth: 1,
//         };

//         encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

//         encoder.end_encoding();
//         command_buffer.commit();
//         command_buffer.wait_until_completed();

//         // Create a new Tensor to hold the result

//         let mut out = Tensor::new_vector_with_children(
//             DEFAULT_DEVICE.clone(),
//             Arc::new(result_buffer),
//             Some(vec![self.clone(), rhs.clone()]),
//             "+".to_string(),
//             format!("{}+{}", self.label, rhs.label),
//             self.command_queue.clone(),
//             self.size().unwrap(),
//         );

//         let add_backward_pass = self
//             .device
//             .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
//             .expect("Failed to create library from source")
//             .get_function("vector_add_backwards", None)
//             .expect("Failed to get 'vector_add' function from library");

//         let device = self.device.clone();
//         let command_queue = self.command_queue.clone();

//         let out_grad_clone = out.grad.clone();
//         let self_grad_clone = self.grad.clone();
//         let rhs_grad_clone = rhs.grad.clone();

//         let backward_pass = Rc::new(RefCell::new(Box::new(move || {
//             let add_pipeline_state_self = device
//                 .new_compute_pipeline_state_with_function(&add_backward_pass)
//                 .expect("Failed to create compute pipeline state for self");

//             let command_buffer_self = command_queue.new_command_buffer();
//             let encoder_self = command_buffer_self.new_compute_command_encoder();
//             encoder_self.set_compute_pipeline_state(&add_pipeline_state_self);

//             encoder_self.set_buffer(0, Some(&self_grad_clone), 0);
//             encoder_self.set_buffer(1, Some(&out_grad_clone), 0);
//             encoder_self.set_buffer(2, Some(&self_grad_clone), 0); // The result is stored back in self_grad

//             encoder_self.dispatch_thread_groups(num_threadgroups, threads_per_group);

//             encoder_self.end_encoding();
//             command_buffer_self.commit();
//             command_buffer_self.wait_until_completed();

//             let add_pipeline_state_rhs = device
//                 .new_compute_pipeline_state_with_function(&add_backward_pass)
//                 .expect("Failed to create compute pipeline state for rhs");

//             let command_buffer_rhs = command_queue.new_command_buffer();
//             let encoder_rhs = command_buffer_rhs.new_compute_command_encoder();
//             encoder_rhs.set_compute_pipeline_state(&add_pipeline_state_rhs);

//             encoder_rhs.set_buffer(0, Some(&rhs_grad_clone), 0);
//             encoder_rhs.set_buffer(1, Some(&out_grad_clone), 0);
//             encoder_rhs.set_buffer(2, Some(&rhs_grad_clone), 0); // The result is stored back in rhs_grad

//             encoder_rhs.dispatch_thread_groups(num_threadgroups, threads_per_group);

//             encoder_rhs.end_encoding();
//             command_buffer_rhs.commit();
//             command_buffer_rhs.wait_until_completed();
//         }) as Box<dyn FnMut()>));

//         out.set_backward(backward_pass);

//         out
//     }
// }

// impl Mul<&Tensor<Vector>> for &Tensor<Vector> {
//     type Output = Tensor<Vector>;

//     fn mul(self, rhs: &Tensor<Vector>) -> Tensor<Vector> {
//         let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
//             .expect("Failed to read Metal shader file");

//         let mul_function = self
//             .device
//             .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
//             .expect("Failed to create library from source")
//             .get_function("vector_mul", None)
//             .expect("Failed to get 'vector_mul' function from library");
//         let mul_pipeline_state = self
//             .device
//             .new_compute_pipeline_state_with_function(&mul_function)
//             .expect("Failed to create compute pipeline state");

//         let command_buffer = self.command_queue.new_command_buffer();
//         let encoder = command_buffer.new_compute_command_encoder();
//         encoder.set_compute_pipeline_state(&mul_pipeline_state);

//         // Create the result buffer before using it
//         let result_buffer = self.device.new_buffer(
//             self.number.length() as u64,
//             metal::MTLResourceOptions::StorageModeShared,
//         );

//         encoder.set_buffer(0, Some(&self.number), 0);
//         encoder.set_buffer(1, Some(&rhs.number), 0);
//         encoder.set_buffer(2, Some(&result_buffer), 0); // Store the result in the result buffer

//         let threads_per_group = MTLSize {
//             width: 64,
//             height: 1,
//             depth: 1,
//         };
//         let num_threadgroups = MTLSize {
//             width: (self.number.length() / std::mem::size_of::<f32>() as u64 + 63) / 64,
//             height: 1,
//             depth: 1,
//         };

//         encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

//         encoder.end_encoding();
//         command_buffer.commit();
//         command_buffer.wait_until_completed();

//         // Create a new Tensor to hold the result

//         let mut out = Tensor::new_vector_with_children(
//             DEFAULT_DEVICE.clone(),
//             Arc::new(result_buffer),
//             Some(vec![self.clone(), rhs.clone()]),
//             "*".to_string(),
//             format!("{}*{}", self.label, rhs.label),
//             self.command_queue.clone(),
//             self.size().unwrap(),
//         );

//         let mul_backward_pass = self
//             .device
//             .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
//             .expect("Failed to create library from source")
//             .get_function("vector_mul_backwards", None)
//             .expect("Failed to get 'vector_mul' function from library");

//         let device = self.device.clone();
//         let command_queue = self.command_queue.clone();

//         let out_grad_clone = out.grad.clone();
//         let self_grad_clone = self.grad.clone();
//         let rhs_grad_clone = rhs.grad.clone();

//         let self_number_clone = self.number.clone();
//         let rhs_number_clone = rhs.number.clone();

//         let backward_pass = Rc::new(RefCell::new(Box::new(move || {
//             let mul_pipeline_state_self = device
//                 .new_compute_pipeline_state_with_function(&mul_backward_pass)
//                 .expect("Failed to create compute pipeline state for self");

//             let command_buffer_self = command_queue.new_command_buffer();
//             let encoder_self = command_buffer_self.new_compute_command_encoder();
//             encoder_self.set_compute_pipeline_state(&mul_pipeline_state_self);

//             encoder_self.set_buffer(0, Some(&self_grad_clone), 0);
//             encoder_self.set_buffer(1, Some(&rhs_number_clone), 0);
//             encoder_self.set_buffer(2, Some(&out_grad_clone), 0);
//             encoder_self.set_buffer(3, Some(&self_grad_clone), 0); // The result is stored back in self_grad

//             encoder_self.dispatch_thread_groups(num_threadgroups, threads_per_group);

//             encoder_self.end_encoding();
//             command_buffer_self.commit();
//             command_buffer_self.wait_until_completed();

//             let mul_pipeline_state_rhs = device
//                 .new_compute_pipeline_state_with_function(&mul_backward_pass)
//                 .expect("Failed to create compute pipeline state for rhs");
//             let command_buffer_rhs = command_queue.new_command_buffer();
//             let encoder_rhs = command_buffer_rhs.new_compute_command_encoder();
//             encoder_rhs.set_compute_pipeline_state(&mul_pipeline_state_rhs);

//             encoder_rhs.set_buffer(0, Some(&rhs_grad_clone), 0);
//             encoder_rhs.set_buffer(1, Some(&self_number_clone), 0);
//             encoder_rhs.set_buffer(2, Some(&out_grad_clone), 0);
//             encoder_rhs.set_buffer(3, Some(&rhs_grad_clone), 0); // The result is stored back in rhs_grad

//             encoder_rhs.dispatch_thread_groups(num_threadgroups, threads_per_group);

//             encoder_rhs.end_encoding();
//             command_buffer_rhs.commit();
//             command_buffer_rhs.wait_until_completed();
//         }) as Box<dyn FnMut()>));

//         out.set_backward(backward_pass);

//         out
//     }
// }

 */

impl<T: TensorType> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        // Define your tolerance for floating-point comparison
        let tolerance = 1e-6;

        match (self.get_numbers(), other.get_numbers()) {
            (ArrayType::Matrix(self_nums), ArrayType::Matrix(other_nums)) => {
                if self_nums.len() != other_nums.len() {
                    return false;
                }
                for (self_row, other_row) in self_nums.iter().zip(other_nums.iter()) {
                    if self_row.len() != other_row.len() {
                        return false;
                    }
                    for (&self_val, &other_val) in self_row.iter().zip(other_row.iter()) {
                        if (self_val - other_val).abs() > tolerance {
                            return false;
                        }
                    }
                }
            }
            (ArrayType::Vector(self_nums), ArrayType::Vector(other_nums)) => {
                if self_nums.len() != other_nums.len() {
                    return false;
                }
                for (&self_val, &other_val) in self_nums.iter().zip(other_nums.iter()) {
                    if (self_val - other_val).abs() > tolerance {
                        return false;
                    }
                }
            }
            _ => return false,
        }

        self._op == other._op
    }
}

impl<T: TensorType> Eq for Tensor<T> {}

use std::hash::{Hash, Hasher};

use self::gpu::{gpu_encoder, gpu_end, gpu_pipeline, gpu_setup, gpu_trace};

impl<T: TensorType> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash each discretized number and the operation
        let numbers = self.get_numbers();

        match numbers {
            ArrayType::Matrix(nums) => {
                for num in nums.iter() {
                    for &n in num.iter() {
                        let discretized = (n * 1e6).round() as i64;
                        discretized.hash(state);
                    }
                }
                self._op.hash(state);
            }
            ArrayType::Vector(nums) => {
                for &num in nums.iter() {
                    let discretized = (num * 1e6).round() as i64;
                    discretized.hash(state);
                }
                self._op.hash(state);
            } // _ => panic!("Invalid tensor type"),
        }

        // Handle _prev hashing if needed, possibly by hashing the length or similar
    }
}

impl Add<&Tensor<Matrix>> for &Tensor<Matrix> {
    type Output = Tensor<Matrix>;

    fn add(self, rhs: &Tensor<Matrix>) -> Tensor<Matrix> {
        let capture_device = gpu_trace();

        capture_device.begin_scope();

        let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
            .expect("Failed to read Metal shader file");

        let add_function = self
            .device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function("matrix_add", None)
            .expect("Failed to get 'matrix_add' function from library");

        let add_pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&add_function)
            .expect("Failed to create compute pipeline state");

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&add_pipeline_state);

        // Create the result buffer before using it
        let result_buffer = self.device.new_buffer(
            (self.number.borrow().length() as u64) * 2, // Assuming the result matrix has the same size as input matrices
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number.borrow()), 0);
        encoder.set_buffer(1, Some(&rhs.number.borrow()), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);

        let (num_rows, num_cols) = self.size().unwrap();

        let (rhs_rows, _) = rhs.size().unwrap();
        // if !(rhs_rows == 1 && rhs_cols == self_cols)
        //     && !(self_rows == rhs_rows && self_cols == rhs_cols)
        // {
        //     panic!("Right-hand side must be a 1xm vector matching the number of columns of the left-hand side matrix, or both matrices must have the same dimensions");
        // }
        let is_row_wise = if rhs_rows == 1 { 1 } else { 0 };

        let size = [num_rows as i32, num_cols as i32, is_row_wise];

        let size_buffer = self.device.new_buffer_with_data(
            unsafe { std::mem::transmute(size.as_ptr()) },
            (std::mem::size_of::<[i32; 3]>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(3, Some(&size_buffer), 0);

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: ((num_rows * num_cols) as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Create a new Tensor to hold the result
        let mut out = Tensor::new_with_children(
            DEFAULT_DEVICE.clone(),
            result_buffer,
            Some(vec![self.clone(), rhs.clone()]),
            "+".to_string(),
            format!("({})+({})", self.label, rhs.label),
            self.command_queue.clone(),
            (num_rows, num_cols),
        );

        capture_device.end_scope();

        let add_backward_pass = self
            .device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function("matrix_add_backwards", None)
            .expect("Failed to get 'vector_add' function from library");

        let command_queue = self.command_queue.clone();

        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);
        let rhs_grad_clone = Rc::clone(&rhs.grad);

        let self_grad_buffer_size = Rc::clone(&self.grad_buffer_size);
        let rhs_grad_buffer_size = Rc::clone(&rhs.grad_buffer_size);

        let device = self.device.clone();
        let lhs_size = self.size().unwrap().clone();
        let rhs_size = rhs.size().unwrap().clone();

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let add_pipeline_state_self = device
                .new_compute_pipeline_state_with_function(&add_backward_pass)
                .expect("Failed to create compute pipeline state for self");

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&add_pipeline_state_self);

            let lhs_grad_buffer = device.new_buffer(
                (lhs_size.0 * lhs_size.1 * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let rhs_grad_buffer = device.new_buffer(
                (rhs_size.0 * rhs_size.1 * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            encoder.set_buffer(0, Some(&self_grad_clone.borrow()), 0);
            encoder.set_buffer(1, Some(&rhs_grad_clone.borrow()), 0);
            encoder.set_buffer(2, Some(&out_grad_clone.borrow()), 0);
            encoder.set_buffer(3, Some(&lhs_grad_buffer), 0); // The result is stored back in self_grad
            encoder.set_buffer(4, Some(&rhs_grad_buffer), 0); // The result is stored back in rhs_grad
            encoder.set_buffer(5, Some(&size_buffer), 0);

            encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            self_grad_clone
                .borrow()
                .set_purgeable_state(MTLPurgeableState::Empty);
            self_grad_clone.replace(lhs_grad_buffer);
            self_grad_buffer_size.replace(lhs_size);
            rhs_grad_clone
                .borrow()
                .set_purgeable_state(MTLPurgeableState::Empty);
            rhs_grad_clone.replace(rhs_grad_buffer);
            rhs_grad_buffer_size.replace(rhs_size);
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);
        out
    }
}

/*
// // add matrix + vector or vector + matrix
// impl<T: TensorType, U: TensorType> Add<Tensor<U>> for Tensor<T> {
//     type Output = Tensor<Matrix>;

//     fn add(self, rhs: Tensor<U>) -> Self::Output {
//         let lhs_size = self.kind.clone().into();
//         let rhs_size = rhs.kind.clone().into();

//         let matrix_size = match (&lhs_size, &rhs_size) {
//             (TensorKind::Matrix(lhs_matrix), TensorKind::Vector(_)) => Some(lhs_matrix.size),
//             (TensorKind::Vector(_), TensorKind::Matrix(rhs_matrix)) => Some(rhs_matrix.size),
//             _ => None,
//         };

//         let vector_size = match (&lhs_size, &rhs_size) {
//             (TensorKind::Vector(lhs_vector), TensorKind::Vector(_)) => Some(lhs_vector.size),
//             (TensorKind::Matrix(_), TensorKind::Vector(rhs_vector)) => Some(rhs_vector.size),
//             _ => None,
//         };

//         let is_okay: Result<bool, String> = if let Some(matrix_size) = matrix_size {
//             match vector_size {
//                 Some(size) if matrix_size.1 != size => {
//                     Err("Matrix column size must match Vector size for addition.".into())
//                 }
//                 Some(_) => Ok(true),
//                 None => Err("Vector size is undefined.".into()),
//             }
//         } else {
//             Err("Matrix size is undefined.".into())
//         };

//         if let Err(e) = is_okay {
//             panic!("{}", e);
//         }

//         fn get_matrix<T: TensorType, U: TensorType>(
//             lhs: &Tensor<T>,
//             rhs: &Tensor<U>,
//         ) -> Option<(Arc<Buffer>, Arc<Buffer>)> {
//             if let TensorKind::Matrix(_) = lhs.kind.clone().into() {
//                 return Some((lhs.number.clone(), lhs.grad.clone()));
//             } else if let TensorKind::Matrix(_) = rhs.kind.clone().into() {
//                 return Some((rhs.number.clone(), rhs.grad.clone()));
//             }
//             None
//         }

//         fn get_vector<T: TensorType, U: TensorType>(
//             lhs: &Tensor<T>,
//             rhs: &Tensor<U>,
//         ) -> Option<(Arc<Buffer>, Arc<Buffer>)> {
//             if let TensorKind::Vector(_) = lhs.kind.clone().into() {
//                 return Some((lhs.number.clone(), lhs.grad.clone()));
//             } else if let TensorKind::Vector(_) = rhs.kind.clone().into() {
//                 return Some((rhs.number.clone(), rhs.grad.clone()));
//             }
//             None
//         }

//         let (matrix_val, matrix_grad) = get_matrix(&self, &rhs).unwrap();
//         let (vector_val, vector_grad) = get_vector(&self, &rhs).unwrap();

//         let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
//             .expect("Failed to read Metal shader file");

//         let matrix_add_function = self
//             .device
//             .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
//             .expect("Failed to create library from source")
//             .get_function("matrix_add", None)
//             .expect("Failed to get 'matrix_add' function from library");

//         let matrix_add_pipeline_state = self
//             .device
//             .new_compute_pipeline_state_with_function(&matrix_add_function)
//             .expect("Failed to create compute pipeline state");

//         let command_buffer = self.command_queue.new_command_buffer();
//         let encoder = command_buffer.new_compute_command_encoder();
//         encoder.set_compute_pipeline_state(&matrix_add_pipeline_state);

//         // Create the result buffer and initialize it with the values from the matrix
//         let result_buffer_size = (self.number.length() as u64) * std::mem::size_of::<f32>() as u64;
//         let result_buffer = self.device.new_buffer_with_data(
//             matrix_val.contents(),
//             result_buffer_size,
//             metal::MTLResourceOptions::StorageModeShared,
//         );

//         encoder.set_buffer(0, Some(&matrix_val), 0);
//         encoder.set_buffer(1, Some(&vector_val), 0); // Assuming vector_val is treated as a matrix with row size 1
//         encoder.set_buffer(2, Some(&result_buffer), 0); // Store the result in the result buffer

//         // Assuming row size is set to 1 for simplification
//         let matrix_sizes: [u32; 2] = [
//             1,
//             vector_val.length() as u32 / std::mem::size_of::<f32>() as u32,
//         ];
//         let matrix_sizes_buffer = self.device.new_buffer_with_data(
//             unsafe { std::mem::transmute(&matrix_sizes as *const u32) },
//             (matrix_sizes.len() * std::mem::size_of::<u32>()) as u64,
//             metal::MTLResourceOptions::StorageModeShared,
//         );

//         encoder.set_buffer(3, Some(&matrix_sizes_buffer), 0);

//         let threads_per_group = MTLSize {
//             width: 64,
//             height: 1,
//             depth: 1,
//         };
//         let num_threadgroups = MTLSize {
//             width: (vector_val.length() + 63) / 64,
//             height: 1,
//             depth: 1,
//         };

//         encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

//         encoder.end_encoding();
//         command_buffer.commit();
//         command_buffer.wait_until_completed();

//         // Create a new Tensor to hold the result
//         let mut out = Tensor::new_with_children(
//             DEFAULT_DEVICE.clone(),
//             Arc::new(result_buffer),
//             Some(vec![self.clone(), rhs.clone()]),
//             "+".to_string(),
//             format!("{}+{}", self.label, rhs.label),
//             self.command_queue.clone(),
//             matrix_size.unwrap(),
//         );

//         let add_backward_pass = self
//             .device
//             .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
//             .expect("Failed to create library from source")
//             .get_function("vector_add_backwards", None)
//             .expect("Failed to get 'vector_add' function from library");

//         let device = self.device.clone();
//         let command_queue = self.command_queue.clone();

//         let out_grad_clone = out.grad.clone();
//         let self_grad_clone = self.grad.clone();
//         let rhs_grad_clone = rhs.grad.clone();

//         let backward_pass = Rc::new(RefCell::new(Box::new(move || {
//             let add_pipeline_state_self = device
//                 .new_compute_pipeline_state_with_function(&add_backward_pass)
//                 .expect("Failed to create compute pipeline state for self");

//             let command_buffer = command_queue.new_command_buffer();
//             let encoder = command_buffer.new_compute_command_encoder();
//             encoder.set_compute_pipeline_state(&add_pipeline_state_self);

//             encoder.set_buffer(0, Some(&self_grad_clone), 0);
//             encoder.set_buffer(1, Some(&rhs_grad_clone), 0);
//             encoder.set_buffer(2, Some(&out_grad_clone), 0);
//             encoder.set_buffer(3, Some(&self_grad_clone), 0); // The result is stored back in self_grad
//             encoder.set_buffer(4, Some(&rhs_grad_clone), 0); // The result is stored back in rhs_grad
//             encoder.set_buffer(5, Some(&matrix_sizes_buffer), 0);

//             encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

//             encoder.end_encoding();
//             command_buffer.commit();
//             command_buffer.wait_until_completed();
//         }) as Box<dyn FnMut()>));

//         out.set_backward(backward_pass);
//         out
//     }
// }

 */

fn compute_output_matrix_size(lhs: (usize, usize), rhs: (usize, usize)) -> (usize, usize) {
    let (lhs_rows, _) = lhs;
    let (_, rhs_cols) = rhs;

    let output_rows = lhs_rows;
    let output_cols = rhs_cols;

    (output_rows, output_cols)
}

pub fn u32size(size: (usize, usize)) -> u64 {
    let (rows, cols) = size;
    ((rows * cols) * std::mem::size_of::<f32>()) as u64
}

impl Mul<f32> for &Tensor<Matrix> {
    type Output = Tensor<Matrix>;

    fn mul(self, rhs: f32) -> Tensor<Matrix> {
        let capture_device = gpu_trace();

        capture_device.begin_scope();

        let (encoder, command_buffer) =
            gpu_setup("matrix_scalar_mul", &self.device, &self.command_queue);

        let scalar_buffer = self.device.new_buffer_with_data(
            unsafe { std::mem::transmute(&rhs) },
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_buffer_size = u32size(self.size().unwrap());

        let result_buffer = self.device.new_buffer(
            result_buffer_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number.borrow()), 0);
        encoder.set_buffer(1, Some(&scalar_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: (self.number.borrow().length() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        gpu_end(
            num_threadgroups,
            threads_per_group,
            &encoder,
            &command_buffer,
        );

        // Create a new Tensor to hold the result
        let mut out = Tensor::new_with_children(
            DEFAULT_DEVICE.clone(),
            result_buffer,
            Some(vec![self.clone()]),
            "*".to_string(),
            format!("({})*({})", self.label, rhs),
            self.command_queue.clone(),
            self.size().unwrap(),
        );

        capture_device.end_scope();

        let mul_backwards_pipeline_state =
            gpu_pipeline("matrix_scalar_mul_backwards", &self.device);

        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);
        let scalar = rhs.clone();

        let device_clone = Arc::clone(&self.device);

        let result_grad_buffer_size_clone = result_buffer_size.clone();

        let command_queue = self.command_queue.clone();

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let (encoder, command_buffer) =
                gpu_encoder(mul_backwards_pipeline_state.clone(), &command_queue);

            let result_grad_buffer = device_clone.new_buffer(
                result_grad_buffer_size_clone,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let scalar_buffer = device_clone.new_buffer_with_data(
                unsafe { std::mem::transmute(&scalar) },
                std::mem::size_of::<f32>() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            encoder.set_buffer(0, Some(&out_grad_clone.borrow()), 0);
            encoder.set_buffer(1, Some(&scalar_buffer), 0);
            encoder.set_buffer(2, Some(&result_grad_buffer), 0);

            gpu_end(
                num_threadgroups,
                threads_per_group,
                &encoder,
                &command_buffer,
            );

            self_grad_clone.replace(result_grad_buffer);
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);
        out
    }
}

impl Add<f32> for &Tensor<Matrix> {
    type Output = Tensor<Matrix>;

    fn add(self, other: f32) -> Tensor<Matrix> {
        let device = Arc::clone(&self.device);
        let command_queue = Arc::clone(&self.command_queue);

        let (encoder, command_buffer) = gpu_setup("matrix_add_scalar", &device, &command_queue);

        let scalar_buffer = device.new_buffer_with_data(
            unsafe { std::mem::transmute(&other) },
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_buffer_size = u32size(self.size().unwrap());

        let result_buffer = device.new_buffer(
            result_buffer_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number.borrow()), 0);
        encoder.set_buffer(1, Some(&scalar_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: (self.number.borrow().length() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        gpu_end(
            num_threadgroups,
            threads_per_group,
            &encoder,
            &command_buffer,
        );

        let mut out = Tensor::new_with_children(
            device,
            result_buffer,
            Some(vec![self.clone()]),
            "+".to_string(),
            format!("({})+({})", self.label, other),
            self.command_queue.clone(),
            self.size().unwrap(),
        );

        scalar_buffer.set_purgeable_state(MTLPurgeableState::Empty);

        let device = Arc::clone(&self.device);
        let command_queue = Arc::clone(&self.command_queue);
        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);

        let size = self.size().unwrap();

        let add_scalar_backwards_pipeline_state =
            gpu_pipeline("matrix_add_scalar_backwards", &device);

        let backward_pass = {
            Rc::new(RefCell::new(Box::new(move || {
                let (encoder, command_buffer) =
                    gpu_encoder(add_scalar_backwards_pipeline_state.clone(), &command_queue);

                let result_grad_buffer =
                    device.new_buffer(u32size(size), metal::MTLResourceOptions::StorageModeShared);

                encoder.set_buffer(0, Some(&out_grad_clone.borrow()), 0);
                encoder.set_buffer(1, Some(&self_grad_clone.borrow()), 0);
                encoder.set_buffer(2, Some(&result_grad_buffer), 0);

                gpu_end(
                    num_threadgroups,
                    threads_per_group,
                    &encoder,
                    &command_buffer,
                );

                self_grad_clone
                    .borrow()
                    .set_purgeable_state(MTLPurgeableState::Empty);
                self_grad_clone.replace(result_grad_buffer);
            }) as Box<dyn FnMut()>))
        };

        out.set_backward(backward_pass);
        out
    }
}

impl Mul<&Tensor<Matrix>> for &Tensor<Matrix> {
    type Output = Tensor<Matrix>;

    fn mul(self, rhs: &Tensor<Matrix>) -> Tensor<Matrix> {
        let capture_device = gpu_trace();

        capture_device.begin_scope();

        let lhs_size = self
            .size()
            .expect("Failed to get size of left-hand side matrix");
        let rhs_size = rhs
            .size()
            .expect("Failed to get size of right-hand side matrix");

        if lhs_size.1 != rhs_size.0 {
            panic!("Number of columns in the left-hand side matrix must match the number of rows in the right-hand side matrix for multiplication.");
        }

        let (encoder, command_buffer) =
            gpu_setup("matrix_mul_tiled", &self.device, &self.command_queue);

        let out_size =
            compute_output_matrix_size(self.size().unwrap().clone(), rhs.size().unwrap().clone());

        // Create the result buffer and initialize it with the values from the matrix
        let result_buffer_size = u32size(out_size);

        let result_buffer = self.device.new_buffer(
            result_buffer_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number.borrow()), 0);
        encoder.set_buffer(1, Some(&rhs.number.borrow()), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);

        let matrix_sizes: [u32; 3] = [lhs_size.0 as u32, lhs_size.1 as u32, rhs_size.1 as u32];

        let matrix_sizes_buffer = self.device.new_buffer_with_data(
            unsafe { std::mem::transmute(&matrix_sizes as *const u32) },
            (matrix_sizes.len() * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(3, Some(&matrix_sizes_buffer), 0);

        let threads_per_group = MTLSize {
            width: 16,  // Corresponds to TILE_SIZE in the Metal shader
            height: 16, // Corresponds to TILE_SIZE in the Metal shader
            depth: 1,
        };

        let num_threadgroups = MTLSize {
            width: ((rhs_size.1 + 15) / 16) as u64, // numColsMatrix2 divided by TILE_SIZE, rounded up
            height: ((lhs_size.0 + 15) / 16) as u64, // numRowsMatrix1 divided by TILE_SIZE, rounded up
            depth: 1,
        };

        encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Create a new Tensor to hold the result
        let mut out = Tensor::new_with_children(
            DEFAULT_DEVICE.clone(),
            result_buffer,
            Some(vec![self.clone(), rhs.clone()]),
            "*".to_string(),
            format!("({})*({})", self.label, rhs.label),
            self.command_queue.clone(),
            out_size,
        );

        matrix_sizes_buffer.set_purgeable_state(MTLPurgeableState::Empty);

        capture_device.end_scope();

        let mul_backwards_pipeline = gpu_pipeline("matrix_mul_tiled_backwards", &self.device);

        let command_queue = self.command_queue.clone();

        let lhs_number_clone = Rc::clone(&self.number);
        let rhs_number_clone = Rc::clone(&rhs.number);
        let out_grad_clone = Rc::clone(&out.grad);
        let self_grad_clone = Rc::clone(&self.grad);
        let rhs_grad_clone = Rc::clone(&rhs.grad);
        let self_grad_buffer_size = Rc::clone(&self.grad_buffer_size);
        let rhs_grad_buffer_size = Rc::clone(&rhs.grad_buffer_size);

        let device = self.device.clone();

        let lhs_size = lhs_size.clone();
        let (m, _) = lhs_size;
        let rhs_size = rhs_size.clone();
        let (n, p) = rhs_size;

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let (encoder, command_buffer) =
                gpu_encoder(mul_backwards_pipeline.clone(), &command_queue);

            // println!("backward pass for mul {}", label);

            let lhs_grad_buffer_size = (m * n * std::mem::size_of::<f32>()) as u64;

            // println!("lhs grad buffer size {:?}", (m, n));

            self_grad_buffer_size.replace((m, n));

            let lhs_grad_buffer = device.new_buffer(
                lhs_grad_buffer_size,
                metal::MTLResourceOptions::StorageModeShared,
            );

            rhs_grad_buffer_size.replace((n, p));

            // println!("lhs grad buffer size {:?}", (n, p));

            let rhs_grad_buffer_size = (n * p * std::mem::size_of::<f32>()) as u64;

            let rhs_grad_buffer = device.new_buffer(
                rhs_grad_buffer_size,
                metal::MTLResourceOptions::StorageModeShared,
            );

            encoder.set_buffer(0, Some(&lhs_number_clone.borrow()), 0);
            encoder.set_buffer(1, Some(&rhs_number_clone.borrow()), 0);
            encoder.set_buffer(2, Some(&out_grad_clone.borrow()), 0);

            encoder.set_buffer(3, Some(&lhs_grad_buffer), 0); // The result is stored back in self_grad
            encoder.set_buffer(4, Some(&rhs_grad_buffer), 0); // The result is stored back in rhs_grad
            encoder.set_buffer(5, Some(&matrix_sizes_buffer), 0);

            encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            self_grad_clone
                .borrow()
                .set_purgeable_state(MTLPurgeableState::Empty);
            rhs_grad_clone
                .borrow()
                .set_purgeable_state(MTLPurgeableState::Empty);

            matrix_sizes_buffer.set_purgeable_state(MTLPurgeableState::Empty);

            self_grad_clone.replace(lhs_grad_buffer);
            rhs_grad_clone.replace(rhs_grad_buffer);
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);
        out
    }
}

impl Sub<&Tensor<Matrix>> for &Tensor<Matrix> {
    type Output = Tensor<Matrix>;

    fn sub(self, rhs: &Tensor<Matrix>) -> Tensor<Matrix> {
        return self + &(rhs * -1.0f32);
    }
}
