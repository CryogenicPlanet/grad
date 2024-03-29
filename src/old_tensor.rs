use core::num;
use metal::*;
use std::mem::ManuallyDrop;
use std::{cell::RefCell, fs, rc::Rc, sync::Arc};
use uuid::Uuid;

pub mod utils {
    use std::collections::HashSet;

    use super::{Tensor, TensorType};

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
}

lazy_static! {
    static ref DEFAULT_DEVICE: Arc<Device> =
        Arc::new(Device::system_default().expect("Failed to create Metal device"));
    static ref DEFAULT_COMMAND_QUEUE: Arc<CommandQueue> =
        Arc::new(DEFAULT_DEVICE.new_command_queue());
}

enum SizeType {
    Vector(usize),
    Matrix(usize, usize),
}

impl From<(usize, usize)> for SizeType {
    fn from(item: (usize, usize)) -> Self {
        SizeType::Matrix(item.0, item.1)
    }
}

impl From<usize> for SizeType {
    fn from(item: usize) -> Self {
        SizeType::Vector(item)
    }
}

enum ArrayType {
    Vector(Vec<f32>),
    Matrix(Vec<Vec<f32>>),
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

pub trait TensorType {
    type Type;
    type SizeType: Into<SizeType>;
    type ArrayType: Into<ArrayType>;
}

struct Matrix;
struct Vector;

impl TensorType for Matrix {
    type SizeType = (usize, usize);
    type Type = Matrix;
    type ArrayType = Vec<Vec<f32>>;
}

impl TensorType for Vector {
    type SizeType = usize;
    type Type = Vector;
    type ArrayType = Vec<f32>;
}

#[derive(Clone)]
pub struct Tensor<T: TensorType> {
    uuid: String,
    size: T::SizeType,
    _type: T::Type,
    buffer_size: usize,

    pub number: Arc<Buffer>, // GPU buffer for numbers
    grad: Arc<Buffer>,       // GPU buffer for gradients
    label: String,
    _prev: Option<Vec<Tensor<T>>>,
    _op: String,
    _backwards: Option<Rc<RefCell<Box<dyn FnMut()>>>>, // Adjusted to use Rc for cloning and made optional

    device: Arc<Device>,              // Metal device reference
    command_queue: Arc<CommandQueue>, // Command queue for GPU commands
}

impl<T: TensorType> Tensor<T> {
    fn get_raw_val(&self, buffer: &Arc<Buffer>, size: T::SizeType) -> T::ArrayType {
        let size_enum: SizeType = size.into(); // Convert T::SizeType into SizeType enum
        match size_enum {
            SizeType::Vector(size) => {
                let length = buffer.length() as usize / std::mem::size_of::<f32>();
                let contents = buffer.contents();
                let data =
                    unsafe { std::slice::from_raw_parts(contents as *const f32, size).to_vec() };
                data // Convert Vec<f32> into T::ArrayType
            }
            SizeType::Matrix(rows, cols) => {
                let length = buffer.length() as usize / std::mem::size_of::<f32>();
                let contents = buffer.contents();
                let mut data = Vec::with_capacity(rows);
                unsafe {
                    for i in 0..rows {
                        let row = std::slice::from_raw_parts(
                            contents.offset((i * cols) as isize) as *const f32,
                            cols,
                        )
                        .to_vec();
                        data.push(row);
                    }
                }
                T::ArrayType::from(data) // Convert Vec<Vec<f32>> into T::ArrayType
            }
        }
    }

    pub fn get_children(&self) -> Option<Vec<Tensor<T>>> {
        self._prev.clone()
    }

    pub fn get_op(&self) -> String {
        self._op.clone()
    }

    pub fn get_numbers(&self) -> T::ArrayType {
        self.get_raw_val(&self.number, self.size)
    }

    pub fn get_grad(&self) -> T::ArrayType {
        self.get_raw_val(&self.grad, self.size)
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

        for node in sorted.iter().rev() {
            node.maybe_backwards();
        }
    }

    pub fn set_grad(&self, grad: f32) {
        let grad_vec = vec![grad; self.buffer_size];

        let grad_data = grad_vec;
        let grad_ptr = grad_data.as_ptr() as *const std::ffi::c_void;
        let grad_len = (grad_data.len() * std::mem::size_of::<f32>()) as u64;
        unsafe {
            std::ptr::copy(
                grad_ptr,
                self.grad.contents() as *mut std::ffi::c_void,
                grad_len as usize,
            );
        }
    }
}

impl Tensor<Vector> {
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
            size: size,
            buffer_size: size,
            number: Arc::new(number_buffer),
            grad: Arc::new(grad_buffer),
            device,
            command_queue: DEFAULT_COMMAND_QUEUE.clone(),
            _prev: children,
            _op: op,
            label,
            _backwards: None,
            _type: Vector,
        }
    }

    fn new_vector_with_children(
        device: Arc<Device>,
        result_buffer: Arc<Buffer>,
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
            size,
            buffer_size: size,
            number: result_buffer,
            grad: Arc::new(grad_buffer),
            label,
            _prev: children,
            _op: op,
            _backwards: None,
            device,
            command_queue,
            _type: Vector,
        }
    }
}

impl Tensor<Matrix> {
    fn new_f32(
        device: Arc<Device>,
        numbers: Vec<Vec<f32>>,
        children: Option<Vec<Tensor<Matrix>>>,
        op: String,
        label: String,
    ) -> Self {
        // if !Self::check_valid_matrix(&numbers) {
        //     panic!("Invalid matrix");
        // }

        let size = (numbers.len(), numbers[0].len());

        let buffer_size = size.0 as usize * size.1 as usize * std::mem::size_of::<f32>();

        let numbers_flat: Vec<f32> = numbers.into_iter().flatten().collect();

        let number_buffer = device.new_buffer_with_data(
            unsafe { std::mem::transmute(numbers_flat.as_ptr()) },
            (numbers_flat.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let grad_buffer = device.new_buffer_with_data(
            unsafe {
                std::mem::transmute(vec![0f32; buffer_size / std::mem::size_of::<f32>()].as_ptr())
            },
            buffer_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Tensor {
            uuid: Uuid::new_v4().to_string(),
            size: size,
            buffer_size,
            number: Arc::new(number_buffer),
            grad: Arc::new(grad_buffer),
            device,
            command_queue: DEFAULT_COMMAND_QUEUE.clone(),
            _prev: children,
            _op: op,
            label,
            _backwards: None,
            _type: Matrix,
        }
    }

    fn new_with_children(
        device: Arc<Device>,
        result_buffer: Arc<Buffer>,
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
            size,
            number: result_buffer,
            grad: Arc::new(grad_buffer),
            label,
            buffer_size: grad_buffer_size,
            _prev: children,
            _op: op,
            _backwards: None,
            device,
            command_queue,
            _type: Matrix,
        }
    }
}

use std::ops::{Add, Mul};

impl Add<&Tensor<Vector>> for &Tensor<Vector> {
    type Output = Tensor<Vector>;

    fn add(self, rhs: &Tensor<Vector>) -> Tensor<Vector> {
        let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
            .expect("Failed to read Metal shader file");

        let add_function = self
            .device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function("vector_add", None)
            .expect("Failed to get 'vector_add' function from library");

        let add_pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&add_function)
            .expect("Failed to create compute pipeline state");

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&add_pipeline_state);

        // Create the result buffer before using it
        let result_buffer = self.device.new_buffer(
            self.number.length() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number), 0);
        encoder.set_buffer(1, Some(&rhs.number), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0); // Store the result in the result buffer

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: (self.number.length() / std::mem::size_of::<f32>() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Create a new Tensor to hold the result

        let mut out = Tensor::new_vector_with_children(
            DEFAULT_DEVICE.clone(),
            Arc::new(result_buffer),
            Some(vec![self.clone(), rhs.clone()]),
            "+".to_string(),
            format!("{}+{}", self.label, rhs.label),
            self.command_queue.clone(),
            self.size,
        );

        let add_backward_pass = self
            .device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function("vector_add_backwards", None)
            .expect("Failed to get 'vector_add' function from library");

        let device = self.device.clone();
        let command_queue = self.command_queue.clone();

        let out_grad_clone = out.grad.clone();
        let self_grad_clone = self.grad.clone();
        let rhs_grad_clone = rhs.grad.clone();

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let add_pipeline_state_self = device
                .new_compute_pipeline_state_with_function(&add_backward_pass)
                .expect("Failed to create compute pipeline state for self");

            let command_buffer_self = command_queue.new_command_buffer();
            let encoder_self = command_buffer_self.new_compute_command_encoder();
            encoder_self.set_compute_pipeline_state(&add_pipeline_state_self);

            encoder_self.set_buffer(0, Some(&self_grad_clone), 0);
            encoder_self.set_buffer(1, Some(&out_grad_clone), 0);
            encoder_self.set_buffer(2, Some(&self_grad_clone), 0); // The result is stored back in self_grad

            encoder_self.dispatch_thread_groups(num_threadgroups, threads_per_group);

            encoder_self.end_encoding();
            command_buffer_self.commit();
            command_buffer_self.wait_until_completed();

            let add_pipeline_state_rhs = device
                .new_compute_pipeline_state_with_function(&add_backward_pass)
                .expect("Failed to create compute pipeline state for rhs");

            let command_buffer_rhs = command_queue.new_command_buffer();
            let encoder_rhs = command_buffer_rhs.new_compute_command_encoder();
            encoder_rhs.set_compute_pipeline_state(&add_pipeline_state_rhs);

            encoder_rhs.set_buffer(0, Some(&rhs_grad_clone), 0);
            encoder_rhs.set_buffer(1, Some(&out_grad_clone), 0);
            encoder_rhs.set_buffer(2, Some(&rhs_grad_clone), 0); // The result is stored back in rhs_grad

            encoder_rhs.dispatch_thread_groups(num_threadgroups, threads_per_group);

            encoder_rhs.end_encoding();
            command_buffer_rhs.commit();
            command_buffer_rhs.wait_until_completed();
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);

        out
    }
}

impl Mul<&Tensor<Vector>> for &Tensor<Vector> {
    type Output = Tensor<Vector>;

    fn mul(self, rhs: &Tensor<Vector>) -> Tensor<Vector> {
        let shader_source = fs::read_to_string("src/kernels/tensor_ops.metal")
            .expect("Failed to read Metal shader file");

        let mul_function = self
            .device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function("vector_mul", None)
            .expect("Failed to get 'vector_mul' function from library");
        let mul_pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&mul_function)
            .expect("Failed to create compute pipeline state");

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&mul_pipeline_state);

        // Create the result buffer before using it
        let result_buffer = self.device.new_buffer(
            self.number.length() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(0, Some(&self.number), 0);
        encoder.set_buffer(1, Some(&rhs.number), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0); // Store the result in the result buffer

        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: (self.number.length() / std::mem::size_of::<f32>() as u64 + 63) / 64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(num_threadgroups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Create a new Tensor to hold the result

        let mut out = Tensor::new_vector_with_children(
            DEFAULT_DEVICE.clone(),
            Arc::new(result_buffer),
            Some(vec![self.clone(), rhs.clone()]),
            "*".to_string(),
            format!("{}*{}", self.label, rhs.label),
            self.command_queue.clone(),
            self.size,
        );

        let mul_backward_pass = self
            .device
            .new_library_with_source(shader_source.as_str(), &metal::CompileOptions::new())
            .expect("Failed to create library from source")
            .get_function("vector_mul_backwards", None)
            .expect("Failed to get 'vector_mul' function from library");

        let device = self.device.clone();
        let command_queue = self.command_queue.clone();

        let out_grad_clone = out.grad.clone();
        let self_grad_clone = self.grad.clone();
        let rhs_grad_clone = rhs.grad.clone();

        let self_number_clone = self.number.clone();
        let rhs_number_clone = rhs.number.clone();

        let backward_pass = Rc::new(RefCell::new(Box::new(move || {
            let mul_pipeline_state_self = device
                .new_compute_pipeline_state_with_function(&mul_backward_pass)
                .expect("Failed to create compute pipeline state for self");

            let command_buffer_self = command_queue.new_command_buffer();
            let encoder_self = command_buffer_self.new_compute_command_encoder();
            encoder_self.set_compute_pipeline_state(&mul_pipeline_state_self);

            encoder_self.set_buffer(0, Some(&self_grad_clone), 0);
            encoder_self.set_buffer(1, Some(&rhs_number_clone), 0);
            encoder_self.set_buffer(2, Some(&out_grad_clone), 0);
            encoder_self.set_buffer(3, Some(&self_grad_clone), 0); // The result is stored back in self_grad

            encoder_self.dispatch_thread_groups(num_threadgroups, threads_per_group);

            encoder_self.end_encoding();
            command_buffer_self.commit();
            command_buffer_self.wait_until_completed();

            let mul_pipeline_state_rhs = device
                .new_compute_pipeline_state_with_function(&mul_backward_pass)
                .expect("Failed to create compute pipeline state for rhs");
            let command_buffer_rhs = command_queue.new_command_buffer();
            let encoder_rhs = command_buffer_rhs.new_compute_command_encoder();
            encoder_rhs.set_compute_pipeline_state(&mul_pipeline_state_rhs);

            encoder_rhs.set_buffer(0, Some(&rhs_grad_clone), 0);
            encoder_rhs.set_buffer(1, Some(&self_number_clone), 0);
            encoder_rhs.set_buffer(2, Some(&out_grad_clone), 0);
            encoder_rhs.set_buffer(3, Some(&rhs_grad_clone), 0); // The result is stored back in rhs_grad

            encoder_rhs.dispatch_thread_groups(num_threadgroups, threads_per_group);

            encoder_rhs.end_encoding();
            command_buffer_rhs.commit();
            command_buffer_rhs.wait_until_completed();
        }) as Box<dyn FnMut()>));

        out.set_backward(backward_pass);

        out
    }
}

impl<T: TensorType> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        // Define your tolerance for floating-point comparison
        let tolerance = 1e-6;

        let self_number = self.get_numbers();

        // self.get_number()
        //     .iter()
        //     .zip(other.get_number().iter())
        //     .all(|(a, b)| (a - b).abs() < tolerance)
        //     && self._op == other._op
        // Assuming _prev comparison is not needed or should be handled differently
    }
}

impl<T: TensorType> Eq for Tensor<T> {}

use std::hash::{Hash, Hasher};

impl<T: TensorType> Hash for Tensor<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash each discretized number and the operation
        let numbers = self.get_numbers();

        match numbers {
            ArrayType::Matrix(nums) => {
                for &num in nums.iter() {
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
            }
            _ => panic!("Invalid tensor type"),
        }

        // Handle _prev hashing if needed, possibly by hashing the length or similar
    }
}
