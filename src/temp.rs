// enum Test {
//     A(A),
//     B(B),
// }

// impl Test {
//     fn common(&self) -> i32 {
//         match self {
//             Test::A(a) => a.common,
//             Test::B(b) => b.common,
//         }
//     }
// }

// struct A {
//     common: i32,
//     a: i32,
// }

// struct B {
//     common: i32,
//     b: i32,
// }

// fn test(test: Test) {
//     // This doesn't work why?
//     // like sorta what is the point of the enum?
//     let x = test.common;
// }
