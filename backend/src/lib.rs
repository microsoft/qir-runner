// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![deny(clippy::all, clippy::pedantic)]

//! # QIR compliant backend for quantum simulation.
//! This libary builds on top of the `qir_stdlib` to implement a full backend for simulation of QIR
//! programs. This includes a broad set of quantum intrinsic operations for sparse state simulation,
//! based on the design from
//! <a href="https://arxiv.org/abs/2105.01533">Leveraging state sparsity for more efficient quantum simulations</a>.

pub mod result_bool;

pub mod exp;

use bitvec::prelude::*;
use num_bigint::BigUint;
use num_complex::Complex64;
use quantum_sparse_sim::QuantumSim;
use std::cell::RefCell;
use std::convert::TryInto;
use std::ffi::c_char;
use std::ffi::c_double;
use std::ffi::{c_void, CString};
use std::io::Write;
use std::mem::size_of;

use result_bool::{
    __quantum__rt__result_equal, __quantum__rt__result_get_one, __quantum__rt__result_get_zero,
};

pub use qir_stdlib::{
    arrays::*, bigints::*, callables::*, math::*, output_recording::*, range_support::*,
    strings::*, tuples::*, *,
};

struct SimulatorState {
    pub sim: QuantumSim,
    pub res: BitVec,
    pub max_qubit_id: usize,
}

thread_local! {
    static SIM_STATE: RefCell<SimulatorState> = RefCell::new(SimulatorState {
        sim: QuantumSim::default(),
        res: bitvec![],
        max_qubit_id: 0
    });
}

/// Sets the seed for the pseudo-random number generator used during measurements.
pub fn set_rng_seed(seed: u64) {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        state.sim.set_rng_seed(seed);
    });
}

/// Initializes the execution environment.
#[no_mangle]
pub extern "C" fn __quantum__rt__initialize(_: *mut c_char) {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        // in order to continue using the same RNG, we need to reset the simulator
        // and keep the same RNG
        state.sim = QuantumSim::new(Some(state.sim.take_rng()));
        state.res = bitvec![];
        state.max_qubit_id = 0;
    });
}

fn ensure_sufficient_qubits(sim: &mut QuantumSim, qubit_id: usize, max: &mut usize) {
    while qubit_id + 1 > *max {
        let _ = sim.allocate();
        *max += 1;
    }
}

/// Maps the given qubits from the given Pauli basis into the computational basis, returning the
/// unwrapped `QirArray`s into a vector of matching Pauli and qubit id tuples.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn map_to_z_basis(
    state: &mut SimulatorState,
    paulis: *const QirArray,
    qubits: *const QirArray,
) -> Vec<(Pauli, usize)> {
    let paulis_size = __quantum__rt__array_get_size_1d(paulis);
    let qubits_size = __quantum__rt__array_get_size_1d(qubits);
    if paulis_size != qubits_size {
        __quantum__rt__fail(__quantum__rt__string_create(
            CString::new("Pauli array and Qubit array must be the same size.")
                .expect("Unable to allocate memory for failure message string.")
                .as_bytes_with_nul()
                .as_ptr() as *mut c_char,
        ));
    }

    let combined_list: Vec<(Pauli, usize)> = (0..paulis_size)
        .filter_map(|index| {
            let p =
                *__quantum__rt__array_get_element_ptr_1d(paulis, index).cast::<Pauli>() as Pauli;
            let q = *__quantum__rt__array_get_element_ptr_1d(qubits, index).cast::<*mut c_void>()
                as usize;
            if let Pauli::I = p {
                None
            } else {
                ensure_sufficient_qubits(&mut state.sim, q, &mut state.max_qubit_id);
                Some((p, q))
            }
        })
        .collect();

    for (pauli, qubit) in &combined_list {
        match pauli {
            Pauli::X => state.sim.h(*qubit),
            Pauli::Y => {
                state.sim.h(*qubit);
                state.sim.s(*qubit);
                state.sim.h(*qubit);
            }
            _ => (),
        }
    }

    combined_list
}

/// Given a vector of Pauli and qubit id pairs, unmaps from the computational basis back into the given
/// Pauli basis. This should be the adjoint of the `map_to_z_basis` operation.
fn unmap_from_z_basis(state: &mut SimulatorState, combined_list: Vec<(Pauli, usize)>) {
    for (pauli, qubit) in combined_list {
        match pauli {
            Pauli::X => state.sim.h(qubit),
            Pauli::Y => {
                state.sim.h(qubit);
                state.sim.sadj(qubit);
                state.sim.h(qubit);
            }
            _ => (),
        }
    }
}

macro_rules! single_qubit_gate {
    ($(#[$meta:meta])*
    $qir_name:ident, $gate:expr) => {
        $(#[$meta])*
        #[no_mangle]
        pub extern "C" fn $qir_name(qubit: *mut c_void) {
            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);

                $gate(&mut state.sim, qubit as usize);
            });
        }
    };
}

single_qubit_gate!(
    /// QIR API for performing the H gate on the given qubit.
    __quantum__qis__h__body,
    QuantumSim::h
);
single_qubit_gate!(
    /// QIR API for performing the S gate on the given qubit.
    __quantum__qis__s__body,
    QuantumSim::s
);
single_qubit_gate!(
    /// QIR API for performing the Adjoint S gate on the given qubit.
    __quantum__qis__s__adj,
    QuantumSim::sadj
);
single_qubit_gate!(
    /// QIR API for performing the T gate on the given qubit.
    __quantum__qis__t__body,
    QuantumSim::t
);
single_qubit_gate!(
    /// QIR API for performing the Adjoint T gate on the given qubit.
    __quantum__qis__t__adj,
    QuantumSim::tadj
);
single_qubit_gate!(
    /// QIR API for performing the X gate on the given qubit.
    __quantum__qis__x__body,
    QuantumSim::x
);
single_qubit_gate!(
    /// QIR API for performing the Y gate on the given qubit.
    __quantum__qis__y__body,
    QuantumSim::y
);
single_qubit_gate!(
    /// QIR API for performing the Z gate on the given qubit.
    __quantum__qis__z__body,
    QuantumSim::z
);

macro_rules! controlled_qubit_gate {
    ($(#[$meta:meta])*
    $qir_name:ident, $gate:expr, 1) => {
        $(#[$meta])*
        #[no_mangle]
        pub extern "C" fn $qir_name(control: *mut c_void, target: *mut c_void) {
            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                ensure_sufficient_qubits(&mut state.sim, target as usize, &mut state.max_qubit_id);
                ensure_sufficient_qubits(&mut state.sim, control as usize, &mut state.max_qubit_id);

                $gate(&mut state.sim, &[control as usize], target as usize);
            });
        }
    };

    ($(#[$meta:meta])*
    $qir_name:ident, $gate:expr, 2) => {
        $(#[$meta])*
        #[no_mangle]
        pub extern "C" fn $qir_name(
            control_1: *mut c_void,
            control_2: *mut c_void,
            target: *mut c_void,
        ) {
            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                ensure_sufficient_qubits(&mut state.sim, target as usize, &mut state.max_qubit_id);
                ensure_sufficient_qubits(&mut state.sim, control_1 as usize, &mut state.max_qubit_id);
                ensure_sufficient_qubits(&mut state.sim, control_2 as usize, &mut state.max_qubit_id);

                $gate(&mut state.sim, &[control_1 as usize, control_2 as usize], target as usize);
            });
        }
    };
}

controlled_qubit_gate!(
    /// QIR API for performing the CNOT gate with the given qubits.
    __quantum__qis__cnot__body,
    QuantumSim::mcx,
    1
);
controlled_qubit_gate!(
    /// QIR API for performing the CNOT gate with the given qubits.
    __quantum__qis__cx__body,
    QuantumSim::mcx,
    1
);
controlled_qubit_gate!(
    /// QIR API for performing the CCNOT gate with the given qubits.
    __quantum__qis__ccx__body,
    QuantumSim::mcx,
    2
);
controlled_qubit_gate!(
    /// QIR API for performing the CY gate with the given qubits.
    __quantum__qis__cy__body,
    QuantumSim::mcy,
    1
);
controlled_qubit_gate!(
    /// QIR API for performing the CZ gate with the given qubits.
    __quantum__qis__cz__body,
    QuantumSim::mcz,
    1
);

macro_rules! single_qubit_rotation {
    ($(#[$meta:meta])*
    $qir_name:ident, $gate:expr) => {
        $(#[$meta])*
        #[no_mangle]
        pub extern "C" fn $qir_name(theta: c_double, qubit: *mut c_void) {
            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);

                $gate(&mut state.sim, theta, qubit as usize);
            });
        }
    };
}

single_qubit_rotation!(
    /// QIR API for applying a Pauli-X rotation with the given angle and qubit.
    __quantum__qis__rx__body,
    QuantumSim::rx
);
single_qubit_rotation!(
    /// QIR API for applying a Pauli-Y rotation with the given angle and qubit.
    __quantum__qis__ry__body,
    QuantumSim::ry
);
single_qubit_rotation!(
    /// QIR API for applying a Pauli-Z rotation with the given angle and qubit.
    __quantum__qis__rz__body,
    QuantumSim::rz
);

macro_rules! multicontrolled_qubit_gate {
    ($(#[$meta:meta])*
    $qir_name:ident, $gate:expr) => {
        $(#[$meta])*
        /// # Safety
        ///
        /// This function should only be called with arrays and tuples created by the QIR runtime library.
        #[no_mangle]
        #[allow(clippy::cast_ptr_alignment)]
        pub unsafe extern "C" fn $qir_name(ctls: *const QirArray, qubit: *mut c_void) {
            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);
                let ctls_size = __quantum__rt__array_get_size_1d(ctls);
                let ctls_list: Vec<usize> = (0..ctls_size)
                    .map(|index| {
                        let q = *__quantum__rt__array_get_element_ptr_1d(ctls, index)
                            .cast::<*mut c_void>() as usize;
                        ensure_sufficient_qubits(&mut state.sim, q, &mut state.max_qubit_id);
                        q
                    })
                    .collect();

                $gate(&mut state.sim, &ctls_list, qubit as usize);
            });
        }
    };
}

multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled H gate with the given qubits.
    __quantum__qis__h__ctl,
    QuantumSim::mch
);
multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled S gate with the given qubits.
    __quantum__qis__s__ctl,
    QuantumSim::mcs
);
multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled Adjoint S gate with the given qubits.
    __quantum__qis__s__ctladj,
    QuantumSim::mcsadj
);
multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled T gate with the given qubits.
    __quantum__qis__t__ctl,
    QuantumSim::mct
);
multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled Adjoint T gate with the given qubits.
    __quantum__qis__t__ctladj,
    QuantumSim::mctadj
);
multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled X gate with the given qubits.
    __quantum__qis__x__ctl,
    QuantumSim::mcx
);
multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled Y gate with the given qubits.
    __quantum__qis__y__ctl,
    QuantumSim::mcy
);
multicontrolled_qubit_gate!(
    /// QIR API for performing the multicontrolled Z gate with the given qubits.
    __quantum__qis__z__ctl,
    QuantumSim::mcz
);

#[derive(Copy, Clone)]
#[repr(C)]
struct RotationArgs {
    theta: c_double,
    qubit: *mut c_void,
}

macro_rules! multicontrolled_qubit_rotation {
    ($(#[$meta:meta])*
    $qir_name:ident, $gate:expr) => {
        $(#[$meta])*
        /// # Safety
        ///
        /// This function should only be called with arrays and tuples created by the QIR runtime library.
        #[no_mangle]
        #[allow(clippy::cast_ptr_alignment)]
        pub unsafe extern "C" fn $qir_name(
            ctls: *const QirArray,
            arg_tuple: *mut *const Vec<u8>,
        ) {
            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();

                let args = *arg_tuple.cast::<RotationArgs>();

                ensure_sufficient_qubits(&mut state.sim, args.qubit as usize, &mut state.max_qubit_id);
                let ctls_size = __quantum__rt__array_get_size_1d(ctls);
                let ctls_list: Vec<usize> = (0..ctls_size)
                    .map(|index| {
                        let q = *__quantum__rt__array_get_element_ptr_1d(ctls, index)
                            .cast::<*mut c_void>() as usize;
                        ensure_sufficient_qubits(&mut state.sim, q, &mut state.max_qubit_id);
                        q
                    })
                    .collect();

                $gate(
                    &mut state.sim,
                    &ctls_list,
                    args.theta,
                    args.qubit as usize,
                );
            });
        }
    };
}

multicontrolled_qubit_rotation!(
    /// QIR API for applying a multicontrolled Pauli-X rotation with the given angle and qubit.
    __quantum__qis__rx__ctl,
    QuantumSim::mcrx
);
multicontrolled_qubit_rotation!(
    /// QIR API for applying a multicontrolled Pauli-Y rotation with the given angle and qubit.
    __quantum__qis__ry__ctl,
    QuantumSim::mcry
);
multicontrolled_qubit_rotation!(
    /// QIR API for applying a multicontrolled Pauli-Z rotation with the given angle and qubit.
    __quantum__qis__rz__ctl,
    QuantumSim::mcrz
);

/// QIR API for applying a joint rotation Pauli-Y rotation with the given angle for the two target qubit.
#[no_mangle]
pub extern "C" fn __quantum__qis__rxx__body(
    theta: c_double,
    qubit1: *mut c_void,
    qubit2: *mut c_void,
) {
    __quantum__qis__h__body(qubit1);

    __quantum__qis__h__body(qubit2);

    __quantum__qis__rzz__body(theta, qubit1, qubit2);

    __quantum__qis__h__body(qubit2);

    __quantum__qis__h__body(qubit1);
}

/// QIR API for applying a joint rotation Pauli-Y rotation with the given angle for the two target qubit.
#[no_mangle]
pub extern "C" fn __quantum__qis__ryy__body(
    theta: c_double,
    qubit1: *mut c_void,
    qubit2: *mut c_void,
) {
    __quantum__qis__h__body(qubit1);
    __quantum__qis__s__body(qubit1);
    __quantum__qis__h__body(qubit1);

    __quantum__qis__h__body(qubit2);
    __quantum__qis__s__body(qubit2);
    __quantum__qis__h__body(qubit2);

    __quantum__qis__rzz__body(theta, qubit1, qubit2);

    __quantum__qis__h__body(qubit2);
    __quantum__qis__s__adj(qubit2);
    __quantum__qis__h__body(qubit2);

    __quantum__qis__h__body(qubit1);
    __quantum__qis__s__adj(qubit1);
    __quantum__qis__h__body(qubit1);
}

/// QIR API for applying a joint rotation Pauli-Z rotation with the given angle for the two target qubit.
#[no_mangle]
pub extern "C" fn __quantum__qis__rzz__body(
    theta: c_double,
    qubit1: *mut c_void,
    qubit2: *mut c_void,
) {
    __quantum__qis__cx__body(qubit2, qubit1);
    __quantum__qis__rz__body(theta, qubit1);
    __quantum__qis__cx__body(qubit2, qubit1);
}

/// QIR API for applying a rotation about the given Pauli axis with the given angle and qubit.
#[no_mangle]
pub extern "C" fn __quantum__qis__r__body(pauli: Pauli, theta: c_double, qubit: *mut c_void) {
    match pauli {
        Pauli::I => (),
        Pauli::X => __quantum__qis__rx__body(theta, qubit),
        Pauli::Y => __quantum__qis__ry__body(theta, qubit),
        Pauli::Z => __quantum__qis__rz__body(theta, qubit),
    }
}

/// QIR API for applying an adjoint rotation about the given Pauli axis with the given angle and qubit.
#[no_mangle]
pub extern "C" fn __quantum__qis__r__adj(pauli: Pauli, theta: c_double, qubit: *mut c_void) {
    __quantum__qis__r__body(pauli, -theta, qubit);
}

#[derive(Copy, Clone)]
#[repr(C)]
struct PauliRotationArgs {
    pauli: Pauli,
    theta: c_double,
    qubit: *mut c_void,
}

/// QIR API for applying a controlled rotation about the given Pauli axis with the given angle and qubit.
/// # Safety
///
/// This function should only be called with arrays and tuples created by the QIR runtime library.
#[allow(clippy::cast_ptr_alignment)]
#[no_mangle]
pub unsafe extern "C" fn __quantum__qis__r__ctl(
    ctls: *const QirArray,
    arg_tuple: *mut *const Vec<u8>,
) {
    let args = *arg_tuple.cast::<PauliRotationArgs>();
    let rot_args = RotationArgs {
        theta: args.theta,
        qubit: args.qubit,
    };
    let rot_arg_tuple = __quantum__rt__tuple_create(size_of::<RotationArgs>() as u64);
    *rot_arg_tuple.cast::<RotationArgs>() = rot_args;

    match args.pauli {
        Pauli::X => __quantum__qis__rx__ctl(ctls, rot_arg_tuple),
        Pauli::Y => __quantum__qis__ry__ctl(ctls, rot_arg_tuple),
        Pauli::Z => __quantum__qis__rz__ctl(ctls, rot_arg_tuple),
        Pauli::I => {
            if __quantum__rt__array_get_size_1d(ctls) > 0 {
                SIM_STATE.with(|sim_state| {
                    let state = &mut *sim_state.borrow_mut();

                    ensure_sufficient_qubits(
                        &mut state.sim,
                        args.qubit as usize,
                        &mut state.max_qubit_id,
                    );
                    let ctls_size = __quantum__rt__array_get_size_1d(ctls);
                    let ctls_list: Vec<usize> = (0..ctls_size)
                        .map(|index| {
                            let q = *__quantum__rt__array_get_element_ptr_1d(ctls, index)
                                .cast::<*mut c_void>() as usize;
                            ensure_sufficient_qubits(&mut state.sim, q, &mut state.max_qubit_id);
                            q
                        })
                        .collect();

                    if let Some((head, rest)) = ctls_list.split_first() {
                        state.sim.mcphase(
                            rest,
                            Complex64::exp(Complex64::new(0.0, -args.theta / 2.0)),
                            *head,
                        );
                    }
                });
            }
        }
    };

    __quantum__rt__tuple_update_reference_count(rot_arg_tuple, -1);
}

/// QIR API for applying an adjoint controlled rotation about the given Pauli axis with the given angle and qubit.
/// # Safety
///
/// This function should only be called with arrays and tuples created by the QIR runtime library.
#[no_mangle]
pub unsafe extern "C" fn __quantum__qis__r__ctladj(
    ctls: *const QirArray,
    arg_tuple: *mut *const Vec<u8>,
) {
    let args = *arg_tuple.cast::<PauliRotationArgs>();
    let new_args = PauliRotationArgs {
        pauli: args.pauli,
        theta: -args.theta,
        qubit: args.qubit,
    };
    let new_arg_tuple = __quantum__rt__tuple_create(size_of::<PauliRotationArgs>() as u64);
    *new_arg_tuple.cast::<PauliRotationArgs>() = new_args;
    __quantum__qis__r__ctl(ctls, new_arg_tuple);
    __quantum__rt__tuple_update_reference_count(new_arg_tuple, -1);
}

/// QIR API for applying a SWAP gate to the given qubits.
#[no_mangle]
pub extern "C" fn __quantum__qis__swap__body(qubit1: *mut c_void, qubit2: *mut c_void) {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        ensure_sufficient_qubits(&mut state.sim, qubit1 as usize, &mut state.max_qubit_id);
        ensure_sufficient_qubits(&mut state.sim, qubit2 as usize, &mut state.max_qubit_id);

        state.sim.swap_qubit_ids(qubit1 as usize, qubit2 as usize);
    });
}

/// QIR API for resetting the given qubit in the computational basis.
#[no_mangle]
pub extern "C" fn __quantum__qis__reset__body(qubit: *mut c_void) {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);

        if state.sim.measure(qubit as usize) {
            state.sim.x(qubit as usize);
        }
    });
}

/// QIR API for measuring the given qubit and storing the measured value with the given result identifier,
/// then resetting it in the computational basis.
#[allow(clippy::missing_panics_doc)]
// reason="Panics can only occur if the result that was just collected is not found in the BitVec, which should not happen."
#[no_mangle]
pub extern "C" fn __quantum__qis__mresetz__body(qubit: *mut c_void, result: *mut c_void) {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        let res_id = result as usize;
        ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);

        if state.res.len() < res_id + 1 {
            state.res.resize(res_id + 1, false);
        }

        let res = state.sim.measure(qubit as usize);

        if res {
            state.sim.x(qubit as usize);
        }

        *state
            .res
            .get_mut(res_id)
            .expect("Result with given id missing after expansion.") = res;
    });
}

/// QIR API for measuring the given qubit in the computation basis and storing the measured value with the given result identifier.
#[allow(clippy::missing_panics_doc)]
// reason="Panics can only occur if the result index is not found in the BitVec after resizing, which should not happen."
#[no_mangle]
pub extern "C" fn __quantum__qis__mz__body(qubit: *mut c_void, result: *mut c_void) {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        let res_id = result as usize;
        ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);

        if state.res.len() < res_id + 1 {
            state.res.resize(res_id + 1, false);
        }

        *state
            .res
            .get_mut(res_id)
            .expect("Result with given id missing after expansion.") =
            state.sim.measure(qubit as usize);
    });
}

/// QIR API that reads the Boolean value corresponding to the given result identifier, where true
/// indicates a |1⟩ state and false indicates a |0⟩ state.
#[allow(clippy::missing_panics_doc)]
// reason="Panics can only occur if the result index is not found in the BitVec after resizing, which should not happen."
#[no_mangle]
pub extern "C" fn __quantum__qis__read_result__body(result: *mut c_void) -> bool {
    SIM_STATE.with(|sim_state| {
        let res = &mut sim_state.borrow_mut().res;
        let res_id = result as usize;
        if res.len() < res_id + 1 {
            res.resize(res_id + 1, false);
        }

        let b = *res
            .get(res_id)
            .expect("Result with given id missing after expansion.");
        b
    })
}

/// QIR API that measures a given qubit in the computational basis, returning a runtime managed result value.
#[no_mangle]
pub extern "C" fn __quantum__qis__m__body(qubit: *mut c_void) -> *mut c_void {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);

        if state.sim.measure(qubit as usize) {
            __quantum__rt__result_get_one()
        } else {
            __quantum__rt__result_get_zero()
        }
    })
}

/// QIR API that performs joint measurement of the given qubits in the corresponding Pauli bases, returning the parity as a runtime managed result value.
/// # Safety
///
/// This function should only be called with arrays created by the QIR runtime library.
/// # Panics
///
/// This function will panic if the provided paulis and qubits arrays are not of the same size.
#[allow(clippy::cast_ptr_alignment)]
#[no_mangle]
pub unsafe extern "C" fn __quantum__qis__measure__body(
    paulis: *const QirArray,
    qubits: *const QirArray,
) -> *mut c_void {
    SIM_STATE.with(|sim_state| {
        let mut state = sim_state.borrow_mut();

        let combined_list = map_to_z_basis(&mut state, paulis, qubits);

        let res = state.sim.joint_measure(
            &combined_list
                .iter()
                .map(|(_, q)| *q)
                .collect::<Vec<usize>>(),
        );

        unmap_from_z_basis(&mut state, combined_list);

        if res {
            __quantum__rt__result_get_one()
        } else {
            __quantum__rt__result_get_zero()
        }
    })
}

/// Rust API for checking internal simulator state and returning true only if the given qubit is in exactly the |0⟩ state.
pub fn qubit_is_zero(qubit: *mut c_void) -> bool {
    SIM_STATE.with(|sim_state| {
        let state = &mut *sim_state.borrow_mut();
        ensure_sufficient_qubits(&mut state.sim, qubit as usize, &mut state.max_qubit_id);

        state.sim.qubit_is_zero(qubit as usize)
    })
}

/// QIR API for checking internal simulator state and verifying the probability of the given parity measurement result
/// for the given qubits in the given Pauli bases is equal to the expected probability, within the given tolerance.
/// # Safety
///
/// This function should only be called with arrays created by the QIR runtime library.
#[no_mangle]
pub unsafe extern "C" fn __quantum__qis__assertmeasurementprobability__body(
    paulis: *const QirArray,
    qubits: *const QirArray,
    result: *mut c_void,
    prob: c_double,
    msg: *const CString,
    tol: c_double,
) {
    SIM_STATE.with(|sim_state| {
        let mut state = sim_state.borrow_mut();

        let combined_list = map_to_z_basis(&mut state, paulis, qubits);

        let mut actual_prob = state.sim.joint_probability(
            &combined_list
                .iter()
                .map(|(_, q)| *q)
                .collect::<Vec<usize>>(),
        );

        if __quantum__rt__result_equal(result, __quantum__rt__result_get_zero()) {
            actual_prob = 1.0 - actual_prob;
        }

        if (actual_prob - prob).abs() > tol {
            __quantum__rt__fail(msg);
        }

        unmap_from_z_basis(&mut state, combined_list);
    });
}

#[derive(Copy, Clone)]
#[repr(C)]
struct AssertMeasurementProbabilityArgs {
    paulis: *const QirArray,
    qubits: *const QirArray,
    result: *mut c_void,
    prob: c_double,
    msg: *const CString,
    tol: c_double,
}

/// QIR API for checking internal simulator state and verifying the probability of the given parity measurement result
/// for the given qubits in the given Pauli bases is equal to the expected probability, within the given tolerance.
/// Note that control qubits are ignored.
/// # Safety
///
/// This function should only be called with arrays created by the QIR runtime library.
#[no_mangle]
pub unsafe extern "C" fn __quantum__qis__assertmeasurementprobability__ctl(
    _ctls: *const QirArray,
    arg_tuple: *mut *const Vec<u8>,
) {
    let args = *arg_tuple.cast::<AssertMeasurementProbabilityArgs>();
    __quantum__qis__assertmeasurementprobability__body(
        args.paulis,
        args.qubits,
        args.result,
        args.prob,
        args.msg,
        args.tol,
    );
}

pub mod legacy_output {
    use std::ffi::c_void;

    use qir_stdlib::output_recording::record_output_str;

    use crate::{
        result_bool::{__quantum__rt__result_equal, __quantum__rt__result_get_one},
        SIM_STATE,
    };

    #[allow(clippy::missing_panics_doc)]
    // reason="Panics can only occur if the result index is not found in the BitVec after resizing, which should not happen."
    #[allow(non_snake_case)]
    pub extern "C" fn __quantum__rt__result_record_output(result: *mut c_void) {
        SIM_STATE.with(|sim_state| {
            let res = &mut sim_state.borrow_mut().res;
            let res_id = result as usize;
            let b = if res.is_empty() {
                // No static measurements have been used, so default to dynamic handling.
                __quantum__rt__result_equal(result, __quantum__rt__result_get_one())
            } else {
                if res.len() < res_id + 1 {
                    res.resize(res_id + 1, false);
                }
                *res.get(res_id)
                    .expect("Result with given id missing after expansion.")
            };

            record_output_str(&format!("RESULT\t{}", if b { "1" } else { "0" }))
                .expect("Failed to write result output");
        });
    }
}

/// QIR API for recording the given result into the program output.
#[allow(clippy::missing_panics_doc)]
// reason="Panics can only occur if the result index is not found in the BitVec after resizing, which should not happen."
/// # Safety
/// This function will panic if the tag cannot be written to the output buffer.
#[no_mangle]
pub unsafe extern "C" fn __quantum__rt__result_record_output(
    result: *mut c_void,
    tag: *mut c_char,
) {
    SIM_STATE.with(|sim_state| {
        let res = &mut sim_state.borrow_mut().res;
        let res_id = result as usize;
        let b = if res.is_empty() {
            // No static measurements have been used, so default to dynamic handling.
            __quantum__rt__result_equal(result, __quantum__rt__result_get_one())
        } else {
            if res.len() < res_id + 1 {
                res.resize(res_id + 1, false);
            }
            *res.get(res_id)
                .expect("Result with given id missing after expansion.")
        };

        let val: i64 = i64::from(b);
        record_output("RESULT", &val, tag).expect("Failed to write result output");
    });
}

/// QIR API that allocates the next available qubit in the simulation.
#[no_mangle]
pub extern "C" fn __quantum__rt__qubit_allocate() -> *mut c_void {
    SIM_STATE.with(|sim_state| {
        let mut state = sim_state.borrow_mut();
        let qubit_id = state.sim.allocate();

        // Increase the max qubit id global so that `ensure_sufficient_qubits` wont trigger more allocations.
        // NOTE: static allocation and dynamic allocation shouldn't be used together, so this is safe to do.
        state.max_qubit_id = state.max_qubit_id.max(qubit_id + 1);

        qubit_id as *mut c_void
    })
}

/// QIR API for allocating the given number of qubits in the simulation, returning them as a runtime managed array.
/// # Panics
/// This function will panic if the requested array size is too large to be described with the system pointer size.
#[allow(clippy::cast_ptr_alignment)]
#[no_mangle]
pub extern "C" fn __quantum__rt__qubit_allocate_array(size: u64) -> *const QirArray {
    let arr = __quantum__rt__array_create_1d(
        size_of::<usize>()
            .try_into()
            .expect("System pointer size too large to be described with u32."),
        size,
    );
    for index in 0..size {
        unsafe {
            let elem = __quantum__rt__array_get_element_ptr_1d(arr, index).cast::<*mut c_void>();
            *elem = __quantum__rt__qubit_allocate();
        }
    }
    arr
}

/// QIR API for releasing the given runtime managed qubit array.
/// # Safety
///
/// This function should only be called with arrays created by `__quantum__rt__qubit_allocate_array`.
#[allow(clippy::cast_ptr_alignment)]
#[no_mangle]
pub unsafe extern "C" fn __quantum__rt__qubit_release_array(arr: *const QirArray) {
    for index in 0..__quantum__rt__array_get_size_1d(arr) {
        let elem = __quantum__rt__array_get_element_ptr_1d(arr, index).cast::<*mut c_void>();
        __quantum__rt__qubit_release(*elem);
    }
    __quantum__rt__array_update_alias_count(arr, -1);
}

/// QIR API for releasing the given qubit from the simulation.
#[no_mangle]
pub extern "C" fn __quantum__rt__qubit_release(qubit: *mut c_void) {
    SIM_STATE.with(|sim_state| {
        let mut state = sim_state.borrow_mut();
        state.sim.release(qubit as usize);
    });
}

/// QIR API for getting the string interpretation of a qubit identifier.
/// # Panics
/// This function will panic if memory cannot be allocated for the underyling string.
#[no_mangle]
pub extern "C" fn __quantum__rt__qubit_to_string(qubit: *mut c_void) -> *const CString {
    unsafe {
        __quantum__rt__string_create(
            CString::new(format!("{}", qubit as usize))
                .expect("Unable to allocate memory for qubit string.")
                .as_bytes_with_nul()
                .as_ptr() as *mut c_char,
        )
    }
}

/// Rust API for getting a snapshot of current quantum state. The state is a sorted copy of
/// the current sparse state represented by a vector of pairs of indices and complex numbers along
/// with the total number of currently allocated qubits to help in interpreting the state.
#[must_use]
pub fn capture_quantum_state() -> (Vec<(BigUint, Complex64)>, usize) {
    SIM_STATE.with(|sim_state| {
        let mut state = sim_state.borrow_mut();
        state.sim.get_state()
    })
}

/// QIR API for dumping full internal simulator state.
/// # Panics
/// This function will panic if the output buffer is not available.
#[no_mangle]
pub extern "C" fn __quantum__qis__dumpmachine__body(location: *mut c_void) {
    if !location.is_null() {
        unimplemented!("Dump to location is not implemented.")
    }
    SIM_STATE.with(|sim_state| {
        let mut state = sim_state.borrow_mut();

        if !state.res.is_empty() {
            OUTPUT.with(|output| {
                let mut output = output.borrow_mut();
                output
                    .write_fmt(format_args!("Global Results: {}", state.res))
                    .expect("Failed to write global results");
                output.write_newline();
            });
        }
        OUTPUT.with(|output| {
            let mut output = output.borrow_mut();
            output
                .write_all(state.sim.dump().as_bytes())
                .expect("Failed to write simulator state");
        });
    });
}

/// QIR API for the barrier operation. This is a no-op in simulation.
#[no_mangle]
pub extern "C" fn __quantum__qis__barrier__body() {
    // No-op
}

#[cfg(test)]
mod tests {
    use std::{f64::consts::PI, ffi::c_void, ptr::null_mut};

    use crate::{
        __quantum__qis__cnot__body, __quantum__qis__cx__body, __quantum__qis__cz__body,
        __quantum__qis__dumpmachine__body, __quantum__qis__h__body, __quantum__qis__m__body,
        __quantum__qis__mresetz__body, __quantum__qis__mz__body, __quantum__qis__read_result__body,
        __quantum__qis__rx__body, __quantum__qis__rxx__body, __quantum__qis__ry__body,
        __quantum__qis__ryy__body, __quantum__qis__rz__body, __quantum__qis__rzz__body,
        __quantum__qis__s__adj, __quantum__qis__s__body, __quantum__qis__x__body,
        __quantum__rt__qubit_allocate, __quantum__rt__qubit_allocate_array,
        __quantum__rt__qubit_release, __quantum__rt__qubit_release_array,
        __quantum__rt__result_equal, capture_quantum_state, map_to_z_basis, qubit_is_zero,
        result_bool::__quantum__rt__result_get_one, unmap_from_z_basis, SIM_STATE,
    };
    use num_bigint::BigUint;
    use qir_stdlib::{
        arrays::{
            __quantum__rt__array_create_1d, __quantum__rt__array_get_element_ptr_1d,
            __quantum__rt__array_update_reference_count,
        },
        Pauli,
    };

    #[test]
    fn basic_test_static() {
        let q0 = 5 as *mut c_void;
        let r0 = std::ptr::null_mut();
        let r1 = 1 as *mut c_void;
        __quantum__qis__mz__body(q0, r0);
        assert!(!__quantum__qis__read_result__body(r0));
        __quantum__qis__x__body(q0);
        __quantum__qis__mz__body(q0, r1);
        assert!(__quantum__qis__read_result__body(r1));
        __quantum__qis__x__body(q0);
        __quantum__qis__mz__body(q0, r0);
        assert!(!__quantum__qis__read_result__body(r0));
        assert!(!__quantum__qis__read_result__body(3 as *mut c_void));
        __quantum__qis__dumpmachine__body(null_mut());
    }

    #[allow(clippy::cast_ptr_alignment)]
    #[test]
    fn basic_test_dynamic() {
        let q1 = __quantum__rt__qubit_allocate();
        let q2 = __quantum__rt__qubit_allocate();
        __quantum__qis__h__body(q1);
        __quantum__qis__cnot__body(q1, q2);
        let r1 = __quantum__qis__m__body(q1);
        let r2 = __quantum__qis__m__body(q2);
        assert!(__quantum__rt__result_equal(r1, r2));
        __quantum__qis__dumpmachine__body(null_mut());
        __quantum__rt__qubit_release(q2);
        __quantum__rt__qubit_release(q1);
        let qs = __quantum__rt__qubit_allocate_array(4);
        unsafe {
            let q_elem = __quantum__rt__array_get_element_ptr_1d(qs, 3).cast::<*mut c_void>();
            __quantum__qis__x__body(*q_elem);
            __quantum__qis__dumpmachine__body(null_mut());
            let r = __quantum__qis__m__body(*q_elem);
            assert!(__quantum__rt__result_equal(
                r,
                __quantum__rt__result_get_one()
            ));
            __quantum__rt__qubit_release_array(qs);
        }
    }

    #[test]
    fn test_qubit_is_zero() {
        let q0 = __quantum__rt__qubit_allocate();
        assert!(qubit_is_zero(q0));
        __quantum__qis__x__body(q0);
        assert!(!qubit_is_zero(q0));
        __quantum__qis__h__body(q0);
        assert!(!qubit_is_zero(q0));
        let r = __quantum__qis__m__body(q0);
        assert!(
            qubit_is_zero(q0) != __quantum__rt__result_equal(r, __quantum__rt__result_get_one())
        );
    }

    #[allow(clippy::cast_ptr_alignment)]
    #[test]
    fn test_map_unmap_are_adjoint() {
        unsafe fn check_map_unmap(pauli: Pauli) {
            let check_qubit = __quantum__rt__qubit_allocate();
            let qubits = __quantum__rt__qubit_allocate_array(1);
            let q = *__quantum__rt__array_get_element_ptr_1d(qubits, 0).cast::<*mut c_void>();
            let paulis = __quantum__rt__array_create_1d(1, 1);
            *__quantum__rt__array_get_element_ptr_1d(paulis, 0).cast::<Pauli>() = pauli;

            __quantum__qis__h__body(check_qubit);
            __quantum__qis__cnot__body(check_qubit, q);

            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                let combined_list = map_to_z_basis(state, paulis, qubits);
                unmap_from_z_basis(state, combined_list);
            });

            __quantum__qis__cnot__body(check_qubit, q);
            __quantum__qis__h__body(check_qubit);

            assert!(qubit_is_zero(q));
            assert!(qubit_is_zero(check_qubit));

            __quantum__rt__array_update_reference_count(paulis, -1);
            __quantum__rt__qubit_release_array(qubits);
            __quantum__rt__qubit_release(check_qubit);
        }

        unsafe {
            check_map_unmap(Pauli::X);
            check_map_unmap(Pauli::Y);
            check_map_unmap(Pauli::Z);
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
    #[test]
    fn test_map_pauli_x() {
        let qubits = __quantum__rt__qubit_allocate_array(1);
        unsafe {
            let q = *__quantum__rt__array_get_element_ptr_1d(qubits, 0).cast::<*mut c_void>();
            let paulis = __quantum__rt__array_create_1d(1, 1);
            *__quantum__rt__array_get_element_ptr_1d(paulis, 0).cast::<Pauli>() = Pauli::X;

            __quantum__qis__h__body(q);

            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                let _ = map_to_z_basis(state, paulis, qubits);
            });

            qubit_is_zero(q);

            __quantum__rt__array_update_reference_count(paulis, -1);
            __quantum__rt__qubit_release_array(qubits);
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
    #[test]
    fn test_map_pauli_y() {
        let qubits = __quantum__rt__qubit_allocate_array(1);
        unsafe {
            let q = *__quantum__rt__array_get_element_ptr_1d(qubits, 0).cast::<*mut c_void>();
            let paulis = __quantum__rt__array_create_1d(1, 1);
            *__quantum__rt__array_get_element_ptr_1d(paulis, 0).cast::<Pauli>() = Pauli::Y;

            __quantum__qis__h__body(q);
            __quantum__qis__s__adj(q);
            __quantum__qis__h__body(q);

            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                let _ = map_to_z_basis(state, paulis, qubits);
            });

            qubit_is_zero(q);

            __quantum__rt__array_update_reference_count(paulis, -1);
            __quantum__rt__qubit_release_array(qubits);
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
    #[test]
    fn test_map_pauli_z() {
        let qubits = __quantum__rt__qubit_allocate_array(1);
        unsafe {
            let q = *__quantum__rt__array_get_element_ptr_1d(qubits, 0).cast::<*mut c_void>();
            let paulis = __quantum__rt__array_create_1d(1, 1);
            *__quantum__rt__array_get_element_ptr_1d(paulis, 0).cast::<Pauli>() = Pauli::Z;

            SIM_STATE.with(|sim_state| {
                let state = &mut *sim_state.borrow_mut();
                let _ = map_to_z_basis(state, paulis, qubits);
            });

            qubit_is_zero(q);

            __quantum__rt__array_update_reference_count(paulis, -1);
            __quantum__rt__qubit_release_array(qubits);
        }
    }

    #[test]
    fn test_joint_zz() {
        let check_qubit = __quantum__rt__qubit_allocate();
        let q0 = __quantum__rt__qubit_allocate();
        let q1 = __quantum__rt__qubit_allocate();

        __quantum__qis__h__body(check_qubit);
        __quantum__qis__cx__body(check_qubit, q0);
        __quantum__qis__cx__body(check_qubit, q1);

        __quantum__qis__rzz__body(PI / 2.0, q0, q1);
        __quantum__qis__rz__body(-PI / 2.0, q0);
        __quantum__qis__rz__body(-PI / 2.0, q1);

        __quantum__qis__cz__body(q0, q1);

        __quantum__qis__cx__body(check_qubit, q1);
        __quantum__qis__cx__body(check_qubit, q0);
        __quantum__qis__h__body(check_qubit);

        assert!(qubit_is_zero(check_qubit));
        assert!(qubit_is_zero(q0));
        assert!(qubit_is_zero(q1));
    }

    #[test]
    fn test_joint_yy() {
        let check_qubit = __quantum__rt__qubit_allocate();
        let q0 = __quantum__rt__qubit_allocate();
        let q1 = __quantum__rt__qubit_allocate();

        __quantum__qis__h__body(check_qubit);
        __quantum__qis__cx__body(check_qubit, q0);
        __quantum__qis__cx__body(check_qubit, q1);

        __quantum__qis__h__body(q0);
        __quantum__qis__s__adj(q0);
        __quantum__qis__h__body(q0);
        __quantum__qis__h__body(q1);
        __quantum__qis__s__adj(q1);
        __quantum__qis__h__body(q1);

        __quantum__qis__ryy__body(PI / 2.0, q0, q1);
        __quantum__qis__ry__body(-PI / 2.0, q0);
        __quantum__qis__ry__body(-PI / 2.0, q1);

        __quantum__qis__h__body(q1);
        __quantum__qis__s__body(q1);
        __quantum__qis__h__body(q1);
        __quantum__qis__h__body(q0);
        __quantum__qis__s__body(q0);
        __quantum__qis__h__body(q0);

        __quantum__qis__cz__body(q0, q1);

        __quantum__qis__cx__body(check_qubit, q1);
        __quantum__qis__cx__body(check_qubit, q0);
        __quantum__qis__h__body(check_qubit);

        assert!(qubit_is_zero(check_qubit));
        assert!(qubit_is_zero(q0));
        assert!(qubit_is_zero(q1));
    }

    #[test]
    fn test_joint_xx() {
        let check_qubit = __quantum__rt__qubit_allocate();
        let q0 = __quantum__rt__qubit_allocate();
        let q1 = __quantum__rt__qubit_allocate();

        __quantum__qis__h__body(check_qubit);
        __quantum__qis__cx__body(check_qubit, q0);
        __quantum__qis__cx__body(check_qubit, q1);

        __quantum__qis__h__body(q0);
        __quantum__qis__h__body(q1);

        __quantum__qis__rxx__body(PI / 2.0, q0, q1);
        __quantum__qis__rx__body(-PI / 2.0, q0);
        __quantum__qis__rx__body(-PI / 2.0, q1);

        __quantum__qis__h__body(q0);
        __quantum__qis__h__body(q1);

        __quantum__qis__cz__body(q0, q1);

        __quantum__qis__cx__body(check_qubit, q1);
        __quantum__qis__cx__body(check_qubit, q0);
        __quantum__qis__h__body(check_qubit);

        assert!(qubit_is_zero(check_qubit));
        assert!(qubit_is_zero(q0));
        assert!(qubit_is_zero(q1));
    }

    #[test]
    fn test_mresetz() {
        let qubit = __quantum__rt__qubit_allocate();
        let r0 = std::ptr::null_mut();
        let r1 = 1 as *mut c_void;
        assert!(qubit_is_zero(qubit));
        __quantum__qis__mresetz__body(qubit, r0);
        assert!(!__quantum__qis__read_result__body(r0));
        assert!(qubit_is_zero(qubit));
        __quantum__qis__x__body(qubit);
        __quantum__qis__mresetz__body(qubit, r1);
        assert!(__quantum__qis__read_result__body(r1));
        assert!(qubit_is_zero(qubit));
    }

    #[test]
    fn test_capture_quantum_state() {
        let qubit = __quantum__rt__qubit_allocate();
        let (state, qubit_count) = capture_quantum_state();
        assert_eq!(qubit_count, 1);
        assert_eq!(state.len(), 1);
        assert_eq!(state[0].0, BigUint::from(0u32));
        __quantum__qis__x__body(qubit);
        let (state, qubit_count) = capture_quantum_state();
        assert_eq!(qubit_count, 1);
        assert_eq!(state.len(), 1);
        assert_eq!(state[0].0, BigUint::from(1u32));
        __quantum__qis__h__body(qubit);
        let qubit2 = __quantum__rt__qubit_allocate();
        let (state, qubit_count) = capture_quantum_state();
        assert_eq!(qubit_count, 2);
        assert_eq!(state.len(), 2);
        assert_eq!(state[0].1, -state[1].1);
        __quantum__qis__h__body(qubit);
        __quantum__qis__x__body(qubit);
        let (state, qubit_count) = capture_quantum_state();
        assert_eq!(qubit_count, 2);
        assert_eq!(state.len(), 1);
        assert_eq!(state[0].0, BigUint::from(0u32));
        __quantum__rt__qubit_release(qubit);
        __quantum__rt__qubit_release(qubit2);
        let (state, qubit_count) = capture_quantum_state();
        assert_eq!(qubit_count, 0);
        assert_eq!(state.len(), 1);
        assert_eq!(state[0].0, BigUint::from(0u32));
    }
}
