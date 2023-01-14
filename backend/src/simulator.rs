// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

use crate::nearly_zero::NearlyZero;
use num_bigint::BigUint;
use num_complex::Complex64;
use num_traits::{One, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{cell::RefCell, f64::consts::FRAC_1_SQRT_2};

pub type SparseState = FxHashMap<BigUint, Complex64>;

/// The `QuantumSim` struct contains the necessary state for tracking the simulation. Each instance of a
/// `QuantumSim` represents an independant simulation.
pub(crate) struct QuantumSim {
    /// The structure that describes the current quantum state.
    pub(crate) state: FxHashMap<BigUint, Complex64>,

    /// The mapping from qubit identifiers to internal state locations.
    pub(crate) id_map: FxHashMap<usize, usize>,

    /// The bitmap that tracks whether a given qubit has an pending H operation queued on it.
    h_flag: BigUint,

    /// The map for tracking queued Pauli-X rotations by a given angle for a given qubit.
    rx_queue: FxHashMap<usize, f64>,

    /// The map for tracking queued Pauli-Y rotations by a given angle for a given qubit.
    ry_queue: FxHashMap<usize, f64>,
}

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

pub(crate) fn set_rng_seed(seed: u64) {
    RNG.with(|rng| rng.replace(StdRng::seed_from_u64(seed)));
}

/// Levels for flushing of queued gates.
#[derive(Debug, Copy, Clone)]
pub(crate) enum FlushLevel {
    H,
    HRx,
    HRxRy,
}

impl Default for QuantumSim {
    fn default() -> Self {
        Self::new()
    }
}

/// Provides the common set of functionality across all quantum simulation types.
impl QuantumSim {
    /// Creates a new sparse state quantum simulator object with empty initial state (no qubits allocated, no operations buffered).
    #[must_use]
    fn new() -> Self {
        let mut initial_state = FxHashMap::default();
        initial_state.insert(BigUint::zero(), Complex64::one());

        QuantumSim {
            state: initial_state,
            id_map: FxHashMap::default(),
            h_flag: BigUint::zero(),
            rx_queue: FxHashMap::default(),
            ry_queue: FxHashMap::default(),
        }
    }

    /// Allocates a fresh qubit, returning its identifier. Note that this will use the lowest available
    /// identifier, and may result in qubits being allocated "in the middle" of an existing register
    /// if those identifiers are available.
    #[must_use]
    pub(crate) fn allocate(&mut self) -> usize {
        // Add the new entry into the FxHashMap at the first available sequential ID and first available
        // sequential location.
        let mut sorted_keys: Vec<&usize> = self.id_map.keys().collect();
        sorted_keys.sort();
        let mut sorted_vals: Vec<&usize> = self.id_map.values().collect();
        sorted_vals.sort();
        let new_key = sorted_keys
            .iter()
            .enumerate()
            .take_while(|(index, key)| index == **key)
            .last()
            .map_or(0_usize, |(_, &&key)| key + 1);
        let new_val = sorted_vals
            .iter()
            .enumerate()
            .take_while(|(index, val)| index == **val)
            .last()
            .map_or(0_usize, |(_, &&val)| val + 1);
        self.id_map.insert(new_key, new_val);

        // Return the new ID that was used.
        new_key
    }

    /// Releases the given qubit, collapsing its state in the process. After release that identifier is
    /// no longer valid for use in other functions and will cause an error if used.
    /// # Panics
    ///
    /// The function will panic if the given id does not correpsond to an allocated qubit.
    pub(crate) fn release(&mut self, id: usize) {
        self.flush_queue(&[id], FlushLevel::HRxRy);

        let loc = self
            .id_map
            .remove(&id)
            .unwrap_or_else(|| panic!("Unable to find qubit with id {}.", id));

        // Measure and collapse the state for this qubit.
        let res = self.measure_impl(loc);

        // If the result of measurement was true then we must set the bit for this qubit in every key
        // to zero to "reset" the qubit.
        if res {
            self.state = self
                .state
                .drain()
                .fold(FxHashMap::default(), |mut accum, (mut k, v)| {
                    k.set_bit(loc as u64, false);
                    accum.insert(k, v);
                    accum
                });
        }
    }

    /// Prints the current state vector to standard output with integer labels for the states, skipping any
    /// states with zero amplitude.
    pub(crate) fn dump(&mut self, output: &mut impl std::io::Write) {
        // Swap all the entries in the state to be ordered by qubit identifier. This makes
        // interpreting the state easier for external consumers that don't have access to the id map.
        let mut sorted_keys: Vec<usize> = self.id_map.keys().copied().collect();
        self.flush_queue(&sorted_keys, FlushLevel::HRxRy);

        sorted_keys.sort_unstable();
        sorted_keys.iter().enumerate().for_each(|(index, &key)| {
            if index != self.id_map[&key] {
                self.swap_qubit_state(self.id_map[&key], index);
                let swapped_key = *self
                    .id_map
                    .iter()
                    .find(|(_, &value)| value == index)
                    .unwrap()
                    .0;
                *(self.id_map.get_mut(&swapped_key).unwrap()) = self.id_map[&key];
                *(self.id_map.get_mut(&key).unwrap()) = index;
            }
        });

        self.dump_impl(false, output);
    }

    /// Utility function that performs the actual output of state (and optionally map) to screen. Can
    /// be called internally from other functions to aid in debugging and does not perform any modification
    /// of the internal structures.
    fn dump_impl(&self, print_id_map: bool, output: &mut impl std::io::Write) {
        if print_id_map {
            output
                .write_fmt(format_args!("MAP: {:?}\n", self.id_map))
                .expect("Unable to write to output");
        };
        output
            .write_fmt(format_args!("{{ "))
            .expect("Unable to write to output");
        let mut sorted_keys = self.state.keys().collect::<Vec<_>>();
        sorted_keys.sort_unstable();
        let (last_key, most_keys) = sorted_keys.split_last().unwrap();
        for key in most_keys {
            let val = self.state.get(key).map_or_else(Complex64::zero, |v| *v);
            output
                .write_fmt(format_args!(
                    "\"|{}\u{27e9}\": [{}, {}], ",
                    key, val.re, val.im
                ))
                .expect("Unable to write to output");
        }
        let last_val = self
            .state
            .get(last_key)
            .map_or_else(Complex64::zero, |v| *v);
        output
            .write_fmt(format_args!(
                "\"|{}\u{27e9}\": [{}, {}] }}\n",
                last_key, last_val.re, last_val.im
            ))
            .expect("Unable to write to output");
    }

    /// Checks the probability of parity measurement in the computational basis for the given set of
    /// qubits.
    /// # Panics
    ///
    /// This function will panic if the given ids do not all correspond to allocated qubits.
    /// This function will panic if there are duplicate ids in the given list.
    #[must_use]
    pub(crate) fn joint_probability(&mut self, ids: &[usize]) -> f64 {
        self.flush_queue(ids, FlushLevel::HRxRy);

        Self::check_for_duplicates(ids);
        let locs: Vec<usize> = ids
            .iter()
            .map(|id| {
                *self
                    .id_map
                    .get(id)
                    .unwrap_or_else(|| panic!("Unable to find qubit with id {}", id))
            })
            .collect();

        self.check_joint_probability(&locs)
    }

    /// Measures the qubit with the given id, collapsing the state based on the measured result.
    /// # Panics
    ///
    /// This funciton will panic if the given identifier does not correspond to an allocated qubit.
    #[must_use]
    pub(crate) fn measure(&mut self, id: usize) -> bool {
        self.flush_queue(&[id], FlushLevel::HRxRy);

        self.measure_impl(
            *self
                .id_map
                .get(&id)
                .unwrap_or_else(|| panic!("Unable to find qubit with id {}", id)),
        )
    }

    /// Utility that performs the actual measurement and collapse of the state for the given
    /// location.
    fn measure_impl(&mut self, loc: usize) -> bool {
        let random_sample = RNG.with(|rng| rng.borrow_mut().gen::<f64>());
        let res = random_sample < self.check_joint_probability(&[loc]);
        self.collapse(loc, res);
        res
    }

    /// Performs a joint measurement to get the parity of the given qubits, collapsing the state
    /// based on the measured result.
    /// # Panics
    ///
    /// This function will panic if any of the given identifiers do not correspond to an allocated qubit.
    /// This function will panic if any of the given identifiers are duplicates.
    #[must_use]
    pub(crate) fn joint_measure(&mut self, ids: &[usize]) -> bool {
        self.flush_queue(ids, FlushLevel::HRxRy);

        Self::check_for_duplicates(ids);
        let locs: Vec<usize> = ids
            .iter()
            .map(|id| {
                *self
                    .id_map
                    .get(id)
                    .unwrap_or_else(|| panic!("Unable to find qubit with id {}", id))
            })
            .collect();

        let random_sample = RNG.with(|rng| rng.borrow_mut().gen::<f64>());
        let res = random_sample < self.check_joint_probability(&locs);
        self.joint_collapse(&locs, res);
        res
    }

    /// Utility to get the sum of all probabilies where an odd number of the bits at the given locations
    /// are set. This corresponds to the probability of jointly measuring those qubits in the computational
    /// basis.
    fn check_joint_probability(&self, locs: &[usize]) -> f64 {
        let mask = locs.iter().fold(BigUint::zero(), |accum, loc| {
            accum | (BigUint::one() << loc)
        });
        self.state.iter().fold(0.0_f64, |accum, (index, val)| {
            if (index & &mask).count_ones() & 1 > 0 {
                accum + val.norm_sqr()
            } else {
                accum
            }
        })
    }

    /// Utility to collapse the probability at the given location based on the boolean value. This means
    /// that if the given value is 'true' then all keys in the sparse state where the given location
    /// has a zero bit will be reduced to zero and removed. Then the sparse state is normalized.
    fn collapse(&mut self, loc: usize, val: bool) {
        self.joint_collapse(&[loc], val);
    }

    /// Utility to collapse the joint probability of a particular set of locations in the sparse state.
    /// The entries that do not correspond to the given boolean value are removed, and then the whole
    /// state is normalized.
    fn joint_collapse(&mut self, locs: &[usize], val: bool) {
        let mask = locs.iter().fold(BigUint::zero(), |accum, loc| {
            accum | (BigUint::one() << loc)
        });

        let mut new_state = FxHashMap::default();
        let mut scaling_denominator = 0.0;
        for (k, v) in self.state.drain() {
            if ((&k & &mask).count_ones() & 1 > 0) == val {
                new_state.insert(k, v);
                scaling_denominator += v.norm_sqr();
            }
        }

        // Normalize the new state using the accumulated scaling.
        let scaling = 1.0 / scaling_denominator.sqrt();
        for (k, v) in new_state.drain() {
            let scaled_value = v * scaling;
            if !scaled_value.is_nearly_zero() {
                self.state.insert(k, scaled_value);
            }
        }
    }

    /// Swaps the mapped ids for the given qubits.
    pub(crate) fn swap_qubit_ids(&mut self, qubit1: usize, qubit2: usize) {
        // Must also swap any queued operations.
        let (h_val1, h_val2) = (
            self.h_flag.bit(qubit1 as u64),
            self.h_flag.bit(qubit2 as u64),
        );
        self.h_flag.set_bit(qubit1 as u64, h_val2);
        self.h_flag.set_bit(qubit2 as u64, h_val1);

        if let Some(rx_val) = self.rx_queue.get(&qubit1) {
            self.rx_queue.insert(qubit2, *rx_val);
        }
        if let Some(rx_val) = self.rx_queue.get(&qubit1) {
            self.rx_queue.insert(qubit1, *rx_val);
        }

        if let Some(ry_val) = self.ry_queue.get(&qubit1) {
            self.ry_queue.insert(qubit2, *ry_val);
        }
        if let Some(ry_val) = self.ry_queue.get(&qubit1) {
            self.ry_queue.insert(qubit1, *ry_val);
        }

        let qubit1_mapped = *self
            .id_map
            .get(&qubit1)
            .unwrap_or_else(|| panic!("Unable to find qubit with id {}", qubit1));
        let qubit2_mapped = *self
            .id_map
            .get(&qubit2)
            .unwrap_or_else(|| panic!("Unable to find qubit with id {}", qubit2));
        *self.id_map.get_mut(&qubit1).unwrap() = qubit2_mapped;
        *self.id_map.get_mut(&qubit2).unwrap() = qubit1_mapped;
    }

    /// Swaps the states of two qubits throughout the sparse state map.
    pub(crate) fn swap_qubit_state(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }

        self.flush_queue(&[qubit1, qubit2], FlushLevel::HRxRy);

        let (q1, q2) = (qubit1 as u64, qubit2 as u64);

        // Swap entries in the sparse state to correspond to swapping of two qubits' locations.
        self.state = self
            .state
            .drain()
            .fold(FxHashMap::default(), |mut accum, (k, v)| {
                if k.bit(q1) == k.bit(q2) {
                    accum.insert(k, v);
                } else {
                    let mut new_k = k.clone();
                    new_k.set_bit(q1, !k.bit(q1));
                    new_k.set_bit(q2, !k.bit(q2));
                    accum.insert(new_k, v);
                }
                accum
            });
    }

    pub(crate) fn check_for_duplicates(ids: &[usize]) {
        let mut unique = FxHashSet::default();
        for id in ids.iter() {
            assert!(
                unique.insert(id),
                "Duplicate qubit id '{}' found in application.",
                id
            );
        }
    }

    /// Verifies that the given target and list of controls does not contain any duplicate entries, and returns
    /// those values mapped to internal identifiers and converted to `u64`.
    fn resolve_and_check_qubits(&self, target: usize, ctls: &[usize]) -> (u64, Vec<u64>) {
        let mut ids = ctls.to_owned();
        ids.push(target);
        Self::check_for_duplicates(&ids);

        let target = *self
            .id_map
            .get(&target)
            .unwrap_or_else(|| panic!("Unable to find qubit with id {}", target))
            as u64;

        let ctls: Vec<u64> = ctls
            .iter()
            .map(|c| {
                *self
                    .id_map
                    .get(c)
                    .unwrap_or_else(|| panic!("Unable to find qubit with id {}", c))
                    as u64
            })
            .collect();

        (target, ctls)
    }

    pub(crate) fn flush_queue(&mut self, qubits: &[usize], level: FlushLevel) {
        for target in qubits {
            if self.h_flag.bit(*target as u64) {
                self.apply_mch(&[], *target);
                self.h_flag.set_bit(*target as u64, false);
            }
            match level {
                FlushLevel::H => (),
                FlushLevel::HRx => self.flush_rx(*target),
                FlushLevel::HRxRy => {
                    self.flush_rx(*target);
                    self.flush_ry(*target);
                }
            }
        }
    }

    fn flush_rx(&mut self, target: usize) {
        if let Some(theta) = self.rx_queue.get(&target) {
            self.mcrotation(&[], *theta, target, false);
            self.rx_queue.remove(&target);
        }
    }

    fn flush_ry(&mut self, target: usize) {
        if let Some(theta) = self.ry_queue.get(&target) {
            self.mcrotation(&[], *theta, target, true);
            self.ry_queue.remove(&target);
        }
    }

    /// Utility for performing an in-place update of the state vector with the given target and controls.
    /// Here, "in-place" indicates that the given transformation operation can calulate a new entry in the
    /// state vector using only one entry of the state vector as input and does not need to refer to any
    /// other entries. This covers the multicontrolled gates except for H, Rx, and Ry and notably keeps the
    /// size of the state vector the same.
    fn controlled_gate<F>(&mut self, ctls: &[usize], target: usize, mut op: F)
    where
        F: FnMut((BigUint, Complex64), u64) -> (BigUint, Complex64),
    {
        let (target, ctls) = self.resolve_and_check_qubits(target, ctls);

        self.state = self.state.drain().into_iter().fold(
            SparseState::default(),
            |mut accum, (index, value)| {
                let (k, v) = if ctls.iter().all(|c| index.bit(*c as u64)) {
                    op((index, value), target as u64)
                } else {
                    (index, value)
                };
                if !v.is_nearly_zero() {
                    accum.insert(k, v);
                }
                accum
            },
        );
    }

    /// Performs the Pauli-X transformation on a single state.
    fn x_transform((mut index, val): (BigUint, Complex64), target: u64) -> (BigUint, Complex64) {
        index.set_bit(target, !index.bit(target));
        (index, val)
    }

    /// Single qubit X gate.
    pub(crate) fn x(&mut self, target: usize) {
        if let Some(entry) = self.ry_queue.get_mut(&target) {
            // XY = -YX, so switch the sign on any queued Ry rotations.
            *entry *= -1.0;
        }
        if self.h_flag.bit(target as u64) {
            // XH = HZ, so execute a Z transformation if there is an H queued.
            self.controlled_gate(&[], target, Self::z_transform);
        } else {
            self.controlled_gate(&[], target, Self::x_transform);
        }
    }

    /// Multi-controlled X gate.
    pub(crate) fn mcx(&mut self, ctls: &[usize], target: usize) {
        if ctls.is_empty() {
            self.x(target);
            return;
        }

        if ctls.len() > 1
            || self.ry_queue.contains_key(&ctls[0])
            || self.rx_queue.contains_key(&ctls[0])
            || (self.h_flag.bit(ctls[0] as u64) && !self.h_flag.bit(target as u64))
        {
            self.flush_queue(ctls, FlushLevel::HRxRy);
        }

        if self.ry_queue.contains_key(&target) {
            self.flush_queue(&[target], FlushLevel::HRxRy);
        }

        if self.h_flag.bit(target as u64) {
            if ctls.len() == 1 && self.h_flag.bit(ctls[0] as u64) {
                // An H on both target and single control means we can perform a CNOT with the control
                // and target switched.
                self.controlled_gate(&[target], ctls[0], Self::x_transform);
            } else {
                // XH = HZ, so perform a mulit-controlled Z here.
                self.controlled_gate(ctls, target, Self::z_transform);
            }
        } else {
            self.controlled_gate(ctls, target, Self::x_transform);
        }
    }

    /// Performs the Pauli-Y transformation on a single state.
    fn y_transform(
        (mut index, mut val): (BigUint, Complex64),
        target: u64,
    ) -> (BigUint, Complex64) {
        index.set_bit(target, !index.bit(target));
        val *= if index.bit(target) {
            Complex64::i()
        } else {
            -Complex64::i()
        };
        (index, val)
    }

    /// Single qubit Y gate.
    pub(crate) fn y(&mut self, target: usize) {
        if let Some(entry) = self.rx_queue.get_mut(&target) {
            // XY = -YX, so flip the sign on any queued Rx rotation.
            *entry *= -1.0;
        }

        self.controlled_gate(&[], target, Self::y_transform);
    }

    /// Multi-controlled Y gate.
    pub(crate) fn mcy(&mut self, ctls: &[usize], target: usize) {
        if ctls.is_empty() {
            self.y(target);
            return;
        }

        self.flush_queue(ctls, FlushLevel::HRxRy);

        if self.rx_queue.contains_key(&target) {
            self.flush_queue(&[target], FlushLevel::HRx);
        }

        if self.h_flag.bit(target as u64) {
            // HY = -YH, so add a phase to one of the controls.
            let (target, ctls) = ctls
                .split_first()
                .expect("Controls list cannot be empty here.");
            self.controlled_gate(ctls, *target, Self::z_transform);
        }

        self.controlled_gate(ctls, target, Self::y_transform);
    }

    /// Performs a phase transformation (a rotation in the computational basis) on a single state.
    fn phase_transform(
        phase: Complex64,
        (index, val): (BigUint, Complex64),
        target: u64,
    ) -> (BigUint, Complex64) {
        let val = val
            * if index.bit(target) {
                phase
            } else {
                Complex64::one()
            };
        (index, val)
    }

    /// Multi-controlled phase rotation ("G" gate).
    pub(crate) fn mcphase(&mut self, ctls: &[usize], phase: Complex64, target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(ctls, target, |(index, val), target| {
            Self::phase_transform(phase, (index, val), target)
        });
    }

    /// Performs the Pauli-Z transformation on a single state.
    fn z_transform((index, val): (BigUint, Complex64), target: u64) -> (BigUint, Complex64) {
        Self::phase_transform(-Complex64::one(), (index, val), target)
    }

    /// Single qubit Z gate.
    pub(crate) fn z(&mut self, target: usize) {
        if let Some(entry) = self.ry_queue.get_mut(&target) {
            // ZY = -YZ, so flip the sign on any queued Ry rotations.
            *entry *= -1.0;
        }

        if let Some(entry) = self.rx_queue.get_mut(&target) {
            // ZX = -XZ, so flip the sign on any queued Rx rotations.
            *entry *= -1.0;
        }

        if self.h_flag.bit(target as u64) {
            // HZ = XH, so execute an X if an H is queued.
            self.controlled_gate(&[], target, Self::x_transform);
        } else {
            self.controlled_gate(&[], target, Self::z_transform);
        }
    }

    /// Multi-controlled Z gate.
    pub(crate) fn mcz(&mut self, ctls: &[usize], target: usize) {
        if ctls.is_empty() {
            self.z(target);
            return;
        }

        // Count up the instances of queued H and Rx/Ry on controls and target, treating rotations as 2.
        let count = ctls.iter().fold(0, |accum, c| {
            accum
                + i32::from(self.h_flag.bit(*c as u64))
                + if self.rx_queue.contains_key(c) || self.ry_queue.contains_key(c) {
                    2
                } else {
                    0
                }
        }) + i32::from(self.h_flag.bit(target as u64))
            + if self.rx_queue.contains_key(&target) || self.ry_queue.contains_key(&target) {
                2
            } else {
                0
            };

        if count == 1 {
            // Only when count is exactly one can we optimize, meaning there is exactly one H on either
            // the target or one control. Create a new controls list and target where the target is whichever
            // qubit has the H queued.
            let (ctls, target): (Vec<usize>, usize) =
                if let Some(h_ctl) = ctls.iter().find(|c| self.h_flag.bit(**c as u64)) {
                    // The H is queued on one control, so create a new controls list that swaps that control for the original target.
                    (
                        ctls.iter()
                            .map(|c| if c == h_ctl { target } else { *c })
                            .collect(),
                        *h_ctl,
                    )
                } else {
                    // The H is queued on the target, so use the original values.
                    (ctls.to_owned(), target)
                };
            // With a single H queued, treat the multi-controlled Z as a multi-controlled X.
            self.controlled_gate(&ctls, target, Self::x_transform);
        } else {
            self.flush_queue(ctls, FlushLevel::HRxRy);
            self.flush_queue(&[target], FlushLevel::HRxRy);
            self.controlled_gate(ctls, target, Self::z_transform);
        }
    }

    /// Performs the S transformation on a single state.
    fn s_transform((index, val): (BigUint, Complex64), target: u64) -> (BigUint, Complex64) {
        Self::phase_transform(Complex64::i(), (index, val), target)
    }

    /// Single qubit S gate.
    pub(crate) fn s(&mut self, target: usize) {
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(&[], target, Self::s_transform);
    }

    /// Multi-controlled S gate.
    pub(crate) fn mcs(&mut self, ctls: &[usize], target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(ctls, target, Self::s_transform);
    }

    /// Performs the adjoint S transformation on a signle state.
    fn sadj_transform((index, val): (BigUint, Complex64), target: u64) -> (BigUint, Complex64) {
        Self::phase_transform(-Complex64::i(), (index, val), target)
    }

    /// Single qubit Adjoint S Gate.
    pub(crate) fn sadj(&mut self, target: usize) {
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(&[], target, Self::sadj_transform);
    }

    /// Multi-controlled Adjoint S gate.
    pub(crate) fn mcsadj(&mut self, ctls: &[usize], target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(ctls, target, Self::sadj_transform);
    }

    /// Performs the T transformation on a single state.
    fn t_transform((index, val): (BigUint, Complex64), target: u64) -> (BigUint, Complex64) {
        Self::phase_transform(
            Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2),
            (index, val),
            target,
        )
    }

    /// Single qubit T gate.
    pub(crate) fn t(&mut self, target: usize) {
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(&[], target, Self::t_transform);
    }

    /// Multi-controlled T gate.
    pub(crate) fn mct(&mut self, ctls: &[usize], target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(ctls, target, Self::t_transform);
    }

    /// Performs the adjoint T transformation to a single state.
    fn tadj_transform((index, val): (BigUint, Complex64), target: u64) -> (BigUint, Complex64) {
        Self::phase_transform(
            Complex64::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            (index, val),
            target,
        )
    }

    /// Single qubit Adjoint T gate.
    pub(crate) fn tadj(&mut self, target: usize) {
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(&[], target, Self::tadj_transform);
    }

    /// Multi-controlled Adjoint T gate.
    pub(crate) fn mctadj(&mut self, ctls: &[usize], target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(ctls, target, Self::tadj_transform);
    }

    /// Performs the Rz transformation with the given angle to a single state.
    fn rz_transform(
        (index, val): (BigUint, Complex64),
        theta: f64,
        target: u64,
    ) -> (BigUint, Complex64) {
        let val = val
            * Complex64::exp(Complex64::new(
                0.0,
                theta / 2.0 * if index.bit(target) { 1.0 } else { -1.0 },
            ));
        (index, val)
    }

    /// Single qubit Rz gate.
    pub(crate) fn rz(&mut self, theta: f64, target: usize) {
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(&[], target, |(index, val), target| {
            Self::rz_transform((index, val), theta, target)
        });
    }

    /// Multi-controlled Rz gate.
    pub(crate) fn mcrz(&mut self, ctls: &[usize], theta: f64, target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);
        self.flush_queue(&[target], FlushLevel::HRxRy);
        self.controlled_gate(ctls, target, |(index, val), target| {
            Self::rz_transform((index, val), theta, target)
        });
    }

    /// Single qubit H gate.
    pub(crate) fn h(&mut self, target: usize) {
        if let Some(entry) = self.ry_queue.get_mut(&target) {
            // YH = -HY, so flip the sign on any queued Ry rotations.
            *entry *= -1.0;
        }

        if self.rx_queue.contains_key(&target) {
            // Can't commute well with queued Rx, so flush those ops.
            self.flush_queue(&[target], FlushLevel::HRx);
        }

        self.h_flag
            .set_bit(target as u64, !self.h_flag.bit(target as u64));
    }

    /// Multi-controlled H gate.
    pub(crate) fn mch(&mut self, ctls: &[usize], target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);
        if self.ry_queue.contains_key(&target) || self.rx_queue.contains_key(&target) {
            self.flush_queue(&[target], FlushLevel::HRxRy);
        }

        self.apply_mch(ctls, target);
    }

    /// Apply the full state transformation corresponding to the multi-controlled H gate. Note that
    /// this can increase the size of the state vector by introducing new non-zero states
    /// or decrease the size by bringing some states to zero.
    fn apply_mch(&mut self, ctls: &[usize], target: usize) {
        let (target, ctls) = self.resolve_and_check_qubits(target, ctls);

        // This operation cannot be done in-place so create a new empty state vector to populate.
        let mut new_state = SparseState::default();

        let mut flipped = BigUint::zero();
        flipped.set_bit(target, true);

        for (index, value) in &self.state {
            if ctls.iter().all(|c| index.bit(*c)) {
                let flipped_index = index ^ &flipped;
                if !self.state.contains_key(&flipped_index) {
                    // The state vector does not have an entry for the state where the target is flipped
                    // and all other qubits are the same, meaning there is no superposition for this state.
                    // Create the additional state caluclating the resulting superposition.
                    let mut zero_bit_index = index.clone();
                    zero_bit_index.set_bit(target, false);
                    new_state.insert(zero_bit_index, value * std::f64::consts::FRAC_1_SQRT_2);

                    let mut one_bit_index = index.clone();
                    one_bit_index.set_bit(target, true);
                    new_state.insert(
                        one_bit_index,
                        value
                            * std::f64::consts::FRAC_1_SQRT_2
                            * (if index.bit(target) { -1.0 } else { 1.0 }),
                    );
                } else if !index.bit(target) {
                    // The state vector already has a superposition for this state, so calculate the resulting
                    // updates using the value from the flipped state. Note we only want to perform this for one
                    // of the states to avoid duplication, so we pick the Zero state by checking the target bit
                    // in the index is not set.
                    let flipped_value = &self.state[&flipped_index];

                    let new_val = (value + flipped_value) as Complex64;
                    if !new_val.is_nearly_zero() {
                        new_state.insert(index.clone(), new_val * std::f64::consts::FRAC_1_SQRT_2);
                    }

                    let new_val = (value - flipped_value) as Complex64;
                    if !new_val.is_nearly_zero() {
                        new_state
                            .insert(index | &flipped, new_val * std::f64::consts::FRAC_1_SQRT_2);
                    }
                }
            } else {
                new_state.insert(index.clone(), *value);
            }
        }

        self.state = new_state;
    }

    /// Performs a rotation in the non-computational basis, which cannot be done in-place. This
    /// corresponds to an Rx or Ry depending on the requested sign flip, and notably can increase or
    /// decrease the size of the state vector.
    fn mcrotation(&mut self, ctls: &[usize], theta: f64, target: usize, sign_flip: bool) {
        // Calculate the matrix entries for the rotation by the given angle, respecting the sign flip.
        let m00 = Complex64::new(f64::cos(theta / 2.0), 0.0);
        let m01 = Complex64::new(0.0, f64::sin(theta / -2.0))
            * if sign_flip {
                -Complex64::i()
            } else {
                Complex64::one()
            };

        if m00.is_nearly_zero() {
            // This is just a Pauli rotation.
            if sign_flip {
                self.mcy(ctls, target);
            } else {
                self.mcx(ctls, target);
            }
        } else if m01.is_nearly_zero() {
            // This is just identity, so we can no-op.
        } else {
            let (target, ctls) = self.resolve_and_check_qubits(target, ctls);
            let mut new_state = SparseState::default();
            let m10 = m01 * if sign_flip { -1.0 } else { 1.0 };
            let mut flipped = BigUint::zero();
            flipped.set_bit(target, true);

            for (index, value) in &self.state {
                if ctls.iter().all(|c| index.bit(*c)) {
                    let flipped_index = index ^ &flipped;
                    if !self.state.contains_key(&flipped_index) {
                        // The state vector doesn't have an entry for the flipped target bit, so there
                        // isn't a superposition. Calculate the superposition using the matrix entries.
                        if index.bit(target) {
                            new_state.insert(flipped_index, value * m01);
                            new_state.insert(index.clone(), value * m00);
                        } else {
                            new_state.insert(index.clone(), value * m00);
                            new_state.insert(flipped_index, value * m10);
                        }
                    } else if !index.bit(target) {
                        // There is already a superposition of the target for this state, so calculate the new
                        // entries using the values from the flipped state. Note we only want to do this for one of
                        // the states, so we pick the Zero state by checking the target bit in the index is not set.
                        let flipped_val = self.state[&flipped_index];

                        let new_val = (value * m00 + flipped_val * m01) as Complex64;
                        if !new_val.is_nearly_zero() {
                            new_state.insert(index.clone(), new_val);
                        }

                        let new_val = (value * m10 + flipped_val * m00) as Complex64;
                        if !new_val.is_nearly_zero() {
                            new_state.insert(flipped_index, new_val);
                        }
                    }
                } else {
                    new_state.insert(index.clone(), *value);
                }
            }

            self.state = new_state;
        }
    }

    /// Single qubit Rx gate.
    pub(crate) fn rx(&mut self, theta: f64, target: usize) {
        self.flush_queue(&[target], FlushLevel::HRxRy);
        if let Some(entry) = self.rx_queue.get_mut(&target) {
            *entry += theta;
            if entry.is_nearly_zero() {
                self.rx_queue.remove(&target);
            }
        } else {
            self.rx_queue.insert(target, theta);
        }
    }

    /// Multi-controlled Rx gate.
    pub(crate) fn mcrx(&mut self, ctls: &[usize], theta: f64, target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);

        if self.ry_queue.contains_key(&target) {
            self.flush_queue(&[target], FlushLevel::HRxRy);
        } else if self.h_flag.bit(target as u64) {
            self.flush_queue(&[target], FlushLevel::H);
        }

        self.mcrotation(ctls, theta, target, false);
    }

    /// Single qubit Ry gate.
    pub(crate) fn ry(&mut self, theta: f64, target: usize) {
        if let Some(entry) = self.ry_queue.get_mut(&target) {
            *entry += theta;
            if entry.is_nearly_zero() {
                self.ry_queue.remove(&target);
            }
        } else {
            self.ry_queue.insert(target, theta);
        }
    }

    /// Multi-controlled Ry gate.
    pub(crate) fn mcry(&mut self, ctls: &[usize], theta: f64, target: usize) {
        self.flush_queue(ctls, FlushLevel::HRxRy);

        if self.rx_queue.contains_key(&target) {
            self.flush_queue(&[target], FlushLevel::HRx);
        } else if self.h_flag.bit(target as u64) {
            self.flush_queue(&[target], FlushLevel::H);
        }

        self.mcrotation(ctls, theta, target, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn almost_equal(a: f64, b: f64) -> bool {
        a.max(b) - b.min(a) <= 1e-10
    }

    // Test that basic allocation and release of qubits doesn't fail.
    #[test]
    fn test_alloc_release() {
        let sim = &mut QuantumSim::default();
        for i in 0..16 {
            assert_eq!(sim.allocate(), i);
        }
        sim.release(4);
        sim.release(7);
        sim.release(12);
        assert_eq!(sim.allocate(), 4);
        for i in 0..7 {
            sim.release(i);
        }
        for i in 8..12 {
            sim.release(i);
        }
        for i in 13..16 {
            sim.release(i);
        }
    }

    /// Verifies that application of gates to a qubit results in the correct probabilities.
    #[test]
    fn test_probability() {
        let mut sim = QuantumSim::default();
        let q = sim.allocate();
        let extra = sim.allocate();
        assert!(almost_equal(0.0, sim.joint_probability(&[q])));
        sim.x(q);
        assert!(almost_equal(1.0, sim.joint_probability(&[q])));
        sim.x(q);
        assert!(almost_equal(0.0, sim.joint_probability(&[q])));
        sim.h(q);
        assert!(almost_equal(0.5, sim.joint_probability(&[q])));
        sim.h(q);
        assert!(almost_equal(0.0, sim.joint_probability(&[q])));
        sim.x(q);
        sim.h(q);
        sim.s(q);
        assert!(almost_equal(0.5, sim.joint_probability(&[q])));
        sim.sadj(q);
        sim.h(q);
        sim.x(q);
        assert!(almost_equal(0.0, sim.joint_probability(&[q])));
        sim.release(extra);
        sim.release(q);
    }

    /// Verify that a qubit in superposition has probability corresponding the measured value and
    /// can be operationally reset back into the ground state.
    #[test]
    fn test_measure() {
        let mut sim = QuantumSim::default();
        let q = sim.allocate();
        let extra = sim.allocate();
        assert!(!sim.measure(q));
        sim.x(q);
        assert!(sim.measure(q));
        let mut res = false;
        while !res {
            sim.h(q);
            res = sim.measure(q);
            assert!(almost_equal(
                sim.joint_probability(&[q]),
                if res { 1.0 } else { 0.0 }
            ));
            if res {
                sim.x(q);
            }
        }
        assert!(almost_equal(sim.joint_probability(&[q]), 0.0));
        sim.release(extra);
        sim.release(q);
    }

    // Verify that out of order release of non-zero qubits behaves as expected, namely qubits that
    // are not released are still in the expected states, newly allocated qubits use the available spot
    // and start in a zero state.
    #[test]
    fn test_out_of_order_release() {
        let sim = &mut QuantumSim::default();
        for i in 0..5 {
            assert_eq!(sim.allocate(), i);
            sim.x(i);
        }

        // Release out of order.
        sim.release(3);

        // Remaining qubits should all still be in one.
        assert_eq!(sim.state.len(), 1);
        assert!(!sim.joint_probability(&[0]).is_nearly_zero());
        assert!(!sim.joint_probability(&[1]).is_nearly_zero());
        assert!(!sim.joint_probability(&[2]).is_nearly_zero());
        assert!(!sim.joint_probability(&[4]).is_nearly_zero());

        // Cheat and peak at the released location to make sure it has been zeroed out.
        assert!(sim.check_joint_probability(&[3]).is_nearly_zero());

        // Next allocation should be the empty spot, and it should be in zero state.
        assert_eq!(sim.allocate(), 3);
        assert!(sim.joint_probability(&[3]).is_nearly_zero());

        for i in 0..5 {
            sim.release(i);
        }
        assert_eq!(sim.state.len(), 1);
    }

    /// Verify joint probability works as expected, namely that it corresponds to the parity of the
    /// qubits.
    #[test]
    fn test_joint_probability() {
        let mut sim = QuantumSim::default();
        let q0 = sim.allocate();
        let q1 = sim.allocate();
        assert!(almost_equal(0.0, sim.joint_probability(&[q0, q1])));
        sim.x(q0);
        assert!(almost_equal(1.0, sim.joint_probability(&[q0, q1])));
        sim.x(q1);
        assert!(almost_equal(0.0, sim.joint_probability(&[q0, q1])));
        assert!(almost_equal(1.0, sim.joint_probability(&[q0])));
        assert!(almost_equal(1.0, sim.joint_probability(&[q1])));
        sim.h(q0);
        assert!(almost_equal(0.5, sim.joint_probability(&[q0, q1])));
        sim.release(q1);
        sim.release(q0);
    }

    /// Verify joint measurement works as expected, namely that it corresponds to the parity of the
    /// qubits.
    #[test]
    fn test_joint_measurement() {
        let mut sim = QuantumSim::default();
        let q0 = sim.allocate();
        let q1 = sim.allocate();
        assert!(!sim.joint_measure(&[q0, q1]));
        sim.x(q0);
        assert!(sim.joint_measure(&[q0, q1]));
        sim.x(q1);
        assert!(!sim.joint_measure(&[q0, q1]));
        assert!(sim.joint_measure(&[q0]));
        assert!(sim.joint_measure(&[q1]));
        sim.h(q0);
        let res = sim.joint_measure(&[q0, q1]);
        assert!(almost_equal(
            if res { 1.0 } else { 0.0 },
            sim.joint_probability(&[q0, q1])
        ));
        sim.release(q1);
        sim.release(q0);
    }

    /// Test multiple controls.
    #[test]
    fn test_multiple_controls() {
        let mut sim = QuantumSim::default();
        let q0 = sim.allocate();
        let q1 = sim.allocate();
        let q2 = sim.allocate();
        assert!(almost_equal(0.0, sim.joint_probability(&[q0])));
        sim.h(q0);
        assert!(almost_equal(0.5, sim.joint_probability(&[q0])));
        sim.h(q0);
        assert!(almost_equal(0.0, sim.joint_probability(&[q0])));
        sim.mch(&[q1], q0);
        assert!(almost_equal(0.0, sim.joint_probability(&[q0])));
        sim.x(q1);
        sim.mch(&[q1], q0);
        assert!(almost_equal(0.5, sim.joint_probability(&[q0])));
        sim.mch(&[q2, q1], q0);
        assert!(almost_equal(0.5, sim.joint_probability(&[q0])));
        sim.x(q2);
        sim.mch(&[q2, q1], q0);
        assert!(almost_equal(0.0, sim.joint_probability(&[q0])));
        sim.x(q0);
        sim.x(q1);
        sim.release(q2);
        sim.release(q1);
        sim.release(q0);
    }

    /// Verify that targets cannot be duplicated.
    #[test]
    #[should_panic(expected = "Duplicate qubit id '0' found in application.")]
    fn test_duplicate_target() {
        let mut sim = QuantumSim::new();
        let q = sim.allocate();
        sim.mcx(&[q], q);
    }

    /// Verify that controls cannot be duplicated.
    #[test]
    #[should_panic(expected = "Duplicate qubit id '1' found in application.")]
    fn test_duplicate_control() {
        let mut sim = QuantumSim::new();
        let q = sim.allocate();
        let c = sim.allocate();
        sim.mcx(&[c, c], q);
    }

    /// Verify that targets aren't in controls.
    #[test]
    #[should_panic(expected = "Duplicate qubit id '0' found in application.")]
    fn test_target_in_control() {
        let mut sim = QuantumSim::new();
        let q = sim.allocate();
        let c = sim.allocate();
        sim.mcx(&[c, q], q);
    }

    /// Large, entangled state handling.
    #[test]
    fn test_large_state() {
        let mut sim = QuantumSim::new();
        let ctl = sim.allocate();
        sim.h(ctl);
        for _ in 0..4999 {
            let q = sim.allocate();
            sim.mcx(&[ctl], q);
        }
        let _ = sim.measure(ctl);
        for i in 0..5000 {
            sim.release(i);
        }
    }

    /// Verify seeded RNG is predictable.
    #[test]
    fn test_seeded_rng() {
        set_rng_seed(42);
        let mut sim = QuantumSim::new();
        let q = sim.allocate();
        let mut val1 = 0_u64;
        for i in 0..64 {
            sim.h(q);
            if sim.measure(q) {
                val1 += 1 << i;
            }
        }
        set_rng_seed(42);
        let mut sim = QuantumSim::new();
        let q = sim.allocate();
        let mut val2 = 0_u64;
        for i in 0..64 {
            sim.h(q);
            if sim.measure(q) {
                val2 += 1 << i;
            }
        }
        assert_eq!(val1, val2);
    }

    /// Utility for testing operation equivalence.
    fn assert_operation_equal_referenced<F1, F2>(mut op: F1, mut reference: F2, count: usize)
    where
        F1: FnMut(&mut QuantumSim, &[usize]),
        F2: FnMut(&mut QuantumSim, &[usize]),
    {
        enum QueuedOp {
            NoOp,
            H,
            Rx,
            Ry,
        }

        for inner_op in [QueuedOp::NoOp, QueuedOp::H, QueuedOp::Rx, QueuedOp::Ry] {
            let mut sim = QuantumSim::default();

            // Allocte the control we use to verify behavior.
            let ctl = sim.allocate();
            sim.h(ctl);

            // Allocate the requested number of targets, entangling the control with them.
            let mut qs = vec![];
            for _ in 0..count {
                let q = sim.allocate();
                sim.mcx(&[ctl], q);
                qs.push(q);
            }

            // To test queuing, try the op after running each of the different intermediate operationsthat
            // can be queued.
            match inner_op {
                QueuedOp::NoOp => (),
                QueuedOp::H => {
                    for &q in &qs {
                        sim.h(q);
                    }
                }
                QueuedOp::Rx => {
                    for &q in &qs {
                        sim.rx(PI / 7.0, q);
                    }
                }
                QueuedOp::Ry => {
                    for &q in &qs {
                        sim.ry(PI / 7.0, q);
                    }
                }
            }

            op(&mut sim, &qs);

            // Trigger a flush between the op and expected adjoint reference to ensure the reference is
            // run without any queued, commuted operations.
            let _ = sim.joint_probability(&qs);

            reference(&mut sim, &qs);

            // Perform the adjoint of any additional ops. We check the joint probability of the target
            // qubits before and after to force a flush of the operation queue. This helps us verify queuing, as the
            // original operation will have used the queue and commuting while the adjoint perform here will not.
            let _ = sim.joint_probability(&qs);
            match inner_op {
                QueuedOp::NoOp => (),
                QueuedOp::H => {
                    for &q in &qs {
                        sim.h(q);
                    }
                }
                QueuedOp::Rx => {
                    for &q in &qs {
                        sim.rx(PI / -7.0, q);
                    }
                }
                QueuedOp::Ry => {
                    for &q in &qs {
                        sim.ry(PI / -7.0, q);
                    }
                }
            }
            let _ = sim.joint_probability(&qs);

            // Undo the entanglement.
            for q in qs {
                sim.mcx(&[ctl], q);
            }
            sim.h(ctl);

            // We know the operations are equal if the control is left in the zero state.
            assert!(sim.joint_probability(&[ctl]).is_nearly_zero());

            // Sparse state vector should have one entry for |0⟩.
            // Dump the state first to force a flush of any queued operations.
            sim.dump(&mut std::io::stdout());
            assert_eq!(sim.state.len(), 1);
        }
    }

    #[test]
    fn test_h() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.h(qs[0]);
            },
            |sim, qs| {
                sim.h(qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_x() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.x(qs[0]);
            },
            |sim, qs| {
                sim.x(qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_y() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.y(qs[0]);
            },
            |sim, qs| {
                sim.y(qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_z() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.z(qs[0]);
            },
            |sim, qs| {
                sim.z(qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_s() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.s(qs[0]);
            },
            |sim, qs| {
                sim.sadj(qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_sadj() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.sadj(qs[0]);
            },
            |sim, qs| {
                sim.s(qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_cx() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.mcx(&[qs[0]], qs[1]);
            },
            |sim, qs| {
                sim.mcx(&[qs[0]], qs[1]);
            },
            2,
        );
    }

    #[test]
    fn test_cz() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.mcz(&[qs[0]], qs[1]);
            },
            |sim, qs| {
                sim.mcz(&[qs[0]], qs[1]);
            },
            2,
        );
    }

    #[test]
    fn test_swap() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.swap_qubit_ids(qs[0], qs[1]);
            },
            |sim, qs| {
                sim.swap_qubit_ids(qs[0], qs[1]);
            },
            2,
        );
    }

    #[test]
    fn test_rz() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.rz(PI / 7.0, qs[0]);
            },
            |sim, qs| {
                sim.rz(-PI / 7.0, qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_rx() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.rx(PI / 7.0, qs[0]);
            },
            |sim, qs| {
                sim.rx(-PI / 7.0, qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_ry() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.ry(PI / 7.0, qs[0]);
            },
            |sim, qs| {
                sim.ry(-PI / 7.0, qs[0]);
            },
            1,
        );
    }

    #[test]
    fn test_mcri() {
        assert_operation_equal_referenced(
            |sim, qs| {
                sim.mcphase(
                    &qs[2..3],
                    Complex64::exp(Complex64::new(0.0, -(PI / 7.0) / 2.0)),
                    qs[1],
                );
            },
            |sim, qs| {
                sim.mcphase(
                    &qs[2..3],
                    Complex64::exp(Complex64::new(0.0, (PI / 7.0) / 2.0)),
                    qs[1],
                );
            },
            3,
        );
    }
}
