// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

use std::io::BufWriter;

use runner::{run_bitcode, run_file, OUTPUT};

// This group of tests verifies the behavior of QIR execution with a series of quantum gate checks based on the Choi–Jamiołkowski Isomorphism.
// They will verify the behavior of body, adjoint, controlled, and controlled adjoint specializations of each gate against decompositions thereof,
// and additionally makes use of several features from the QIR standard library, namely callables, tuples, arrays, and strings.
// They are based on code from https://github.com/microsoft/qsharp-runtime/tree/main/src/Simulation/TargetDefinitions/Tests.
// Note: these tests are computationally intensive and disabled in debug by default (they will run with `cargo test --release` and in CI).
// To enable, comment out the line below.
#[cfg(not(debug_assertions))]
#[cfg(test)]
mod cji_tests {
    use runner::run_bitcode;
    use std::io::sink;

    #[test]
    fn test_h_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(bitcode, Some("DecompositionTests__VerifyH"), 1, &mut sink())
    }

    #[test]
    fn test_s_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(bitcode, Some("DecompositionTests__VerifyS"), 1, &mut sink())
    }

    #[test]
    fn test_t_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(bitcode, Some("DecompositionTests__VerifyT"), 1, &mut sink())
    }

    #[test]
    fn test_x_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(bitcode, Some("DecompositionTests__VerifyX"), 1, &mut sink())
    }

    #[test]
    fn test_y_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(bitcode, Some("DecompositionTests__VerifyY"), 1, &mut sink())
    }

    #[test]
    fn test_z_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(bitcode, Some("DecompositionTests__VerifyZ"), 1, &mut sink())
    }

    #[test]
    fn test_cnot_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyCNOT"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_cx_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyCX"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_cy_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyCY"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_cz_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyCZ"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_ccnot_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyCCNOT"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_rx_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyRx"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_ry_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyRy"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_rz_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyRz"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_rxx_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyRxx"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_ryy_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyRyy"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_rzz_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyRzz"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_r_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(bitcode, Some("DecompositionTests__VerifyR"), 1, &mut sink())
    }

    #[test]
    fn test_r1_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyR1"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_swap_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifySWAP"),
            1,
            &mut sink(),
        )
    }

    #[test]
    fn test_exp_with_cji() -> Result<(), String> {
        let bitcode = include_bytes!("resources/cji.bc");
        run_bitcode(
            bitcode,
            Some("DecompositionTests__VerifyExp"),
            1,
            &mut sink(),
        )
    }
}

// This test confirms selection of a particular named entry point from a file with two entry points
// works as expected. The function invoked simply returns a hard coded integer, which should get displayed
// to stdout, making use of stdlib functions for int to string and message.
#[test]
fn test_full_qir_simple() -> Result<(), String> {
    let bitcode = include_bytes!("resources/full-qir.bc");
    run_bitcode(
        bitcode,
        Some("QIR_App_Test__Simple"),
        1,
        &mut std::io::sink(),
    )
}

// This test confirms selection of the other named entry point from a file with two entry points
// works as expected. The function invoked uses single qubit allocation, qubit array allocation, callables,
// multi-controlled gates, adjoint gates, and joint measurement with mixed Pauli bases, and acts as
// a broad smoke test for the backend. It returns a measurement result, which should get displayed
// to stdout, making use of backend result to string and stdlib message functionality.
#[test]
fn test_full_qir_other() -> Result<(), String> {
    let bitcode = include_bytes!("resources/full-qir.bc");
    run_bitcode(
        bitcode,
        Some("QIR_App_Test__Other"),
        1,
        &mut std::io::sink(),
    )
}

// This tests a simple Bernstein-Vazirani algorithm looking for the encoded pattern |110⟩ in a qubit array.
// It should achieve the desired state with 100% probability, so the program asserts that the qubits are
// in that state, which will trigger a panic with a message like "Qubit in invalid state. Expecting: One"
// if that state is not achieved.
#[test]
fn test_bernstein_vazirani() -> Result<(), String> {
    let bitcode = include_bytes!("resources/bv.bc");
    run_bitcode(bitcode, None, 1, &mut std::io::sink())
}

// This test runs a sample Shor's algorithm for integer factorization. It makes use of quantum execution
// from backend and BigInt support from stdlib. It prints to stdout the value being factored, progress
// and number of retries followed by a tuple of the identified prime factors.
#[test]
fn test_shor() -> Result<(), String> {
    let bitcode = include_bytes!("resources/shor.bc");
    run_bitcode(bitcode, None, 1, &mut std::io::sink())
}

#[test]
fn run_file_errors_on_invalid_ext() {
    let result = run_file("/some/bad/path", None, 1, None, &mut std::io::sink());
    assert!(result.is_err());
    assert_eq!("Unsupported file extension 'None'.", result.unwrap_err());
}

#[test]
fn run_file_errors_on_missing_file() {
    let result = run_file("/some/bad/path.ll", None, 1, None, &mut std::io::sink());
    assert!(result.is_err());
    assert_eq!(
        "no such file or directory",
        result.unwrap_err().to_lowercase()
    );
}

#[test]
fn run_file_random_seed_is_applied_and_persists_across_shots() {
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        output.use_std_out(false);
    });

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("resources")
        .join("random-bit.bc");

    // running 100 random shots with the same seed should be enough to ensure
    // that the output is deterministic
    let shots = 100;
    let rngseed = 42;

    let mut first_output = vec![];
    let result = run_file(path.clone(), None, shots, Some(rngseed), &mut first_output);
    assert!(result.is_ok());

    let mut second_output = vec![];
    let result = run_file(path, None, shots, Some(rngseed), &mut second_output);
    assert!(result.is_ok());

    let first_result = String::from_utf8(first_output).expect("output should be valid utf8");
    let second_result = String::from_utf8(second_output).expect("output should be valid utf8");

    // sanity check that the output has been captured
    // if the output recording is accidentally set to stdout,
    // this will fail.
    assert!(
        first_result.contains("OUTPUT	RESULT	1"),
        "Ensure global output has use_std_out set to false"
    );

    // the output should be the same for each set of shots
    assert_eq!(first_result, second_result);
}

#[test]
fn run_file_errors_on_missing_binding() {
    let bitcode = include_bytes!("resources/missing-intrinsic.bc");
    let result = run_bitcode(bitcode, None, 1, &mut std::io::sink());
    assert!(result.is_err());
    assert_eq!(
        "failed to link some declared functions: __quantum__qis__mycustomintrinsic__body",
        result.unwrap_err().to_lowercase()
    );
}

#[test]
fn mixed_output_recording_calls_fail() {
    let bitcode = include_bytes!("resources/mixed_output.bc");
    let result = run_bitcode(bitcode, None, 1, &mut std::io::sink());
    assert!(result.is_err());
    assert_eq!(
        "function '__quantum__rt__array_record_output' has mismatched parameters: expected 2, found 1",
        result.unwrap_err().to_lowercase()
    );
}

#[test]
fn barrier_is_noop() {
    OUTPUT.with(|output| {
        let mut output = output.borrow_mut();
        output.use_std_out(false);
    });
    let bitcode = include_bytes!("resources/barrier.bc");
    let mut stream = BufWriter::new(Vec::new());
    let result = run_bitcode(bitcode, None, 1, &mut stream);
    assert!(result.is_ok());
    assert_eq!(String::from_utf8(stream.into_inner().unwrap()).unwrap().replace("\r\n", "\n"), "START\nMETADATA\tentry_point\nMETADATA\toutput_labeling_schema\nMETADATA\tqir_profiles\tbase_profile\nMETADATA\trequired_num_qubits\t1\nMETADATA\trequired_num_results\t1\nOUTPUT\tRESULT\t0\nEND\t0\n");
}
