//! Contract snapshot: the exact value of every constant documented as
//! `Contract: results`, `rounding` or `definition` in the constants modules
//! of the workspace, the values the computed bits depend on. Changing one
//! is never silent: it needs an explicit edit here, in the same commit.
//! Floats are pinned by their bits.

#[test]
fn rslab_contract_constants() {
    use rslab::constants as c;
    // kernels
    assert_eq!(c::GROWTH_EPS.to_bits(), 0x3d06849b86a12b9b);
    assert_eq!(c::PANEL_NB, 64);
    assert_eq!(c::PANEL_NB_MIN, 8);
    assert_eq!(c::SCALAR_GATE, 4096);
    assert_eq!(c::PAR_GEMM_MIN_FLOPS, 1_000_000);
    assert_eq!(c::PAR_CDIV_MIN_FLOPS, 8_000_000);
    assert_eq!(c::FORK_MIN_FLOPS, 100_000_000);
    assert_eq!(c::SCHUR_TILE, 256);
    assert_eq!(c::TRAILING_BLOCK, 16);
    assert_eq!(c::COMPLEX_SPLIT_MIN_RATIO, 64);
    assert_eq!(c::COMPLEX_SPLIT_TILE, 256);
    assert_eq!(c::WIDE_PANEL_MIN_NCOL, 512);
    assert_eq!(c::WIDE_PANEL_NB, 128);
    assert_eq!(c::LU_PANEL_NB, 32);
    assert_eq!(c::CMOD_TILES, 16);
    assert_eq!(c::CMOD_TILE_MIN, 32);
    assert_eq!(c::CMOD_TILE_MAX, 256);
    assert_eq!(c::SPLIT_PLANES, 3);
    // klu
    assert_eq!(c::KLU_PIVOT_THRESHOLD.to_bits(), 0x3f50624dd2f1a9fc);
    // krylov
    assert_eq!(c::KRYLOV_TOL.to_bits(), 0x3e45798ee2308c3a);
    assert_eq!(c::KRYLOV_MAX_ITER, 400);
    assert_eq!(c::KRYLOV_BASIS_BUDGET_BYTES, 1_073_741_824);
    assert_eq!(c::KRYLOV_RESTART_MIN, 20);
    assert_eq!(c::KRYLOV_RESTART_MAX, 80);
    assert_eq!(c::KRYLOV_REORTH_ETA.to_bits(), 0x3fe6a09e667f3bcd);
    assert_eq!(c::KRYLOV_ORTHO_CHUNK, 2048);
    // numeric
    assert_eq!(c::TINY.to_bits(), 0x01a56e1fc2f8f359);
    assert_eq!(c::DENSE_SINGULAR_TOL.to_bits(), 0x3d06849b86a12b9b);
    assert_eq!(c::QR_ITER_PER_DIM, 30);
    assert_eq!(c::INVERSE_ITERATION_SHIFT.to_bits(), 0x3ddb7cdfd9d7bdbb);
    assert_eq!(c::RITZ_REAL_TOL.to_bits(), 0x3e45798ee2308c3a);
    assert_eq!(c::RECYCLE_RANK_TOL.to_bits(), 0x3d719799812dea11);
    // ordering
    assert_eq!(c::COMPRESS_MAX_RATIO.to_bits(), 0x3fee666666666666);
    assert_eq!(c::RACE_ND_MIN_N, 10_000);
    assert_eq!(c::RACE_ND_MIN_WORK, 1_250_000_000);
    assert_eq!(c::RACE_ASSUMED_WORKERS, 4);
    assert_eq!(c::RACE_ENSEMBLE_SIZE, 3);
    assert_eq!(c::RACE_ENSEMBLE_MIN_FLOPS, 50_000_000_000);
    assert_eq!(c::NEMIN, 16);
    assert_eq!(c::RELAX_MIN_N, 1024);
    assert_eq!(c::PATH_LIKE_FRACTION.to_bits(), 0x3fa999999999999a);
    assert_eq!(c::ROOT_CAP_MIN_N, 1024);
    assert_eq!(c::ROOT_CAP_FRACTION.to_bits(), 0x3fa999999999999a);
    assert_eq!(c::ROOT_CAP_MAX, 2048);
    assert_eq!(c::RELAX_MAX_WIDTH, 256);
    assert_eq!(c::RELAX_MAX_EXTRA_ROWS, 64);
    assert_eq!(c::MC21_STEPS_PER_ENTRY, 128);
    assert_eq!(c::MC21_MAX_STEPS, 50_000_000);
    // pivoting
    assert_eq!(c::LU_PIVOT_THRESHOLD.to_bits(), 0x3fb999999999999a);
    assert_eq!(
        c::MATCHING_NEGLIGIBLE_DIAGONAL.to_bits(),
        0x3ddb7cdfd9d7bdbb
    );
    assert_eq!(c::REFINE_MAX_STEPS, 2);
    assert_eq!(c::REFINE_TARGET.to_bits(), 0x3cd0000000000000);
    assert_eq!(
        c::AUTO_SUPERNODAL_FLOPS_PER_ENTRY.to_bits(),
        0x4039000000000000
    );
    assert_eq!(c::AUTO_SYMMETRY_TOL.to_bits(), 0x0000000000000000);
    // scaling
    assert_eq!(c::LOG_HUGE.to_bits(), 0x4086280000000000);
    assert_eq!(c::INFNORM_MAX_ITER, 10);
    assert_eq!(c::INFNORM_TOL.to_bits(), 0x3e45798ee2308c3a);
    assert_eq!(c::HUNGARIAN_DENSE_COL_DIV, 10);
    assert_eq!(c::HUNGARIAN_DENSE_MIN_ROWS, 50);
    // solve
    assert_eq!(c::TILE_ROWS, 4);
    assert_eq!(c::SOLVE_BLOCK, 512);
    assert_eq!(c::SOLVE_ANCESTOR_CHUNK, 32);
    assert_eq!(c::SOLVE_APEX_MIN_ENTRIES, 262_144);
    assert_eq!(c::SOLVE_RHS_TILE, 4);
    assert_eq!(c::FOLD_LANES, 4);
}

#[test]
fn metis_contract_constants() {
    use rslab_metis::constants as c;
    assert_eq!(c::SEED, 1);
    assert_eq!(c::NIPARTS, 7);
    assert_eq!(c::COARSEN_FLOOR, 120);
    assert_eq!(c::ND_TO_AMD_SWITCH, 200);
    assert_eq!(c::TWO_HOP_RATIO_THRESHOLD.to_bits(), 0x3feb333333333333);
    assert_eq!(c::MAX_IMBALANCE.to_bits(), 0x3fc999999999999a);
    assert_eq!(c::FM_PASSES, 10);
    assert_eq!(c::MOVE_LIMIT, 1_048_576);
    assert_eq!(c::MAX_OVERSHOOT.to_bits(), 0x4010000000000000);
    assert_eq!(c::TWO_HOP_MIN_VERTICES, 4);
    assert_eq!(c::COARSEN_STALL_RATIO.to_bits(), 0x3fee666666666666);
    assert_eq!(c::FM_NO_IMPROVE_MAX, 50);
    assert_eq!(c::CHILD_SEED_OFFSET, 7);
    assert_eq!(c::UNBALANCED_SPLIT_FRACTION.to_bits(), 0x3feccccccccccccd);
    assert_eq!(c::LABEL_CLASSES, 3);
}

#[test]
fn ordering_core_contract_constants() {
    use rslab_ordering_core::constants as c;
    assert_eq!(c::DENSE_ALPHA.to_bits(), 0x4024000000000000);
    assert_eq!(c::DENSE_DEGREE_MIN, 16);
    assert_eq!(c::ELBOW_DIV, 5);
    assert_eq!(c::BUCKET_STEP_DIV, 8);
    assert_eq!(c::RCM_PERIPHERAL_MAX_HOPS, 16);
}
