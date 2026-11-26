mod common;

use whale_mix_engine::bundle::{load, BundleError};

#[test]
fn qualified_selected_bundle_loads() {
    let dir = tempfile::tempdir().unwrap();
    common::write_bundle(dir.path(), "selected", "qualified", b"not-a-real-model");
    let b = load(dir.path(), false).expect("qualified bundle should load");
    assert_eq!(b.manifest.label_map.outputs.len(), 11);
    assert!(b.known_profile("auto") && b.known_profile("blue_whale"));
    assert!(!b.known_profile("no_such_profile"));
}

#[test]
fn unqualified_and_fixture_bundles_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    common::write_bundle(dir.path(), "selected", "unqualified", b"x");
    assert!(matches!(
        load(dir.path(), false),
        Err(BundleError::NotQualified(_))
    ));
    let dir = tempfile::tempdir().unwrap();
    common::write_bundle(
        dir.path(),
        "fixture-untrained",
        "not-applicable-fixture",
        b"x",
    );
    assert_eq!(
        load(dir.path(), false).unwrap_err(),
        BundleError::FixtureRefused
    );
    assert!(load(dir.path(), true).is_ok());
}

#[test]
fn hash_mismatch_missing_and_bad_schema_are_errors() {
    let dir = tempfile::tempdir().unwrap();
    common::write_bundle(dir.path(), "selected", "qualified", b"original");
    std::fs::write(dir.path().join("model.onnx"), b"tampered").unwrap();
    assert_eq!(
        load(dir.path(), false).unwrap_err(),
        BundleError::HashMismatch
    );

    let empty = tempfile::tempdir().unwrap();
    assert_eq!(load(empty.path(), false).unwrap_err().code(), "missing");

    let dir = tempfile::tempdir().unwrap();
    let mut m = common::write_bundle(dir.path(), "selected", "qualified", b"x");
    m["schema"] = "whalemix-model-bundle/0".into();
    std::fs::write(dir.path().join("bundle.json"), m.to_string()).unwrap();
    assert_eq!(load(dir.path(), false).unwrap_err().code(), "invalid");

    let dir = tempfile::tempdir().unwrap();
    let mut m = common::write_bundle(dir.path(), "selected", "qualified", b"x");
    m["model"]["input"]["shape"][3] = 99.into();
    std::fs::write(dir.path().join("bundle.json"), m.to_string()).unwrap();
    assert_eq!(load(dir.path(), false).unwrap_err().code(), "invalid");
}

#[test]
fn bandwidth_limits_assessable_outputs() {
    let dir = tempfile::tempdir().unwrap();
    common::write_bundle(dir.path(), "selected", "qualified", b"x");
    let b = load(dir.path(), false).unwrap();
    let auto = vec!["auto".to_string()];
    let wide = b.assessable_outputs(&auto, 8000);
    let narrow = b.assessable_outputs(&auto, 200);
    assert!(wide.len() >= narrow.len());
    assert!(wide.contains(&"humpback_song".to_string()));
    assert!(!narrow.contains(&"humpback_song".to_string()));
}
