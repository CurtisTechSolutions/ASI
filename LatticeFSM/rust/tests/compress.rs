//! The matrix folded into its central node, rebuilt from the middle outward, and compressed on a clock.

use latticefsm::compress::{
    base64_decode, base64_encode, compress, expansion, f16_to_f64, f64_to_f16, fidelity, load_core, Core, Precision,
};
use latticefsm::experiment::teach_language;
use latticefsm::geometry::{center, center_out, shell, shell_sizes, shells};
use latticefsm::json::{parse, Json};
use latticefsm::languages::language;
use latticefsm::machine::{Machine, Settings};
use latticefsm::rng::Rng;

fn taught(episodes: usize, settings: Settings) -> Machine {
    let lang = language("even-b").unwrap();
    let mut m = Machine::default_shape(lang.accepting, settings).unwrap();
    teach_language(&mut m, &lang, episodes, &mut Rng::new(2), 6, false, 1.0, None, 0);
    m
}

fn edges(m: &Machine) -> Vec<String> {
    m.lattice.edges.iter().map(|e| e.to_json().dump()).collect()
}

#[test]
fn the_cube_has_a_centre_and_seven_shells() {
    let shape = (13, 13, 13);
    assert_eq!(center(shape), (6, 6, 6));
    assert_eq!(shells(shape), 7);
    assert_eq!(shell_sizes(shape), vec![1, 26, 98, 218, 386, 602, 866]);
    assert_eq!(shell(shape, 0, 6, 6), 6);
    let order = center_out(shape);
    assert_eq!(order[0], (6 * 13 + 6) * 13 + 6);
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(sorted, (0..2197).collect::<Vec<_>>());
    let rings: Vec<usize> = order
        .iter()
        .map(|&c| shell(shape, c / 169, (c / 13) % 13, c % 13))
        .collect();
    assert!(rings.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn exact_is_lossless_bit_for_bit() {
    let m = taught(1500, Settings::default());
    let core = compress(&m, None, None);
    assert_eq!(core.precision, Precision::Exact);
    assert_eq!(core.center(), (6, 6, 6));
    let r = core.decompress(None).unwrap();
    assert_eq!(edges(&r), edges(&m));
    let f = fidelity(&m, &r);
    assert!(f.bool_or("lossless", false));
    assert_eq!(f.num("max_kl", 1.0), 0.0);
    assert_eq!(
        (r.clock, r.state, r.runs, r.credits),
        (m.clock, m.state, m.runs, m.credits)
    );
    assert_eq!(r.stimulation(), m.stimulation());
    let touched = m.lattice.touched().count();
    assert_eq!(core.n_touched(), touched);
    assert_eq!(
        core.bytes(),
        2197usize.div_ceil(8) + touched * (6 * core.int_width + 15 * 8)
    );
    let fresh = Machine::default_shape(&[0], Settings::default()).unwrap();
    assert_eq!(compress(&fresh, None, None).bytes(), 2197usize.div_ceil(8));
}

#[test]
fn smaller_precisions_keep_the_behaviour_and_a_budget_picks_the_least_lossy() {
    let m = taught(1500, Settings::default());
    let sizes: Vec<usize> = Precision::ALL
        .iter()
        .map(|&p| compress(&m, Some(p), None).bytes())
        .collect();
    assert!(sizes[0] > sizes[1] && sizes[1] > sizes[2]);
    for p in [Precision::Float32, Precision::Float16] {
        let f = fidelity(&m, &compress(&m, Some(p), None).decompress(None).unwrap());
        assert!(!f.bool_or("lossless", true));
        assert!(f.bool_or("preserved", false), "{p:?}: {}", f.dump());
    }
    assert_eq!(compress(&m, None, Some(sizes[0])).precision, Precision::Exact);
    assert_eq!(compress(&m, None, Some(sizes[0] - 1)).precision, Precision::Float32);
    assert_eq!(compress(&m, None, Some(sizes[1] - 1)).precision, Precision::Float16);
    assert_eq!(compress(&m, None, Some(1)).precision, Precision::Float16);
    assert!(Precision::parse("float8").is_err());
}

#[test]
fn rebuilding_from_the_middle_outward() {
    let m = taught(1500, Settings::default());
    let core = compress(&m, None, None);
    let sizes = shell_sizes(core.shape);
    let order = center_out(core.shape);
    for k in 0..=sizes.len() {
        let part = core.decompress(Some(k)).unwrap();
        let inside: std::collections::HashSet<usize> =
            order[..sizes[..k].iter().sum::<usize>()].iter().copied().collect();
        for (cell, (eo, ep)) in m.lattice.edges.iter().zip(part.lattice.edges.iter()).enumerate() {
            if inside.contains(&cell) {
                assert_eq!(eo, ep);
            } else {
                assert!(!ep.touched());
            }
        }
    }
    let exp = expansion(&m, &core).unwrap();
    let rows = exp.as_array().unwrap();
    assert_eq!(rows.len(), 7);
    assert_eq!(rows[6].num("greedy_changed", 1.0), 0.0);
    assert_eq!(rows[6].num("touched", 0.0), core.n_touched() as f64);
}

#[test]
fn walking_straight_from_the_code() {
    let mut m = taught(1500, Settings::default());
    let core = compress(&m, None, None);
    for text in ["", "a", "abba", "bbab", "aaaa"] {
        let symbols = m.tokenize(text).unwrap();
        let start = m.run(&symbols, None, Some(0.0), true);
        assert_eq!(core.run(text, false).unwrap(), (start.states(), start.accepted));
        let middle = m.run_from_middle(&symbols, None, Some(0.0), true);
        assert_eq!(core.run(text, true).unwrap(), (middle.states(), middle.accepted));
        assert_eq!(middle.states()[0], 6);
    }
    for s in [0, 6, 12] {
        assert_eq!(
            core.probabilities(s, 0, None, None).unwrap(),
            m.probabilities(s, 0, None, None)
        );
    }
}

#[test]
fn the_code_round_trips_through_its_file() {
    let m = taught(800, Settings::default());
    let dir = std::env::temp_dir().join(format!("latticefsm-core-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for p in Precision::ALL {
        let core = compress(&m, Some(p), None);
        let path = dir
            .join(format!("core-{}.json.gz", p.name()))
            .to_string_lossy()
            .to_string();
        core.save(&path).unwrap();
        let back = load_core(&path).unwrap();
        assert_eq!(
            (&back.ints, &back.floats, &back.touched),
            (&core.ints, &core.floats, &core.touched)
        );
        assert_eq!(
            edges(&back.decompress(None).unwrap()),
            edges(&core.decompress(None).unwrap())
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
    let mut bad = compress(&m, None, None).to_json();
    bad.set("floats", "AAAA".into());
    assert!(Core::from_json(&bad).is_err());
    assert!(Core::from_json(&parse(r#"{"format":"nope"}"#).unwrap()).is_err());
}

#[test]
fn half_floats_and_base64_are_written_out_right() {
    for (x, bits) in [
        (0.0, 0x0000u16),
        (1.0, 0x3c00),
        (-2.0, 0xc000),
        (65504.0, 0x7bff),
        (6.103515625e-05, 0x0400),
    ] {
        assert_eq!(f64_to_f16(x), bits, "{x}");
        assert_eq!(f16_to_f64(bits), x);
    }
    assert_eq!(f64_to_f16(1.0 + 1.0 / 2048.0), 0x3c00); // a tie rounds to even
    assert_eq!(f64_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
    assert_eq!(f64_to_f16(5.960464477539063e-08), 0x0001); // the smallest subnormal
    for h in (0u16..0x7c00).step_by(37) {
        assert_eq!(f64_to_f16(f16_to_f64(h)), h);
    }
    for data in [&b""[..], b"f", b"fo", b"foo", b"foob", b"\x00\xff\x10"] {
        assert_eq!(base64_decode(&base64_encode(data)).unwrap(), data);
    }
    assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
}

#[test]
fn the_matrix_is_compressed_every_n_transitions() {
    let mut m = Machine::default_shape(
        &[0],
        Settings {
            compress_every: 10,
            ..Settings::default()
        },
    )
    .unwrap();
    let s = m.tokenize(&"ab".repeat(12)).unwrap();
    m.run(&s, None, None, false);
    assert_eq!((m.compressions, m.since_compression, m.last_compressed), (2, 4, 20));
    assert!(m.core.is_some());
    m.teach(0, 0, 1, 1.0).unwrap();
    assert_eq!(m.since_compression, 5);
    let q = m.tokenize("aaaaa").unwrap();
    m.run(&q, None, None, true);
    assert_eq!(m.since_compression, 5);
    let stats = m.stats();
    assert_eq!(stats.num("compressions", 0.0), 2.0);
    assert_eq!(stats.get("center").unwrap().dump(), "[6,6,6]");
    let back = Machine::from_json(&parse(&m.to_json().dump()).unwrap()).unwrap();
    assert_eq!(
        (back.compress_every, back.compressions, back.since_compression),
        (10, 2, 5)
    );
}

#[test]
fn an_exact_rebuild_changes_nothing_and_a_lossy_one_is_applied() {
    let plain = taught(800, Settings::default());
    let exact = taught(
        800,
        Settings {
            compress_every: 50,
            compress_rebuild: true,
            ..Settings::default()
        },
    );
    assert_eq!(edges(&exact), edges(&plain));
    assert!(exact.compressions > 10);
    let mut lossy = Machine::default_shape(
        &[0],
        Settings {
            compress_every: 1,
            compress_precision: Precision::Float16,
            compress_rebuild: true,
            ..Settings::default()
        },
    )
    .unwrap();
    lossy.teach(0, 0, 1, 0.1234567).unwrap();
    let r = lossy.edge(0, 0, 1).rewarded;
    assert!(r != 0.1234567 && (r - 0.1234567).abs() < 1e-3, "{r}");
    let mut path_kept = Machine::default_shape(
        &[0],
        Settings {
            compress_every: 2,
            compress_precision: Precision::Float16,
            compress_rebuild: true,
            ..Settings::default()
        },
    )
    .unwrap();
    let s = path_kept.tokenize("abab").unwrap();
    path_kept.run(&s, None, None, false);
    assert_eq!(path_kept.reward(1.0), 4);
    assert!(path_kept.lattice.edges.iter().map(|e| e.rewarded).sum::<f64>() > 0.0);
    let _ = Json::Null;
}
