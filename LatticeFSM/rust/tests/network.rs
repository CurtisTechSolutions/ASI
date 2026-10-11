//! Machines connected into networks: the architecture, the wiring, the credit, the file, English in and out.

use latticefsm::experiment::teach_language;
use latticefsm::languages::{language, ALPHABET};
use latticefsm::machine::{Machine, Settings};
use latticefsm::network::{load_network, parse_architecture, teach_network, Credit, Merge, Network, Spec, Vote};
use latticefsm::phonetic::{inventory, plurals, split};
use latticefsm::rng::Rng;

fn ab() -> Vec<String> {
    ALPHABET.iter().map(|s| s.to_string()).collect()
}

fn net(arch: &str, alphabet: &[String], seed: u64) -> Network {
    Network::from_architecture(
        arch,
        alphabet,
        seed,
        Merge::Concat,
        Vote::Majority,
        &Settings::default(),
    )
    .unwrap()
}

#[test]
fn the_architecture_string() {
    let n = net("13>(13|13f)>4", &ab(), 1);
    let shapes: Vec<_> = n.layers().iter().map(|l| l.machine.lattice.shape()).collect();
    assert_eq!(shapes, vec![(13, 2, 13), (13, 13, 13), (13, 13, 13), (4, 26, 4)]);
    let seeds: Vec<u64> = n.layers().iter().map(|l| l.machine.seed).collect();
    assert_eq!(seeds, vec![1, 2, 3, 4]);
    assert!(matches!(
        parse_architecture("(3>3|3)", Merge::Concat, Vote::Majority, &Settings::default()).unwrap(),
        Spec::Parallel { .. }
    ));
    for bad in ["", "13>", "(13|13", "13x", "(13|)", "0", "63"] {
        assert!(
            Network::from_architecture(bad, &ab(), 1, Merge::Concat, Vote::Majority, &Settings::default()).is_err(),
            "{bad}"
        );
    }
    assert!(n.summary().contains("4 layers"));
}

#[test]
fn one_layer_is_the_machine_it_holds() {
    let lang = language("mod3-a").unwrap();
    let mut m = Machine::over(
        3,
        "ab",
        lang.accepting,
        Settings {
            seed: 4,
            ..Settings::default()
        },
    )
    .unwrap();
    teach_language(&mut m, &lang, 600, &mut Rng::new(4), 6, false, 1.0, None, 0);
    let mut n = net("3", &ab(), 4);
    n.compile(Some(lang.accepting), 0.8, 1.0, 1.0, Credit::Shared).unwrap();
    teach_network(&mut n, &lang, 600, &mut Rng::new(4), 6);
    assert_eq!(
        n.layers()[0].machine.lattice.to_json().dump(),
        m.lattice.to_json().dump()
    );
}

#[test]
fn greedy_passes_are_deterministic_and_draw_nothing() {
    let mut n = net("13>13", &ab(), 1);
    let before: Vec<_> = n.layers().iter().map(|l| l.machine.rng.state()).collect();
    let pass = |n: &mut Network| -> Vec<Vec<usize>> {
        ["ab", "bba", ""]
            .iter()
            .map(|s| {
                let symbols = n.tokenize(s).unwrap();
                n.forward(&symbols, true, Some(0.0)).0
            })
            .collect()
    };
    let first = pass(&mut n);
    let again = pass(&mut n);
    assert_eq!(first, again);
    assert_eq!(first[0], vec![0, 0]);
    let after: Vec<_> = n.layers().iter().map(|l| l.machine.rng.state()).collect();
    assert_eq!(before, after);
}

#[test]
fn the_file_round_trips() {
    let lang = language("even-b").unwrap();
    let mut n = net("(3|3)>3", &ab(), 3);
    n.compile(Some(&[0]), 0.6, 1.0, 1.0, Credit::Own).unwrap();
    teach_network(&mut n, &lang, 300, &mut Rng::new(3), 6);
    let path = std::env::temp_dir().join("latticefsm-network-test.json.gz");
    let path = path.to_str().unwrap();
    n.save(path).unwrap();
    let mut back = load_network(path).unwrap();
    assert_eq!(back.to_json().dump(), n.to_json().dump());
    let strings: Vec<String> = ["", "a", "ab", "bab", "abba", "bbbb"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(back.predict(&strings).unwrap(), n.predict(&strings).unwrap());
}

#[test]
fn english_in_and_out() {
    assert_eq!(inventory(false).len(), 43);
    let mut n = net("enc>dec", &ab(), 1);
    assert_eq!(n.tokenize("cats").unwrap().len(), 6);
    assert_eq!(n.decoder().unwrap().machine.lattice.shape(), (43, 43, 43));
    assert_eq!(n.translate("cat").unwrap(), "");
    let (train, test) = split(&plurals(), 4);
    let h = n.fit_pairs(&train, 10, Some(&test)).unwrap();
    assert!(*h.exact.last().unwrap() > 0.6, "{h:?}");
    assert_eq!(n.translate("cat").unwrap(), "cats");
    let (steps, output, _) = n.trace("dog").unwrap();
    assert_eq!(steps[0].english.as_deref(), Some("dog"));
    assert_eq!(output, "dogs");
    for bad in ["13>enc", "(enc|13)", "enc"] {
        assert!(
            Network::from_architecture(bad, &ab(), 1, Merge::Concat, Vote::Majority, &Settings::default()).is_err(),
            "{bad}"
        );
    }
    assert!(net("enc>13", &ab(), 1).translate("cat").is_err());
}
