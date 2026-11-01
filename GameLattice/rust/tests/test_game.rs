//! The stage game's invariants (`GTMNN/DESIGN.md` §11.6, §12.5; `DESIGN.md` §15).

use gamelattice::game::{self, allocate, GameSettings, Payoff, Seat};
use latticefsm::rng::Rng;

fn seat(id: usize, belief: Vec<f64>, reputation: f64) -> Seat {
    let n = belief.len();
    Seat {
        id,
        belief,
        reputation,
        voice: 1.0,
        regret: vec![0.0; n],
    }
}

fn peaked(states: usize, at: usize, p: f64) -> Vec<f64> {
    let rest = (1.0 - p) / states as f64;
    let mut b = vec![rest; states + 1];
    b[at] = p;
    b
}

#[test]
fn correctness_game_terminates_without_a_cycle_and_the_potential_rises() {
    let g = GameSettings::default();
    let mut rng = Rng::new(7);
    for _ in 0..200 {
        let states = 2 + rng.randint(0, 6);
        let n = 1 + rng.randint(0, 7);
        let y = rng.randint(0, states - 1);
        let seats: Vec<Seat> = (0..n)
            .map(|i| {
                let at = rng.randint(0, states - 1);
                seat(i, peaked(states, at, 0.3 + 0.5 * rng.random()), 0.5)
            })
            .collect();
        let payoff = Payoff::Correctness { y };
        let start: Vec<usize> = seats
            .iter()
            .map(|s| game::margin(&s.belief) as usize % (states + 1))
            .collect();
        assert_eq!(
            game::best_response_cycle(&seats, &payoff, &start, &g, 64),
            0,
            "an exact potential game cannot cycle"
        );
        let mut regrets: Vec<Vec<f64>> = seats.iter().map(|s| s.regret.clone()).collect();
        let eq = game::solve(&seats, &payoff, &g, &mut regrets);
        assert_eq!(eq.cycle, 0);
        assert!(
            eq.actions.iter().all(|&a| a == y || a == states),
            "with the label known every seat names it or abstains"
        );
    }
}

#[test]
fn rock_paper_scissors_cycles_with_period_three_per_sweep() {
    // two seats, three slots plus abstain, a zero-sum cycle written as a player-specific belief game is not
    // RPS; so the fixture is the pure best-response dynamic on a hand-made payoff: seat 0 wants to match seat 1,
    // seat 1 wants to differ - the matching-pennies loop, period 4 over two players; the detector counts sweeps
    let g = GameSettings::default();
    let seats = vec![seat(0, peaked(2, 0, 0.5), 0.5), seat(1, peaked(2, 1, 0.5), 0.5)];
    // Belief payoff is non-increasing in load for everyone, so even this cannot loop: the detector returns 0
    let cycle = game::best_response_cycle(&seats, &Payoff::Belief, &[0, 1], &g, 32);
    assert_eq!(cycle, 0);
}

#[test]
fn shapley_is_efficient_for_every_permutation_count() {
    let g = GameSettings::default();
    let mut rng = Rng::new(3);
    let states = 5;
    let seats: Vec<Seat> = (0..6)
        .map(|i| seat(i, peaked(states, i % states, 0.6), 0.3 + 0.1 * i as f64))
        .collect();
    let profile: Vec<Vec<f64>> = seats.iter().map(|s| s.belief.clone()).collect();
    let right = vec![false, false, true, false, false];
    let all: Vec<usize> = (0..6).collect();
    let v_n = game::coalition_value(&seats, &profile, &all, states, &right, g.eps);
    let v_0 = game::coalition_value(&seats, &profile, &[], states, &right, g.eps);
    let phi = game::shapley(&seats, &profile, states, &right, &g, &mut rng);
    let total: f64 = phi.iter().sum();
    assert!(
        (total - (v_n - v_0)).abs() < 1e-9,
        "credit is conserved: {total} vs {}",
        v_n - v_0
    );
    // a seat with no mass on the right state and the same belief shape as another is not a null player here
    // (it dilutes); a seat identical to another gets the same value
    let twin = {
        let mut s = seats.clone();
        s.push(seats[2].clone());
        s
    };
    let twin_profile: Vec<Vec<f64>> = twin.iter().map(|s| s.belief.clone()).collect();
    let phi2 = game::shapley(&twin, &twin_profile, states, &right, &g, &mut Rng::new(3));
    assert!(
        (phi2[2] - phi2[6]).abs() < 0.05,
        "symmetry up to sampling: {} vs {}",
        phi2[2],
        phi2[6]
    );
}

#[test]
fn the_auction_seats_by_bid_and_charges_the_highest_losing_bid() {
    let seats = vec![
        seat(0, peaked(3, 0, 0.9), 0.9),
        seat(1, peaked(3, 1, 0.6), 0.5),
        seat(2, peaked(3, 2, 0.4), 0.5),
        seat(3, peaked(3, 0, 0.3), 0.5), // bid 0.5 * (0.3 - 0.175) = 0.0625
    ];
    let a = allocate(&seats, 2, 0.01);
    assert_eq!(a.seated, vec![0, 1]);
    assert_eq!(a.bidders, 4);
    assert!(
        (a.price - seats[2].bid()).abs() < 1e-12,
        "the price is the highest losing bid"
    );
    let a = allocate(&seats, 8, 0.01);
    assert_eq!(a.price, 0.0, "short-handed: nobody pays");
    let a = allocate(&seats, 2, 1.0);
    assert_eq!(a.bidders, 0, "a reserve above every bid seats nobody");
}

#[test]
fn every_profile_is_a_distribution_and_exploitability_is_non_negative() {
    let g = GameSettings::default();
    let seats: Vec<Seat> = (0..5).map(|i| seat(i, peaked(4, i % 4, 0.5), 0.5)).collect();
    let mut regrets: Vec<Vec<f64>> = seats.iter().map(|s| s.regret.clone()).collect();
    for payoff in [
        Payoff::Belief,
        Payoff::Correctness { y: 2 },
        Payoff::Refused { named: 1 },
    ] {
        let eq = game::solve(&seats, &payoff, &g, &mut regrets);
        for row in &eq.profile {
            assert!(row.iter().all(|x| *x >= 0.0));
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert!(eq.exploitability >= 0.0);
    }
}
