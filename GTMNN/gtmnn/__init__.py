"""GTMNN -- Game Theory and Micro Neural Networks.

A population of thousands of tiny networks that play a game against each other.
The prediction is the equilibrium; credit is the Shapley value; no gradient ever
crosses a micro boundary.
"""
from gtmnn.activation import SineActivation, sine_activation, sine_partials
from gtmnn.features import Alphabet, FeatureHasher, coverage, coverage_gaps
from gtmnn.micro import MicroPool, ABSTAIN
from gtmnn.payoff import (CorrectnessCongestion, BeliefCongestion, Inverted,
                          PrisonersDilemma, StagHunt, RockPaperScissors)
from gtmnn.auction import SeatAllocation, allocate, bid, charge
from gtmnn.game import StageGame, GameContext, build, aggregate
from gtmnn.equilibrium import Equilibrium, CycleDetector, solve
from gtmnn.shapley import ShapleyResult, shapley_values, shapley_exact, banzhaf_values
from gtmnn.backend import get_backend
from gtmnn.model import GTMNet, TrainConfig, new_model, load_model
from gtmnn.evolve import Evolver, EvolveConfig
from gtmnn.checkpoint import CheckpointManager

__version__ = "0.1.0"
__all__ = ["GTMNet", "TrainConfig", "MicroPool", "FeatureHasher", "Alphabet",
           "StageGame", "GameContext", "solve", "shapley_values", "Evolver",
           "EvolveConfig", "CheckpointManager", "get_backend", "__version__"]
