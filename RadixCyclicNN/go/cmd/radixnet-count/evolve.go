package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// cmdEvolve runs the self-upgrade loop: the model is the generator, a second
// network is the discriminator, and each generation the critic learns real from
// fake while the generator learns from what the critic rejected.
func cmdEvolve(args []string) {
	var data multiFlag
	cfg := radixnet.DefaultEvolveConfig()
	fs := flag.NewFlagSet("evolve", flag.ExitOnError)
	fs.Var(&data, "data", "real corpus, one text per line (repeatable)")
	generations := fs.Int("generations", 0, "generations to run (0 = until interrupted)")
	samples := fs.Int("samples", cfg.Samples, "fakes generated per generation")
	realPer := fs.Int("real-per-generation", cfg.RealPerGeneration, "real texts sampled per generation")
	maxLength := fs.Int("max-length", cfg.MaxLength, "characters per fake")
	temperature := fs.Float64("temperature", cfg.Temperature, "sampling temperature of the fakes")
	discPath := fs.String("discriminator", "", "discriminator model file (default: beside --model)")
	negEpochs := fs.Int("neg-epochs", cfg.NegEpochs, "generator negative-phase epochs")
	posEpochs := fs.Int("pos-epochs", cfg.PosEpochs, "generator positive-phase epochs")
	discNeg := fs.Int("disc-neg-epochs", cfg.DiscNegEpochs, "discriminator negative-phase epochs")
	discPos := fs.Int("disc-pos-epochs", cfg.DiscPosEpochs, "discriminator positive-phase epochs")
	strength := fs.Float64("strength", cfg.Strength, "reward / penalty per pass")
	mode := fs.String("blatant-mode", cfg.BlatantMode,
		"how failed fakes drive the update: none | fail_invert | activation | state")
	margin := fs.Float64("blatant-margin", cfg.BlatantMargin, "per-char gap at which a failure is blatant")
	boost := fs.Float64("blatant-boost", cfg.BlatantBoost, "cap on a failure's penalty multiplier")
	blame := fs.Bool("blame", false, "the discriminator also teaches the negative network what it rejected")
	out := fs.String("out", "", "where to save the generator (default: --model)")
	addNegativeFlag(fs)
	_ = fs.Parse(args)

	if len(data) == 0 {
		fail("-data is required: the real corpus the critic learns from")
	}
	cfg.Samples, cfg.RealPerGeneration, cfg.MaxLength, cfg.Temperature = *samples, *realPer, *maxLength, *temperature
	cfg.NegEpochs, cfg.PosEpochs, cfg.DiscNegEpochs, cfg.DiscPosEpochs = *negEpochs, *posEpochs, *discNeg, *discPos
	cfg.Strength, cfg.BlatantMode, cfg.BlatantMargin, cfg.BlatantBoost = *strength, *mode, *margin, *boost
	cfg.Seed = seedFlag
	corpus := readTexts(data, "lines", 0)
	generator := openModel(false)
	if err := cfg.Validate(generator.Encoding()); err != nil {
		fail("%v", err)
	}
	discriminator, discFile := openDiscriminator(*discPath)
	loop, err := radixnet.NewEvolver(generator, corpus, discriminator, cfg)
	if err != nil {
		fail("%v", err)
	}
	var negative *radixnet.Model
	if *blame {
		negative = openNegative(false)
		loop.Negative = negative
	}
	generationsText := fmt.Sprintf("%d", *generations)
	if *generations == 0 {
		generationsText = "until interrupted"
	}
	say("generator      %s", modelPath)
	say("discriminator  %s", discFile)
	say("corpus         %d text(s)", len(corpus))
	say("generations    %s x %d fake(s), %d real", generationsText, cfg.Samples, cfg.RealPerGeneration)
	say("failures       %s (margin %g, boost %g)", cfg.BlatantMode, cfg.BlatantMargin, cfg.BlatantBoost)
	if negative != nil {
		say("negative       %s (the critic blames every fake it scores below the real texts)", negativeFile())
	}
	say("")
	loop.Stop = interruptible()
	if !jsonMode {
		say("%4s %11s %11s %8s %6s %8s %7s  %s", "gen", "fake", "real", "gap", "fakes", "failures", "2nrl", "sample")
		loop.Progress = func(record map[string]any) {
			say("%4v %11s %11s %8s %6v %8v %7v  %s", record["generation"],
				fmtMark(record["fake_score_mean"]), fmtMark(record["real_score_mean"]), fmtMark(record["gap"]),
				record["fakes"], record["failures"], record["twonrl"], clip(fmt.Sprint(record["sample"]), 32))
		}
	}
	records, err := loop.Run(*generations)
	if err != nil {
		fail("%v", err)
	}
	target := *out
	if target == "" {
		target = modelPath
	}
	if err := generator.Save(target); err != nil {
		fail("cannot save %s: %v", target, err)
	}
	// loop.Discriminator, not the local: a missing file means NewEvolver made a fresh one
	if err := loop.Discriminator.Save(discFile); err != nil {
		fail("cannot save %s: %v", discFile, err)
	}
	say("")
	say("%d generation(s); saved %s and %s", len(records), target, discFile)
	doc := map[string]any{
		"config": cfg, "generations": len(records), "records": records, "saved": target,
		"discriminator": discFile, "stats": generator.Stats(), "negative": nil,
	}
	if negative != nil {
		saved := saveNegative(negative)
		say("negative model: %s", saved)
		reasonTable(negative, 10)
		doc["negative"] = map[string]any{
			"path": saved, "reasons": negative.Reasons(), "stats": negative.Stats(),
		}
	}
	if jsonMode {
		emit(doc)
	}
}

// openDiscriminator loads the critic beside the model (model.count.json ->
// discriminator.json in the same directory), else creates a fresh one.
func openDiscriminator(given string) (*radixnet.Model, string) {
	path := given
	if strings.TrimSpace(path) == "" {
		dir := "."
		if at := strings.LastIndexAny(modelPath, "/\\"); at >= 0 {
			dir = modelPath[:at]
		}
		path = dir + "/discriminator.json"
		if dir == "." {
			path = "discriminator.json"
		}
	}
	if _, err := os.Stat(path); err == nil {
		m, err := radixnet.Load(path)
		if err != nil {
			fail("%s: %v", path, err)
		}
		return m, path
	}
	return nil, path // NewEvolver makes a fresh one, seeded beside the generator's
}
