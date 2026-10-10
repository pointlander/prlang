// Copyright 2026 The prlang Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

func TestTokenize(t *testing.T) {
	got := tokenize("To be, or not to be")
	want := []string{"To", "be,", "or", "not", "to", "be"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("tokenize = %#v", got)
	}
	got = tokenize("know’st o’er ’tis well-known")
	want = []string{"know’st", "o’er", "’tis", "well-known"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("tokenize = %#v", got)
	}
	got = tokenize("A\r\na  b.")
	want = []string{"A\r\n", "a", "b."}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("tokenize = %#v", got)
	}
	got = tokenize("end\r\n\r\nNext")
	want = []string{"end\r\n", "\r\n", "Next"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("tokenize = %#v", got)
	}
	if joinWords(got) != "end\r\n\r\nNext" {
		t.Fatalf("join = %q", joinWords(got))
	}
}

func TestStripGutenberg(t *testing.T) {
	in := "header\n*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\nHello\n*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\nlicense"
	if got := strings.TrimSpace(stripGutenberg(in)); got != "Hello" {
		t.Fatalf("strip = %q", got)
	}
	if got := stripGutenberg("plain text"); got != "plain text" {
		t.Fatalf("strip plain = %q", got)
	}
}

func TestOccurrenceWeights(t *testing.T) {
	g := buildGraph(tokenize("a b a b a c"), []string{"a"})
	w := func(from, to string) float64 {
		fid := findCorpus(t, g, from)
		tid := findCorpus(t, g, to)
		for _, e := range g.adj[fid] {
			if e.to == tid {
				return e.w
			}
		}
		t.Fatalf("missing link %s→%s", from, to)
		return 0
	}
	if math.Abs(w("a", "b")-math.Log(2)) > 1e-9 || math.Abs(w("b", "a")-math.Log(2)) > 1e-9 || w("a", "c") != 0 {
		t.Fatalf("weights a→b=%v b→a=%v a→c=%v", w("a", "b"), w("b", "a"), w("a", "c"))
	}
	if g.heavy <= w("a", "b") {
		t.Fatalf("heavy %v should exceed corpus occurrence %v", g.heavy, w("a", "b"))
	}
}

func TestSentencesAreWords(t *testing.T) {
	g := buildGraph(tokenize("Go now. Stop. Go now."), []string{"Go"})
	if g.sentences != 1 || g.corpus != 4 {
		t.Fatalf("sentences=%d corpus=%d", g.sentences, g.corpus)
	}
	now, sent, stop := findCorpus(t, g, "now."), findCorpus(t, g, "Go now."), findCorpus(t, g, "Stop.")
	w := func(from, to uint32) float64 {
		for _, e := range g.adj[from] {
			if e.to == to {
				return e.w
			}
		}
		t.Fatalf("missing link %d→%d", from, to)
		return 0
	}
	if math.Abs(w(now, sent)-math.Log1p(2)) > 1e-9 {
		t.Fatalf("now.→sentence = %v", w(now, sent))
	}
	if math.Abs(w(sent, stop)-math.Log1p(1)) > 1e-9 {
		t.Fatalf("sentence→Stop. = %v", w(sent, stop))
	}
	if math.Abs(w(stop, sent)-math.Log1p(1)) > 1e-9 {
		t.Fatalf("Stop.→sentence = %v", w(stop, sent))
	}
	// The word bigram is unchanged by the sentence link.
	goID := findCorpus(t, g, "Go")
	if math.Abs(w(goID, now)-math.Log(2)) > 1e-9 {
		t.Fatalf("Go→now. = %v", w(goID, now))
	}

	line := buildGraph(tokenize("A\r\nman died.\r\nNext."), []string{"A\r\n"})
	findCorpus(t, line, "A\r\nman died.\r\n")
}

func TestWalkTakesSentence(t *testing.T) {
	// "xx." follows "aa" once, so that word edge has log(1) = 0. The only
	// positive edge is the sentence "aa xx.", which resumes at "bb."
	g := buildGraph(tokenize("aa xx. bb."), []string{"xx."})
	rank, _ := pageRank(g.adj, g.personal, alpha, rankEps)
	got := walk(rand.New(rand.NewSource(1)), g, rank, 2)
	want := []string{"xx.", "aa xx.", "bb."}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("walk = %#v, want %#v", got, want)
	}
	if g.resume[findCorpus(t, g, "aa xx.")] != findCorpus(t, g, "bb.") {
		t.Fatalf("resume = %q", g.label[g.resume[findCorpus(t, g, "aa xx.")]])
	}
}

func TestPromptLinkedHeavy(t *testing.T) {
	text := strings.Repeat("alpha one two beta three four ", 30)
	prompt := []string{"alpha", "beta"}
	g := buildGraph(tokenize(text), prompt)
	if int(g.start) != g.corpus {
		t.Fatalf("prompt path starts at %d, corpus nodes %d", g.start, g.corpus)
	}
	// First prompt node: chain to the next prompt node, side link into the corpus.
	var chainW, sideW float64
	for _, e := range g.adj[g.start] {
		switch e.to {
		case g.start + 1:
			chainW = e.w
		default:
			if g.label[e.to] == "alpha" && e.to < g.start {
				sideW = e.w
			}
		}
	}
	if chainW != g.chain || sideW != g.heavy {
		t.Fatalf("chain=%v side=%v heavy=%v chainField=%v", chainW, sideW, g.heavy, g.chain)
	}
	if !(chainW > sideW && sideW > 1) {
		t.Fatalf("prompt links are not heavy: chain=%v side=%v", chainW, sideW)
	}
}

func TestReuseLowersProbability(t *testing.T) {
	edges := []edge{{to: 0, w: 1}, {to: 1, w: 1}}
	rank := []float64{1, 1}
	labels := []string{"used", "fresh"}
	used := map[string]int{"used": 3}
	rng := rand.New(rand.NewSource(1))
	fresh := 0
	const n = 400
	for range n {
		if labels[step(rng, edges, rank, labels, used, nil, nil)] == "fresh" {
			fresh++
		}
	}
	// Weights are equal except the used word is at 1/4, so fresh is chosen
	// with probability 0.8.
	if fresh < int(0.7*n) || fresh > int(0.9*n) {
		t.Fatalf("fresh picks %d/%d", fresh, n)
	}
}

func TestPageRankBiasAndWalk(t *testing.T) {
	g := buildGraph(tokenize(strings.Repeat("aa bb aa cc ", 40)), []string{"bb"})
	rank, _ := pageRank(g.adj, g.personal, alpha, rankEps)
	sum := 0.0
	for _, r := range rank {
		sum += r
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("rank sums to %v", sum)
	}
	bb, cc := findCorpus(t, g, "bb"), findCorpus(t, g, "cc")
	if rank[bb] <= rank[cc] {
		t.Fatalf("prompt word bb rank %v should exceed cc rank %v", rank[bb], rank[cc])
	}

	text := strings.Repeat("alpha\r\nbeta three\r\n", 30)
	g = buildGraph(tokenize(text), []string{"alpha\r\n", "beta"})
	rank, _ = pageRank(g.adj, g.personal, alpha, rankEps)
	rng := rand.New(rand.NewSource(1))
	got := walk(rng, g, rank, 3)
	want := []string{"alpha\r\n", "beta", "three\r\n", "alpha\r\n", "beta"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("walk = %#v, want %#v", got, want)
	}
}

func findCorpus(t *testing.T, g graph, word string) uint32 {
	t.Helper()
	for i := 0; i < g.corpus; i++ {
		if g.label[i] == word {
			return uint32(i)
		}
	}
	t.Fatalf("word %q not in corpus", word)
	return 0
}
