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
	want := []string{"to", "be", "or", "not", "to", "be"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("tokenize = %q", got)
	}
	got = tokenize("know’st o’er ’tis well-known")
	want = []string{"know'st", "o'er", "'tis", "well-known"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("tokenize = %q", got)
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

	text := strings.Repeat("alpha one two beta three four ", 30)
	g = buildGraph(tokenize(text), []string{"alpha", "beta"})
	rank, _ = pageRank(g.adj, g.personal, alpha, rankEps)
	rng := rand.New(rand.NewSource(1))
	got := walk(rng, g, rank, 4)
	want := []string{"alpha", "beta", "three", "four", "alpha", "one"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("walk = %q, want %q", got, want)
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
