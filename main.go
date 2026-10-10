// Copyright 2026 The prlang Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command prlang builds a weighted word graph, ranks it, and walks it.
//
// Words from the corpus are nodes. Case is kept, punctuation stays on the
// word it touches, and a line break stays on the word too. Spaces between
// words are not words. A sentence of more than one word is also a word in
// the graph: the words just before and at the end of it point at the
// sentence, and the sentence points at the next word. From a word that
// points at sentences, the walk gives those edges the same total chance as
// the word edges. Taking a sentence writes it and continues from the word
// after it. A word points at the words that follow it, and that link weighs
// the log of how often the pair occurs. A second-order link records which
// word followed a pair, weighed by the log of one plus how often that
// happened. The walk uses those links when the pair was seen. The prompt is
// a path of extra nodes tied to those words with
// heavy links, and PageRank's teleport lands on the prompt words with the
// same heavy weight. The walk starts on the prompt. From each node it steps
// to a neighbor with probability proportional to the link weight times that
// neighbor's PageRank.
//
//	prlang -prompt "To be, or not to be" -n 80
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"unicode"
)

const (
	// alpha is the PageRank damping factor.
	alpha = 0.85
	// rankEps is the L1 convergence tolerance for PageRank.
	rankEps = 1e-8
	// rankIters caps PageRank if it has not converged.
	rankIters = 200
)

func main() {
	file := flag.String("file", "pg100.txt", "corpus file")
	promptFlag := flag.String("prompt", "To be, or not to be", "prompt linked into the graph")
	n := flag.Int("n", 80, "words to generate after the prompt")
	seed := flag.Int64("seed", 1, "random seed")
	flag.Parse()

	if err := run(*file, *promptFlag, *n, *seed); err != nil {
		fmt.Fprintf(os.Stderr, "prlang: %s\n", err)
		os.Exit(1)
	}
}

func run(file, promptText string, n int, seed int64) error {
	if n < 0 {
		return fmt.Errorf("-n must be >= 0")
	}
	data, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	tokens := tokenize(stripGutenberg(string(data)))
	prompt := tokenize(promptText)
	if len(tokens) < 2 {
		return fmt.Errorf("%s has fewer than 2 words", file)
	}
	if len(prompt) == 0 {
		return fmt.Errorf("prompt has no words")
	}

	g := buildGraph(tokens, prompt)
	ranks, iters := pageRank(g.adj, g.personal, alpha, rankEps)
	rng := rand.New(rand.NewSource(seed))
	words := walk(rng, g, ranks, n)

	fmt.Fprintf(os.Stderr, "words=%d vocab=%d sentences=%d pairs=%d nodes=%d edges=%d prompt=%d heavy=%g iters=%d seed=%d\n",
		len(tokens), g.corpus-g.sentences, g.sentences, g.pairs, len(g.label), g.edges, len(prompt), g.heavy, iters, seed)
	fmt.Println(joinWords(words))
	return nil
}

// stripGutenberg drops the Project Gutenberg header and footer when the
// standard markers are present. Text without them is returned unchanged.
func stripGutenberg(s string) string {
	const start = "*** START OF THE PROJECT GUTENBERG EBOOK"
	const end = "*** END OF THE PROJECT GUTENBERG EBOOK"
	if i := strings.Index(s, start); i >= 0 {
		rest := s[i+len(start):]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			s = rest[nl+1:]
		} else {
			s = rest
		}
	}
	if i := strings.Index(s, end); i >= 0 {
		s = s[:i]
	}
	return s
}

// tokenize keeps case. Punctuation stays on the word it touches, so "be," is
// one word. Spaces and tabs between words are separators, not words. A line
// break is part of the word it ends: "\r\n" stays those two characters, and
// a blank line is the word "\r\n".
func tokenize(text string) []string {
	runes := []rune(text)
	out := make([]string, 0, len(runes)/5)
	buf := make([]rune, 0, 16)
	flush := func() {
		if len(buf) == 0 {
			return
		}
		out = append(out, string(buf))
		buf = buf[:0]
	}
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if r == '\r' && i+1 < len(runes) && runes[i+1] == '\n' {
			buf = append(buf, '\r', '\n')
			i++
			flush()
			continue
		}
		if r == '\n' || r == '\r' {
			buf = append(buf, r)
			flush()
			continue
		}
		if unicode.IsSpace(r) {
			flush()
			continue
		}
		buf = append(buf, r)
	}
	flush()
	return out
}

// joinWords puts a space between words. A word that already carries a line
// break is not given another separator.
func joinWords(words []string) string {
	var b strings.Builder
	for i, w := range words {
		if i > 0 && !breakEdge(words[i-1], w) {
			b.WriteByte(' ')
		}
		b.WriteString(w)
	}
	return b.String()
}

func breakEdge(prev, next string) bool {
	return strings.HasSuffix(prev, "\n") || strings.HasSuffix(prev, "\r") ||
		strings.HasPrefix(next, "\n") || strings.HasPrefix(next, "\r")
}

// sentenceEnd reports whether a word ends a sentence. A trailing line break
// and closing quotes are ignored, so "died.\r\n" and "died." both end one.
func sentenceEnd(word string) bool {
	w := strings.TrimRight(word, "\r\n")
	w = strings.TrimRight(w, "\"'”’")
	rs := []rune(w)
	if len(rs) == 0 {
		return false
	}
	switch rs[len(rs)-1] {
	case '.', '!', '?', '…':
		return true
	default:
		return false
	}
}

// sentenceSpans returns [start, end) token spans. A span runs from the
// previous end through the word that finishes the sentence. A one-word
// sentence is already a word and is not given another node.
func sentenceSpans(tokens []string) [][2]int {
	var spans [][2]int
	start := 0
	for i, w := range tokens {
		if !sentenceEnd(w) {
			continue
		}
		spans = append(spans, [2]int{start, i + 1})
		start = i + 1
	}
	return spans
}

type edge struct {
	to uint32
	w  float64
}

// graph is the corpus word graph plus one path node per prompt word.
type graph struct {
	label     []string
	adj       [][]edge
	personal  []float64
	start     uint32
	prompt    int
	corpus    int
	sentences int
	pairs     int
	pairStart int
	edges     int
	isSent    []bool
	resume    []uint32 // word a sentence walk continues from
	follow    []uint32 // corpus word a prompt or pair node stands for
	bigram    map[[2]uint32]uint32
	heavy     float64
	chain     float64
}

// buildGraph links each word to the word that follows it. The weight is the
// natural log of how often that happens, so a very common follower does not
// crowd out the rest. Each sentence of more than one word is added as a word
// whose text is the sentence. The word before it and the word that ends it
// point at that sentence, and the sentence points at the following word and
// at the next sentence. Those links weigh the log of one plus how often they
// occur, so a sentence seen once is still linked. Prompt words then form a
// path of new nodes:
// each points at the next prompt word with a chain weight, and at the same
// word in the corpus with a heavy weight. personal[i] is the unnormalized
// teleport weight of node i. Prompt words carry the heavy weight there too,
// once per time they occur in the prompt.
func buildGraph(tokens, prompt []string) graph {
	index := make(map[string]uint32, 1024)
	g := graph{prompt: len(prompt)}
	intern := func(w string) uint32 {
		if id, ok := index[w]; ok {
			return id
		}
		id := uint32(len(g.label))
		index[w] = id
		g.label = append(g.label, w)
		return id
	}

	ids := make([]uint32, len(tokens))
	for i, w := range tokens {
		ids[i] = intern(w)
	}
	out := make([]map[uint32]float64, len(g.label))
	for i := range out {
		out[i] = map[uint32]float64{}
	}
	for i := 0; i+1 < len(ids); i++ {
		out[ids[i]][ids[i+1]]++
	}
	for _, links := range out {
		for to, count := range links {
			links[to] = math.Log(count)
		}
	}

	grow := func() {
		for len(out) < len(g.label) {
			out = append(out, map[uint32]float64{})
		}
	}
	type pair struct{ from, to uint32 }
	sentCount := make(map[pair]float64)
	linkSent := func(from, to uint32) {
		if from == to {
			return
		}
		sentCount[pair{from, to}]++
	}
	var prev uint32
	havePrev := false
	seenSent := make(map[uint32]struct{})
	for _, sp := range sentenceSpans(tokens) {
		if sp[1]-sp[0] < 2 {
			prev = ids[sp[1]-1]
			havePrev = true
			continue
		}
		sid := intern(joinWords(tokens[sp[0]:sp[1]]))
		grow()
		if _, ok := seenSent[sid]; !ok {
			seenSent[sid] = struct{}{}
			g.sentences++
		}
		if sp[0] > 0 {
			linkSent(ids[sp[0]-1], sid)
		}
		linkSent(ids[sp[1]-1], sid)
		if sp[1] < len(ids) {
			linkSent(sid, ids[sp[1]])
		}
		if havePrev && (sp[0] == 0 || prev != ids[sp[0]-1]) {
			linkSent(prev, sid)
		}
		prev = sid
		havePrev = true
	}
	for p, count := range sentCount {
		out[p.from][p.to] += math.Log1p(count)
	}

	corpusOf := make([]uint32, len(prompt))
	for i, w := range prompt {
		corpusOf[i] = intern(w)
		for len(out) < len(g.label) {
			out = append(out, map[uint32]float64{})
		}
	}
	g.corpus = len(g.label)

	heavy := float64(len(tokens))
	// The forward prompt edge has to beat the side edge into a corpus word.
	// A prompt word can hold a large share of PageRank while a path node holds
	// about one teleport share, and that ratio is on the order of heavy.
	chain := heavy * heavy * 1000
	g.heavy = heavy
	g.chain = chain
	g.start = uint32(len(g.label))
	for _, w := range prompt {
		g.label = append(g.label, w)
		out = append(out, map[uint32]float64{})
	}
	for i := range prompt {
		from := g.start + uint32(i)
		if i+1 < len(prompt) {
			out[from][g.start+uint32(i+1)] = chain
		}
		out[from][corpusOf[i]] += heavy
	}

	g.personal = make([]float64, len(g.label))
	for i := range g.personal {
		g.personal[i] = 1
	}
	for _, id := range corpusOf {
		g.personal[id] += heavy
	}

	out = addSecondOrder(&g, out, ids)
	for i, id := range corpusOf {
		g.follow[g.start+uint32(i)] = id
	}

	g.isSent = make([]bool, len(g.label))
	g.resume = make([]uint32, len(g.label))
	for i := range g.resume {
		g.resume[i] = uint32(i)
	}
	for sid := range seenSent {
		g.isSent[sid] = true
		best := sid
		bestW := -1.0
		for to, w := range out[sid] {
			if _, ok := seenSent[to]; ok {
				continue
			}
			if w > bestW {
				bestW = w
				best = to
			}
		}
		g.resume[sid] = best
	}

	g.adj = make([][]edge, len(out))
	nEdges := 0
	for from, links := range out {
		edges := make([]edge, 0, len(links))
		for to, w := range links {
			edges = append(edges, edge{to: to, w: w})
		}
		sort.Slice(edges, func(i, j int) bool { return edges[i].to < edges[j].to })
		g.adj[from] = edges
		nEdges += len(edges)
	}
	g.edges = nEdges
	return g
}

// addSecondOrder adds a node for every observed word pair. A pair points at
// the pairs that followed it. The link weighs the log of one plus how often
// that trigram occurred, so a continuation seen once is still available.
func addSecondOrder(g *graph, out []map[uint32]float64, ids []uint32) []map[uint32]float64 {
	g.pairStart = len(g.label)
	g.bigram = make(map[[2]uint32]uint32)
	type key struct{ from, to uint32 }
	count := make(map[key]float64)
	var prev uint32
	have := false
	for i := 0; i+1 < len(ids); i++ {
		bk := [2]uint32{ids[i], ids[i+1]}
		id, ok := g.bigram[bk]
		if !ok {
			id = uint32(len(g.label))
			g.bigram[bk] = id
			g.label = append(g.label, g.label[ids[i+1]])
			out = append(out, map[uint32]float64{})
			g.personal = append(g.personal, 0)
		}
		if have {
			count[key{prev, id}]++
		}
		prev = id
		have = true
	}
	for k, c := range count {
		out[k.from][k.to] = math.Log1p(c)
	}
	g.pairs = len(g.bigram)
	g.follow = make([]uint32, len(g.label))
	for i := range g.follow {
		g.follow[i] = uint32(i)
	}
	for bk, id := range g.bigram {
		g.follow[id] = bk[1]
	}
	return out
}

// pageRank computes weighted personalized PageRank. personal is an
// unnormalized teleport weight per node. The returned vector sums to 1.
// iters is the number of power iterations that ran.
func pageRank(adj [][]edge, personal []float64, damp, eps float64) ([]float64, int) {
	n := len(adj)
	if n == 0 {
		return nil, 0
	}
	outSum := make([]float64, n)
	for i, edges := range adj {
		for _, e := range edges {
			outSum[i] += e.w
		}
	}
	p := make([]float64, n)
	psum := 0.0
	for i := range p {
		if personal != nil {
			p[i] = personal[i]
		} else {
			p[i] = 1
		}
		if p[i] < 0 || math.IsNaN(p[i]) {
			p[i] = 0
		}
		psum += p[i]
	}
	if psum == 0 {
		for i := range p {
			p[i] = 1 / float64(n)
		}
	} else {
		for i := range p {
			p[i] /= psum
		}
	}

	rank := make([]float64, n)
	next := make([]float64, n)
	copy(rank, p)
	iters := 0
	for iters = 1; iters <= rankIters; iters++ {
		for i := range next {
			next[i] = 0
		}
		leak := 0.0
		for i, edges := range adj {
			if len(edges) == 0 || outSum[i] == 0 {
				leak += rank[i]
				continue
			}
			scale := damp * rank[i] / outSum[i]
			for _, e := range edges {
				next[e.to] += scale * e.w
			}
		}
		base := (1 - damp) + damp*leak
		delta := 0.0
		for i := range next {
			next[i] += base * p[i]
			delta += math.Abs(next[i] - rank[i])
		}
		rank, next = next, rank
		if delta < eps {
			break
		}
	}
	return rank, iters
}

// walk starts at the first prompt node and emits the prompt words plus n
// more. Stepping from a prompt node onto the same word in the corpus does
// not write the word again. Each word that has been written is less likely
// to be chosen again. After two words, the next step follows the second-order
// links of that pair when the pair was seen. Choosing a sentence writes that
// sentence and continues from the word after it. A node with no outgoing
// link restarts at a node drawn from the PageRank distribution, with that
// same penalty.
func walk(rng *rand.Rand, g graph, rank []float64, n int) []string {
	want := g.prompt + n
	if want < 1 {
		return nil
	}
	used := make(map[string]int)
	words := make([]string, 0, want)
	cur := g.start
	var prev, curWord uint32
	var havePrev, haveCur bool
	note := func(id uint32) {
		w, ok := anchor(g, id)
		if !ok {
			havePrev, haveCur = false, false
			return
		}
		if haveCur {
			prev = curWord
			havePrev = true
		} else {
			havePrev = false
		}
		curWord = w
		haveCur = true
	}
	words = append(words, g.label[cur])
	used[g.label[cur]]++
	note(cur)
	choose := func(at uint32) uint32 {
		if at < g.start && havePrev && haveCur {
			if w, ok := anchor(g, at); ok && w == curWord {
				if edges := orderEdges(g, prev, curWord); len(edges) > 0 {
					return step(rng, edges, rank, g.label, used, g.isSent, g.resume)
				}
			}
		}
		return step(rng, g.adj[at], rank, g.label, used, g.isSent, g.resume)
	}
	for len(words) < want {
		next := choose(cur)
		if isPrompt(g, cur) && next < g.start && g.label[next] == g.label[cur] {
			cur = next
			continue
		}
		if int(next) < len(g.isSent) && g.isSent[next] {
			words = append(words, g.label[next])
			used[g.label[next]]++
			note(next)
			land := landAfter(rng, g, next, rank, used)
			cur = land
			if len(words) < want && land != next {
				words = append(words, g.label[land])
				used[g.label[land]]++
				note(land)
			}
			continue
		}
		cur = next
		words = append(words, g.label[cur])
		used[g.label[cur]]++
		note(cur)
	}
	return words
}

func isPrompt(g graph, id uint32) bool {
	return int(id) >= int(g.start) && int(id) < g.pairStart
}

// anchor is the corpus word a node stands for. Sentences are not words.
func anchor(g graph, id uint32) (uint32, bool) {
	if int(id) < len(g.isSent) && g.isSent[id] {
		return 0, false
	}
	if int(id) >= int(g.start) {
		if g.follow != nil && int(id) < len(g.follow) {
			return g.follow[id], true
		}
		return 0, false
	}
	if int(id) < g.corpus {
		return id, true
	}
	return 0, false
}

// orderEdges lists the words that followed the pair (prev, cur), plus any
// sentence edges leaving cur. Nil means the pair was not seen.
func orderEdges(g graph, prev, cur uint32) []edge {
	if g.bigram == nil {
		return nil
	}
	pid, ok := g.bigram[[2]uint32{prev, cur}]
	if !ok || int(pid) >= len(g.adj) {
		return nil
	}
	acc := make(map[uint32]float64)
	add := func(to uint32, w float64) {
		if w > 0 {
			acc[to] += w
		}
	}
	for _, e := range g.adj[pid] {
		if g.follow != nil && int(e.to) < len(g.follow) {
			add(g.follow[e.to], e.w)
		}
	}
	if int(cur) < len(g.adj) {
		for _, e := range g.adj[cur] {
			if int(e.to) < len(g.isSent) && g.isSent[e.to] {
				add(e.to, e.w)
			}
		}
	}
	if len(acc) == 0 {
		return nil
	}
	edges := make([]edge, 0, len(acc))
	for to, w := range acc {
		edges = append(edges, edge{to: to, w: w})
	}
	sort.Slice(edges, func(i, j int) bool { return edges[i].to < edges[j].to })
	return edges
}

// landAfter samples the word edge that leaves a sentence. Sentence-to-sentence
// edges stay in the graph for PageRank and are not where the walk resumes.
func landAfter(rng *rand.Rand, g graph, sid uint32, rank []float64, used map[string]int) uint32 {
	var edges []edge
	for _, e := range g.adj[sid] {
		if int(e.to) < len(g.isSent) && g.isSent[e.to] {
			continue
		}
		edges = append(edges, e)
	}
	if len(edges) == 0 {
		return g.resume[sid]
	}
	return step(rng, edges, rank, g.label, used, nil, nil)
}

// reusePenalty is 1 the first time a word can be chosen, then 1/2, 1/3, …
// after each time it has been written.
func reusePenalty(used map[string]int, word string) float64 {
	return 1 / float64(1+used[word])
}

// neighborRank is the node's PageRank. A sentence uses the rank of the word
// after it, so the edge is not crushed by the sentence node's own rank.
func neighborRank(to uint32, rank []float64, isSent []bool, resume []uint32) float64 {
	r := rank[to]
	if isSent == nil || int(to) >= len(isSent) || !isSent[to] || resume == nil || int(to) >= len(resume) {
		return r
	}
	rw := resume[to]
	if int(rw) < len(rank) && rank[rw] > r {
		return rank[rw]
	}
	return r
}

func step(rng *rand.Rand, edges []edge, rank []float64, labels []string, used map[string]int, isSent []bool, resume []uint32) uint32 {
	if len(edges) == 0 {
		return sampleRank(rng, rank, labels, used)
	}
	total := 0.0
	sumSent := 0.0
	maxSent := 0.0
	score := make([]float64, len(edges))
	sent := make([]bool, len(edges))
	for i, e := range edges {
		s := e.w * neighborRank(e.to, rank, isSent, resume) * reusePenalty(used, labels[e.to])
		if s < 0 || math.IsNaN(s) {
			s = 0
		}
		score[i] = s
		if isSent != nil && int(e.to) < len(isSent) && isSent[e.to] {
			sent[i] = true
			sumSent += s
			if s > maxSent {
				maxSent = s
			}
			continue
		}
		total += s
	}
	// Sentence edges share one chance equal to the word edges together, so a
	// sentence can be chosen without the crowd of them drowning the walk.
	// Word edges keep their weights relative to each other.
	if sumSent > 0 {
		target := total
		if target == 0 {
			target = maxSent
		}
		if target > 0 {
			scale := target / sumSent
			for i := range score {
				if sent[i] {
					score[i] *= scale
					total += score[i]
				}
			}
		}
	}
	if total == 0 {
		total = 0
		for i, e := range edges {
			score[i] = e.w
			total += e.w
		}
	}
	if total == 0 {
		return edges[len(edges)-1].to
	}
	x := rng.Float64() * total
	acc := 0.0
	for i, e := range edges {
		acc += score[i]
		if x < acc {
			return e.to
		}
	}
	return edges[len(edges)-1].to
}

func sampleRank(rng *rand.Rand, rank []float64, labels []string, used map[string]int) uint32 {
	total := 0.0
	score := make([]float64, len(rank))
	for i, r := range rank {
		if r <= 0 {
			continue
		}
		score[i] = r * reusePenalty(used, labels[i])
		total += score[i]
	}
	if total == 0 {
		return 0
	}
	x := rng.Float64() * total
	acc := 0.0
	for i, s := range score {
		acc += s
		if x < acc {
			return uint32(i)
		}
	}
	return uint32(len(rank) - 1)
}
