"""Prune unexpressed FNV rules only when the complete growth movie is identical.

Post-processing only: not enabled inside evolution. Dormant material can be
useful to future mutations, so preservation of today's development is not a
claim of preserved evolvability. Source checkpoints are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.checkpoint import genome_to_dict, load_checkpoint, save_checkpoint
from substrates.fnv.construction import develop_constructive
from substrates.fnv.construction_ga import clone_constructive, placement_genes
from substrates.fnv.genome import ContextGene, validate_genome


def remove_rules(genome, ids):
    child = clone_constructive(genome)
    for chromosome in child.chromosomes:
        original = list(chromosome.genes)
        cut = chromosome.split
        keep = lambda gene: not (isinstance(gene, ContextGene) and gene.gene_id in ids)
        chromosome.genes = [gene for gene in original if keep(gene)]
        chromosome.split = sum(keep(gene) for gene in original[:cut])
    return child


def prune_dormant_rules(genome):
    """Return an independent, verified genome and a small audit record."""
    validate_genome(genome)
    original = develop_constructive(genome, genome.input_layout, snapshots=True)
    ids = {gene.gene_id for gene in placement_genes(genome)
           if gene.gene_id not in original.active_ids and not gene.spawns_output()}
    candidate = remove_rules(genome, ids)
    attempts = 1
    def same_movie(child):
        trace = develop_constructive(child, child.input_layout, snapshots=True)
        return (trace.snapshots == original.snapshots and trace.grid == original.grid
                and trace.owners == original.owners and trace.branch_depths == original.branch_depths
                and trace.builders == original.builders)
    if not same_movie(candidate):
        # A silent rule can win a no-op and suppress another rule. Only retain
        # deletions that have actually been shown to preserve all growth waves.
        candidate = clone_constructive(genome)
        for gene_id in sorted(ids):
            child = remove_rules(candidate, {gene_id})
            attempts += 1
            if same_movie(child):
                candidate = child
    validate_genome(candidate)
    assert same_movie(candidate)
    before = len(placement_genes(genome))
    after = len(placement_genes(candidate))
    return candidate, {'rules_before': before, 'rules_after': after,
                       'removed': before - after, 'regrowth_checks': attempts + 1,
                       'identical_growth_frames': len(original.snapshots)}


def cold_growth_median(genome, repeats=15):
    import statistics
    times = []
    for _ in range(repeats):
        # snapshots=True bypasses the cache and measures actual development.
        started = time.perf_counter()
        develop_constructive(genome, genome.input_layout, snapshots=True)
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def process(source, output):
    from substrates.fnv.evaluation import evaluate_functional_full
    from substrates.nervous.certification import certify
    state = load_checkpoint(str(source))
    if state['backend'] != 'fnv':
        raise ValueError('Expected an FNV checkpoint')
    if source.resolve() == output.resolve():
        raise ValueError('Refusing to overwrite source checkpoint')
    target = state['target']
    if 'best_genome' in state:
        genome = state['best_genome']
    else:
        genome = max(state['genomes'], key=lambda g: evaluate_functional_full(g, target)[0])
    original_document = genome_to_dict(genome, 'fnv')
    before_score = evaluate_functional_full(genome, target)
    child, record = prune_dormant_rules(genome)
    after_score = evaluate_functional_full(child, target)
    assert before_score == after_score
    before_certification = certify(genome, target, train=before_score[0], backend='fnv')
    after_certification = certify(child, target, train=after_score[0], backend='fnv')
    assert before_certification == after_certification
    assert genome_to_dict(genome, 'fnv') == original_document
    output.parent.mkdir(parents=True, exist_ok=True)
    # Attach a fresh certification result, never an old checkpoint's claim.
    save_checkpoint(str(output), child, after_score[0], target, state.get('arch'),
                    state.get('seed'), 'fnv', state['run_config'], after_certification)
    reloaded = load_checkpoint(str(output))
    assert evaluate_functional_full(reloaded['best_genome'], reloaded['target']) == before_score
    compact = lambda g: len(json.dumps(genome_to_dict(g, 'fnv'), sort_keys=True, separators=(',', ':')).encode())
    record.update(source=str(source), output=str(output), target=target.name,
                  fitness=after_score[0], genotype_bytes_before=compact(genome),
                  genotype_bytes_after=compact(child),
                  cold_growth_median_before_s=cold_growth_median(genome),
                  cold_growth_median_after_s=cold_growth_median(child),
                  all_case_scores_identical=True, source_genome_unmodified=True,
                  saved_checkpoint_replay_identical=True,
                  certification=after_certification,
                  original_and_pruned_certification_identical=True)
    output.with_suffix('.audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(process(args.checkpoint, args.out), indent=2))


if __name__ == '__main__':
    main()
