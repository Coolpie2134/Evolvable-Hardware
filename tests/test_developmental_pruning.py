"""Pruning must preserve ontogeny and must not edit the source genome."""
import random

from runtime.checkpoint import genome_to_dict, genome_from_dict
from substrates.fnv.construction import develop_constructive
from substrates.fnv.construction_ga import placement_genes, mutate_branched
from substrates.fnv.genome import ContextGene, random_functional_genome, validate_genome
from substrates.fnv.catalogue import DEFAULT_FAMILIES
from tools.prune_developmental_checkpoint import prune_dormant_rules


def test_pruning_removes_a_dormant_rule_and_preserves_growth_and_source():
    random.seed(904399)
    genome = random_functional_genome(2, n_inputs=3, output_roles=('Sum', 'Carry'))
    # Impossible-to-observe sentinel combination at a non-root site.
    from substrates.fnv.genome import OUT_STATE, EMPTY_STATE
    doomed = ContextGene(genome.next_gene_id, OUT_STATE, OUT_STATE, OUT_STATE,
                         EMPTY_STATE, 1, 1)
    genome.next_gene_id += 1
    chromosome = genome.chromosomes[0]
    chromosome.genes.insert(0, doomed)
    chromosome.split += 1
    document = genome_to_dict(genome, 'fnv')
    child, audit = prune_dormant_rules(genome)
    assert audit['removed'] >= 1
    assert doomed.gene_id not in {gene.gene_id for gene in placement_genes(child)}
    assert genome_to_dict(genome, 'fnv') == document
    assert develop_constructive(child, child.input_layout, snapshots=True).snapshots == develop_constructive(
        genome, genome.input_layout, snapshots=True).snapshots
    validate_genome(genome_from_dict(genome_to_dict(child, 'fnv'), 'fnv'))


def test_pruning_preserves_growth_after_multiple_developmental_mutations():
    for seed in range(5):
        random.seed(seed + 904400)
        genome = random_functional_genome(2, n_inputs=3, output_roles=('Sum', 'Carry'))
        for _ in range(20):
            mutate_branched(genome, 2, DEFAULT_FAMILIES, 3)
        document = genome_to_dict(genome, 'fnv')
        child, audit = prune_dormant_rules(genome)
        assert audit['rules_after'] <= audit['rules_before']
        assert genome_to_dict(genome, 'fnv') == document
        original = develop_constructive(genome, genome.input_layout, snapshots=True)
        pruned = develop_constructive(child, child.input_layout, snapshots=True)
        assert original.snapshots == pruned.snapshots
        assert original.owners == pruned.owners
        assert original.builders == pruned.builders
