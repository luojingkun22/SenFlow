# MOSAIC Data Release Policy

MOSAIC is constructed from two upstream research corpora:

- PubMed scientific-paper data from the Cohan et al. long-document summarization corpus.
- XSum news data from Narayan et al.

The public review repository is license-aware. It provides code, prompts, seeds, generation settings, document references, sentence labels/spans, and AI-generated replacement sentences needed to inspect the benchmark format and reconstruct the corpus locally.

To avoid indiscriminate redistribution of full upstream source text, the repository does not include full `original_text` or full `hybrid_text` fields for the complete benchmark. Users should obtain the upstream datasets from their official sources and rebuild the full MOSAIC files locally using the released scripts, subject to the upstream datasets' license and redistribution terms.

The included file `MOSAIC/MOSAIC_sample_license_aware.json` is a small schema sample for double-blind review and pipeline inspection only.
