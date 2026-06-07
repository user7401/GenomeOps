"""Shared fixtures and mock data for GenomeOps tests."""

import pytest
import respx
import httpx


MOCK_PIPELINES_RESPONSE = {
    "pipelines": [
        {
            "name": "rnaseq",
            "description": "RNA sequencing analysis pipeline using STAR, RSEM, HISAT2 or Salmon",
            "topics": ["RNA-seq", "transcriptomics"],
            "latest_version": "3.14.0",
        },
        {
            "name": "sarek",
            "description": "Detect germline or somatic variants from normal or tumour/normal samples",
            "topics": ["variant-calling", "WGS", "WES"],
            "latest_version": "3.4.4",
        },
        {
            "name": "chipseq",
            "description": "ChIP-seq peak-calling, QC and differential analysis pipeline",
            "topics": ["ChIP-seq", "epigenetics"],
            "latest_version": "2.0.0",
        },
        {
            "name": "methylseq",
            "description": "Methylation (Bisulfite-Sequencing) analysis pipeline",
            "topics": ["methylation", "WGBS"],
            "latest_version": "2.6.0",
        },
    ]
}

MOCK_RNASEQ_DETAIL = {
    "name": "rnaseq",
    "description": "RNA sequencing analysis pipeline using STAR, RSEM, HISAT2 or Salmon",
    "topics": ["RNA-seq", "transcriptomics"],
    "latest_version": "3.14.0",
}

MOCK_SAMPLESHEET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "nf-core/rnaseq pipeline - params.config schema",
    "items": {
        "type": "object",
        "required": ["sample", "fastq_1", "strandedness"],
        "properties": {
            "sample": {
                "type": "string",
                "description": "Custom sample name.",
            },
            "fastq_1": {
                "type": "string",
                "description": "Full path to R1 FASTQ file.",
                "format": "file-path",
            },
            "fastq_2": {
                "type": "string",
                "description": "Full path to R2 FASTQ file (optional for SE).",
                "format": "file-path",
            },
            "strandedness": {
                "type": "string",
                "description": "Library strandedness.",
                "enum": ["forward", "reverse", "unstranded"],
            },
        },
    },
}

MOCK_NEXTFLOW_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "nf-core/rnaseq pipeline parameters",
    "definitions": {
        "input_output_options": {
            "title": "Input/output options",
            "type": "object",
            "required": ["input", "outdir"],
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Path to samplesheet CSV.",
                    "format": "file-path",
                },
                "outdir": {
                    "type": "string",
                    "description": "Output directory.",
                    "format": "directory-path",
                },
            },
        },
        "reference_genome_options": {
            "title": "Reference genome options",
            "type": "object",
            "properties": {
                "genome": {
                    "type": "string",
                    "description": "Name of iGenomes reference.",
                    "enum": ["GRCh38", "GRCh37", "GRCm38", "mm10"],
                    "default": None,
                },
                "fasta": {
                    "type": "string",
                    "description": "Path to FASTA genome file.",
                    "format": "file-path",
                },
            },
        },
        "alignment_options": {
            "title": "Alignment options",
            "type": "object",
            "properties": {
                "aligner": {
                    "type": "string",
                    "description": "Alignment tool.",
                    "enum": ["star_salmon", "star_rsem", "hisat2"],
                    "default": "star_salmon",
                },
            },
        },
    },
    "allOf": [
        {"$ref": "#/definitions/input_output_options"},
        {"$ref": "#/definitions/reference_genome_options"},
        {"$ref": "#/definitions/alignment_options"},
    ],
}


@pytest.fixture
def mock_nfcore_api():
    """Mock the nf-co.re API and GitHub API fallback responses."""
    with respx.mock(assert_all_called=False) as mock:
        # --- nf-co.re primary API ---
        mock.get("https://nf-co.re/api/v2/pipelines").mock(
            return_value=httpx.Response(200, json=MOCK_PIPELINES_RESPONSE)
        )
        mock.get("https://nf-co.re/api/v2/pipelines/rnaseq").mock(
            return_value=httpx.Response(200, json=MOCK_RNASEQ_DETAIL)
        )
        mock.get("https://nf-co.re/api/v2/pipelines/sarek").mock(
            return_value=httpx.Response(200, json={
                "name": "sarek",
                "description": "Variant calling pipeline",
                "topics": ["variant-calling"],
                "latest_version": "3.4.4",
            })
        )
        mock.get("https://nf-co.re/api/v2/pipelines/notexist").mock(
            return_value=httpx.Response(404)
        )
        mock.get("https://nf-co.re/api/v2/pipelines/rnseq").mock(
            return_value=httpx.Response(404)
        )

        # --- GitHub API fallback (triggered when nf-co.re returns 404) ---
        mock.get("https://api.github.com/repos/nf-core/notexist").mock(
            return_value=httpx.Response(404)
        )
        mock.get("https://api.github.com/repos/nf-core/rnseq").mock(
            return_value=httpx.Response(404)
        )

        yield mock


@pytest.fixture
def mock_github_schemas():
    """Mock GitHub raw content responses for schemas."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(
            "https://raw.githubusercontent.com/nf-core/rnaseq/master/assets/schema_input.json"
        ).mock(return_value=httpx.Response(200, json=MOCK_SAMPLESHEET_SCHEMA))

        mock.get(
            "https://raw.githubusercontent.com/nf-core/rnaseq/3.14.0/assets/schema_input.json"
        ).mock(return_value=httpx.Response(200, json=MOCK_SAMPLESHEET_SCHEMA))

        mock.get(
            "https://raw.githubusercontent.com/nf-core/rnaseq/master/nextflow_schema.json"
        ).mock(return_value=httpx.Response(200, json=MOCK_NEXTFLOW_SCHEMA))

        mock.get(
            "https://raw.githubusercontent.com/nf-core/notexist/master/assets/schema_input.json"
        ).mock(return_value=httpx.Response(404))

        mock.get(
            "https://raw.githubusercontent.com/nf-core/notexist/master/nextflow_schema.json"
        ).mock(return_value=httpx.Response(404))

        yield mock
