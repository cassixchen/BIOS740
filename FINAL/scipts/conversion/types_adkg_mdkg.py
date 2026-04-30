import json
import os

adkg_types = {
    "entities": {
        "disease": {
            "short": "disease",
            "verbose": "Diseases, disorders, pathological conditions"
        },
        "drug": {
            "short": "drug",
            "verbose": "Drugs, compounds, therapeutic agents"
        },
        "gene": {
            "short": "gene",
            "verbose": "Genes, proteins, genetic elements"
        },
        "method": {
            "short": "method",
            "verbose": "Methods, tools, diagnostic approaches"
        },
        "mutation": {
            "short": "mutation",
            "verbose": "Genetic variants and polymorphisms"
        },
        "other": {
            "short": "other",
            "verbose": "Miscellaneous biomedical concepts"
        }
    },

    "relations": {
        "abbreviation_for": {
            "short": "abbreviation_for",
            "verbose": "The head entity is an abbreviation or acronym for the tail entity. Both refer to the same concept.",
            "symmetric": False
        },
        "associated_with": {
            "short": "associated_with",
            "verbose": "General association or correlation where no specific causal direction applies.",
            "symmetric": True
        },
        "hyponym_of": {
            "short": "hyponym_of",
            "verbose": "The head entity is a subtype or instance of the tail entity (IS-A relation).",
            "symmetric": False
        },
        "treatment_for": {
            "short": "treatment_for",
            "verbose": "The head entity is a treatment or intervention for the tail entity.",
            "symmetric": False
        },
        "risk_factor_of": {
            "short": "risk_factor_of",
            "verbose": "The head entity increases risk or likelihood of the tail entity.",
            "symmetric": False
        },
        "help_diagnose": {
            "short": "help_diagnose",
            "verbose": "The head entity is used to diagnose or assess the tail entity.",
            "symmetric": False
        },
        "characteristic_of": {
            "short": "characteristic_of",
            "verbose": "The head entity is a characteristic or manifestation of the tail entity.",
            "symmetric": False
        },
        "treatment_target_for": {
            "short": "treatment_target_for",
            "verbose": "The head entity is a molecular/cellular target for treating the tail entity.",
            "symmetric": False
        }
    }
}



mdkg_types = {
    "entities": {
        "disease": {
            "short": "disease",
            "verbose": "Diseases and disorders"
        },
        "drug": {
            "short": "drug",
            "verbose": "Drugs and therapeutic agents"
        },
        "gene": {
            "short": "gene",
            "verbose": "Genes and proteins"
        },
        "method": {
            "short": "method",
            "verbose": "Diagnostic and clinical methods"
        },
        "Health_factors": {
            "short": "health_factors",
            "verbose": "Demographics, lifestyle, environmental factors"
        },
        "physiology": {
            "short": "physiology",
            "verbose": "Physiological processes and mechanisms"
        },
        "region": {
            "short": "region",
            "verbose": "Anatomical regions and structures"
        },
        "signs": {
            "short": "signs",
            "verbose": "Measurable clinical indicators"
        },
        "symptom": {
            "short": "symptom",
            "verbose": "Observable clinical manifestations"
        }
    },

    "relations": {
        "abbreviation_for": {
            "short": "abbreviation_for",
            "verbose": "The head entity is an abbreviation or acronym for the tail entity. Both refer to the same concept.",
            "symmetric": False
        },
        "associated_with": {
            "short": "associated_with",
            "verbose": "General association or correlation where no specific causal direction applies.",
            "symmetric": True
        },
        "hyponym_of": {
            "short": "hyponym_of",
            "verbose": "The head entity is a subtype or instance of the tail entity (IS-A relation).",
            "symmetric": False
        },
        "treatment_for": {
            "short": "treatment_for",
            "verbose": "The head entity is a treatment or intervention for the tail entity.",
            "symmetric": False
        },
        "risk_factor_of": {
            "short": "risk_factor_of",
            "verbose": "The head entity increases risk or likelihood of the tail entity.",
            "symmetric": False
        },
        "help_diagnose": {
            "short": "help_diagnose",
            "verbose": "The head entity is used to diagnose or assess the tail entity.",
            "symmetric": False
        },
        "characteristic_of": {
            "short": "characteristic_of",
            "verbose": "The head entity is a characteristic or manifestation of the tail entity.",
            "symmetric": False
        },
        "occurs_in": {
            "short": "occurs_in",
            "verbose": "The head entity occurs in a population or disease context.",
            "symmetric": False
        },
        "located_in": {
            "short": "located_in",
            "verbose": "The head entity is spatially located in or expressed within the tail entity.",
            "symmetric": False
        }
    }
}

out_path = "data/datasets/adkg/adkg_types.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    json.dump(adkg_types, f, indent=2)
    
out_path = "data/datasets/mdkg/mdkg_types.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    json.dump(mdkg_types, f, indent=2)


