#!/usr/bin/env python3
"""
Module: map_entities.py
Provides auditable, rule-based and lexicon-backed mapping of BC5CDR entities
to the project's target NER schema:
  - Disease -> DISEASE
  - Chemical -> MEDICATION (only when verified as a pharmaceutical drug/substance)
  - Non-medication chemicals -> Retained as CHEMICAL
"""

import re
import csv
import json
import logging
from typing import Dict, List, Tuple, Set, Any

logger = logging.getLogger(__name__)

# Standard USAN/INN Pharmacological stems
PHARMACOLOGICAL_STEMS = [
    r".*olol$",        # Beta blockers (e.g. propranolol, atenolol, metoprolol)
    r".*pril$",        # ACE inhibitors (e.g. captopril, enalapril)
    r".*sartan$",      # Angiotensin II receptor antagonists (e.g. losartan)
    r".*statin$",      # HMG-CoA reductase inhibitors (e.g. atorvastatin, simvastatin)
    r".*cillin$",      # Penicillins (e.g. ampicillin, amoxicillin)
    r".*mycin$",       # Macrolides / aminoglycosides (e.g. streptomycin, erythromycin)
    r".*micin$",       # Aminoglycosides (e.g. gentamicin)
    r".*oxacin$",      # Fluoroquinolones (e.g. ciprofloxacin)
    r".*cycline$",     # Tetracyclines (e.g. doxycycline)
    r".*prazole$",     # Proton pump inhibitors (e.g. omeprazole)
    r".*tidine$",      # H2 receptor antagonists (e.g. cimetidine, ranitidine)
    r".*dipine$",      # Dihydropyridine calcium channel blockers (e.g. nifedipine)
    r".*mab$",         # Monoclonal antibodies (e.g. rituximab, trastuzumab)
    r".*nib$",         # Kinase inhibitors (e.g. imatinib)
    r".*afil$",        # PDE5 inhibitors (e.g. sildenafil)
    r".*azepam$",      # Benzodiazepines (e.g. diazepam, lorazepam)
    r".*azolam$",      # Benzodiazepines (e.g. midazolam, alprazolam)
    r".*azine$",       # Phenothiazines (e.g. chlorpromazine)
    r".*asone$",       # Corticosteroids (e.g. dexamethasone)
    r".*olone$",       # Corticosteroids (e.g. prednisolone)
    r".*avir$",        # Antivirals (e.g. ribavirin, acyclovir, ritonavir)
    r".*caine$",       # Local anesthetics (e.g. lidocaine, bupivacaine, procaine)
    r".*terol$",       # Beta-2 agonists (e.g. isoproterenol, albuterol)
    r".*barbital$",    # Barbiturates (e.g. phenobarbital)
    r".*zepine$",      # Anticonvulsants (e.g. carbamazepine)
    r".*triptan$",     # Serotonin 5-HT1 agonists (e.g. sumatriptan)
    r".*glitazone$",   # Thiazolidinediones (e.g. rosiglitazone, pioglitazone)
    r".*gliptin$",     # DPP-4 inhibitors (e.g. sitagliptin)
    r".*parin$",       # Heparins (e.g. enoxaparin, tinzaparin)
    r".*dopa$",        # Dopaminergic agents (e.g. levodopa, methyldopa, carbidopa)
]

# Explicit non-medication exclusion list (endogenous metabolites, minerals, electrolytes, industrial toxins)
NON_MEDICATION_EXCLUSIONS: Set[str] = {
    # Endogenous metabolites & biomarkers
    "creatinine", "urea", "uric acid", "homocysteine", "bilirubin", "cholesterol",
    "lactate", "lactic acid", "glucose", "glycogen", "pyruvate", "citrate",
    "malondialdehyde", "mda", "glutathione", "gsh", "superoxide", "nitric oxide", "no",
    # Electrolytes, minerals, elemental metals
    "calcium", "sodium", "potassium", "magnesium", "lead", "iron", "mercury",
    "cadmium", "copper", "zinc", "arsenic", "chloride", "phosphate", "sulfate",
    "na", "k", "ca", "mg", "fe", "pb", "cd", "hg", "cu", "zn",
    # Endogenous neurotransmitters/hormones when described physiologically
    "serotonin", "glutamate", "gaba", "acetylcholine", "glycine",
    # Environmental/laboratory reagents and toxins
    "carbon monoxide", "co", "carbon tetrachloride", "ccl4", "benzene", "toluene",
    "formaldehyde", "paraquat", "cyanide", "asbestos", "dimethyl sulfoxide", "dmso",
    "hydrogen peroxide", "oxygen", "ozone", "silica", "rotenone", "kainic acid",
    "streptozotocin", "lipopolysaccharide", "lps", "bleomycin" # frequently used as animal disease models
}

# Curated High-Confidence Medication Gazetteers (Frequent BC5CDR Drugs)
CURATED_MEDICATIONS: Set[str] = {
    # Analgesics & Anesthetics
    "morphine", "cocaine", "fentanyl", "codeine", "methadone", "buprenorphine", "oxycodone",
    "aspirin", "acetylsalicylic acid", "acetaminophen", "paracetamol", "ibuprofen", "indomethacin",
    "ketoprofen", "diclofenac", "naproxen", "celecoxib", "tramadol", "ketamine", "propofol",
    "lidocaine", "lignocaine", "bupivacaine", "ropivacaine", "halothane", "isoflurane", "sevoflurane",
    # Chemotherapy & Antineoplastics
    "doxorubicin", "adriamycin", "dox", "cisplatin", "cyclophosphamide", "paclitaxel", "taxol",
    "methotrexate", "fluorouracil", "5-fluorouracil", "5 - fu", "vincristine", "vinblastine",
    "bleomycin", "carboplatin", "oxaliplatin", "tamoxifen", "etoposide", "daunorubicin",
    "cytarabine", "busulfan", "mitomycin", "hydroxyurea", "chlorambucil", "irinotecan",
    # Cardiovascular & Vasoactive
    "clonidine", "propranolol", "atenolol", "metoprolol", "carvedilol", "esmolol", "labetalol",
    "amiodarone", "verapamil", "diltiazem", "nifedipine", "nicardipine", "nimodipine",
    "captopril", "enalapril", "lisinopril", "ramipril", "losartan", "valsartan",
    "methyldopa", "alpha-methyldopa", "alpha - methyldopa",
    "digoxin", "digitoxin", "warfarin", "heparin", "low molecular weight heparin",
    "epinephrine", "adrenaline", "norepinephrine", "noradrenaline", "isoproterenol", "isoprenaline",
    "dopamine", "dobutamine", "phenylephrine", "quinidine", "procainamide", "lidocaine",
    "nitroprusside", "sodium nitroprusside", "nitroglycerin", "hydralazine", "minoxidil",
    "furosemide", "hydrochlorothiazide", "spironolactone", "mannitol",
    # Neurological & Psychiatric
    "lithium", "haloperidol", "chlorpromazine", "clozapine", "olanzapine", "risperidone",
    "quetiapine", "aripiprazole", "sulpiride", "fluphenazine", "pimozide", "thioridazine",
    "diazepam", "lorazepam", "midazolam", "clonazepam", "alprazolam", "chlordiazepoxide",
    "phenobarbital", "pentobarbital", "thiopental", "phenytoin", "diphenylhydantoin",
    "carbamazepine", "valproate", "valproic acid", "sodium valproate", "ethosuximide",
    "lamotrigine", "topiramate", "gabapentin", "levodopa", "l-dopa", "carbidopa",
    "scopolamine", "atropine", "pilocarpine", "physostigmine", "neostigmine",
    "amphetamine", "d-amphetamine", "methamphetamine", "methylphenidate", "caffeine",
    "nicotine", "naloxone", "naltrexone", "apomorphine",
    # Antimicrobials
    "gentamicin", "tobramycin", "amikacin", "streptomycin", "kanamycin", "neomycin",
    "penicillin", "penicillin g", "amoxicillin", "ampicillin", "methicillin", "oxacillin",
    "cephalosporin", "cefazolin", "ceftriaxone", "cefotaxime", "ceftazidime",
    "vancomycin", "erythromycin", "azithromycin", "clarithromycin", "clindamycin",
    "ciprofloxacin", "ofloxacin", "levofloxacin", "tetracycline", "doxycycline",
    "rifampicin", "rifampin", "isoniazid", "pyrazinamide", "ethambutol",
    "amphotericin b", "fluconazole", "ketoconazole", "itraconazole", "nystatin",
    "acyclovir", "ganciclovir", "zidovudine", "azt", "ribavirin", "chloroquine",
    # Immunosuppressants & Steroids
    "cyclosporine", "cyclosporin a", "tacrolimus", "fk506", "sirolimus", "rapamycin",
    "azathioprine", "mycophenolate", "mycophenolate mofetil", "thalidomide",
    "dexamethasone", "prednisone", "prednisolone", "methylprednisolone", "hydrocortisone",
    "cortisone", "betamethasone", "triamcinolone", "beclomethasone",
    # Endocrine & Metabolic
    "insulin", "metformin", "glibenclamide", "glyburide", "glipizide", "tolbutamide",
    "thyroxine", "levothyroxine", "propylthiouracil", "methimazole", "somatostatin",
    "tamoxifen", "estrogen", "estradiol", "progesterone", "testosterone"
}

class EntityMapper:
    """Auditable entity mapping engine for BC5CDR to target NER schema."""
    
    def __init__(self, retain_unmapped_chemical: bool = False):
        self.retain_unmapped_chemical = retain_unmapped_chemical
        self.mapping_decisions: List[Dict[str, Any]] = []
        
    def classify_chemical_mention(self, mention_text: str) -> Tuple[str, str]:
        """
        Classify a chemical mention as MEDICATION or CHEMICAL based on auditable rules.
        Returns: (target_class, rationale)
        """
        norm_text = mention_text.strip().lower()
        
        # Rule 1: Check explicit non-medication exclusion list
        if norm_text in NON_MEDICATION_EXCLUSIONS:
            return "CHEMICAL", f"Excluded: endogenous metabolite/electrolyte/toxin '{norm_text}'"
            
        # Rule 2: Check curated medication gazetteer
        if norm_text in CURATED_MEDICATIONS:
            return "MEDICATION", f"Curated gazetteer match for approved pharmaceutical '{norm_text}'"
            
        # Check if individual words in multi-word phrase match curated drugs
        words = re.findall(r"[a-z0-9]+", norm_text)
        for w in words:
            if w in CURATED_MEDICATIONS:
                return "MEDICATION", f"Curated drug component '{w}' in mention '{norm_text}'"
                
        # Rule 3: Check USAN/INN pharmacological stem regexes
        for pattern in PHARMACOLOGICAL_STEMS:
            for w in words:
                if re.match(pattern, w):
                    return "MEDICATION", f"Pharmacological stem match ({pattern}) in token '{w}'"
                    
        # Rule 4: Default fallback
        if self.retain_unmapped_chemical:
            return "CHEMICAL", f"Default fallback: unverified chemical entity '{norm_text}' retained as CHEMICAL"
        else:
            return "CHEMICAL", f"Default decision: non-medication chemical '{norm_text}' retained as CHEMICAL"

    def map_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a single record from BC5CDR raw schema to target schema.
        Input record must contain 'tokens' and 'raw_tags'.
        """
        tokens = record["tokens"]
        raw_tags = record["raw_tags"]
        
        target_tags = []
        i = 0
        n = len(tokens)
        
        while i < n:
            tag = raw_tags[i]
            
            if tag == "O":
                target_tags.append("O")
                i += 1
            elif tag == "B-Disease":
                target_tags.append("B-DISEASE")
                i += 1
            elif tag == "I-Disease":
                target_tags.append("I-DISEASE")
                i += 1
            elif tag == "B-Chemical":
                # Extract full multi-token chemical mention
                chem_tokens = [tokens[i]]
                j = i + 1
                while j < n and raw_tags[j] == "I-Chemical":
                    chem_tokens.append(tokens[j])
                    j += 1
                
                mention_text = " ".join(chem_tokens)
                decision, rationale = self.classify_chemical_mention(mention_text)
                
                self.mapping_decisions.append({
                    "mention": mention_text,
                    "tokens": chem_tokens,
                    "target_class": decision,
                    "rationale": rationale,
                    "source_split": record.get("source_split", "unknown")
                })
                
                if decision == "MEDICATION":
                    target_tags.append("B-MEDICATION")
                    for _ in range(len(chem_tokens) - 1):
                        target_tags.append("I-MEDICATION")
                else:
                    target_tags.append("B-CHEMICAL")
                    for _ in range(len(chem_tokens) - 1):
                        target_tags.append("I-CHEMICAL")
                        
                i = j
            elif tag == "I-Chemical":
                # Standalone I-Chemical without B-Chemical (invalid BIO transition handled by validator)
                target_tags.append("I-CHEMICAL")
                i += 1
            else:
                target_tags.append("O")
                i += 1
                
        out_record = dict(record)
        out_record["ner_tags"] = target_tags
        return out_record

    def export_audit_log(self, csv_path: str):
        """Export all individual mapping decisions to a CSV audit file."""
        if not self.mapping_decisions:
            return
        fieldnames = ["mention", "target_class", "rationale", "source_split"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.mapping_decisions:
                writer.writerow(row)
        logger.info(f"Exported {len(self.mapping_decisions)} mapping decisions to {csv_path}")
