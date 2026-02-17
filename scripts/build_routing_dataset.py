#!/usr/bin/env python3
"""
Build routing dataset from existing routeguard datasets.

Each entry will have:
- instruction: The routing prompt with full conversation
- input: Empty string
- output: "ROUTE = <LETTER>" based on the source dataset
- data_source: The original dataset name
"""

import json
import re
from pathlib import Path

# Dataset info path
DATASET_INFO_PATH = "/data/wenjie_jacky_mo/Debug_LM/data/dataset_info.json"
OUTPUT_PATH = "/data/wenjie_jacky_mo/Debug_LM/data/routing_dataset_train_single_original_source.json"

# Routing categories mapping
# A: Animal Abuse, Self Harm, Violence aiding and abetting incitement, Drug abuse weapons banned substance, Terrorism Organized Crime
# B: Child Abuse, Sexually explicit adult content
# C: Financial crime property crime theft, Privacy violation, Drug abuse weapons banned substance
# D: Misinformation regarding ethics laws and safety, Discrimination stereotype injustice, Hate speech offensive language, Controversial topics politics, Non violent unethical behavior

DATASET_TO_ROUTE = {
    # Category A
    "animal_abuse_train": "A",
    "animal_abuse_safe_train": "A",
    "self_harm_train": "A",
    "self_harm_safe_train": "A",
    "violence_aiding_and_abetting_incitement_train": "A",
    "violence_aiding_and_abetting_incitement_safe_train": "A",
    "drug_abuse_weapons_banned_substance_train": "A",
    "drug_abuse_weapons_banned_substance_safe_train": "A",
    "terrorism_organized_crime_train": "A",
    "terrorism_organized_crime_safe_train": "A",
    
    # Category B
    "child_abuse_train": "B",
    "child_abuse_safe_train": "B",
    "sexually_explicit_adult_content_train": "B",
    "sexually_explicit_adult_content_safe_train": "B",
    
    # Category C
    "financial_crime_property_crime_theft_train": "C",
    "financial_crime_property_crime_theft_safe_train": "C",
    "privacy_violation_train": "C",
    "privacy_violation_safe_train": "C",
    
    # Category D
    "misinformation_regarding_ethics_laws_and_safety_train": "D",
    "misinformation_regarding_ethics_laws_and_safety_safe_train": "D",
    "discrimination_stereotype_injustice_train": "D",
    "discrimination_stereotype_injustice_safe_train": "D",
    "hate_speech_offensive_language_train": "D",
    "hate_speech_offensive_language_safe_train": "D",
    "controversial_topics_politics_train": "D",
    "controversial_topics_politics_safe_train": "D",
    "non_violent_unethical_behavior_train": "D",
    "non_violent_unethical_behavior_safe_train": "D",
    
    # Datasets that need assignment (assigning based on closest match)
    # safe_train - general safe content, assign to D (most general category)
    "safe_train": "D",
    "safe_safe_train": "D",
    # mental_health relates to self harm
    "mental_health_over-reliance_crisis_safe_train": "A",
    # copyright_violations - closest to financial crime / non-violent unethical
    "copyright_violations_safe_train": "D",
    # sensitive_information - closest to privacy violation
    "sensitive_information_organization_government_safe_train": "C",
    # cyberattack - closest to financial crime
    "cyberattack_safe_train": "C",
}

PROMPT_TEMPLATE = """[Role]
You are RouteGuard, a conversation router.

Your task is NOT to answer the user.
Your task is to assign the conversation to exactly ONE routing category that best matches the user's topic and potential risk family.

Important:
- Do NOT decide whether the conversation is safe or unsafe.
- Every conversation must be routed to one category (no "SAFE" / "NONE" / "OTHER").
- Choose the category that is MOST RELEVANT to the conversation.

---

[Conversation]
{FULL_CONVERSATION}

---

[Routing Categories]
A. Animal Abuse, Self Harm, Violence aiding and abetting incitement, Drug abuse weapons banned substance, Terrorism Organized Crime
B. Child Abuse, Sexually explicit adult content
C. Financial crime property crime theft, Privacy violation, Drug abuse weapons banned substance
D. Misinformation regarding ethics laws and safety, Discrimination stereotype injustice, Hate speech offensive language, Controversial topics politics, Non violent unethical behavior 

---

[Routing Rules]
- Focus on the topic of the conversation.
- Route based on the dominant theme / risk family, even if it is only discussed abstractly.
- If multiple categories apply, pick the ONE that is most central to the user's message.
- Do NOT explain your reasoning.
- Output only the category letter.

---

[Output Format]ROUTE = <LETTER>"""


def extract_conversation(instruction: str) -> str:
    """Extract the conversation part from the instruction field."""
    # Pattern to match [Conversation] ... [Instructions]
    pattern = r'\[Conversation\]\s*(.*?)\s*\[Instructions\]'
    match = re.search(pattern, instruction, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to find just [Conversation] and take everything after
    pattern2 = r'\[Conversation\]\s*(.*?)$'
    match2 = re.search(pattern2, instruction, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    
    return instruction  # Return full instruction if pattern not found


def build_routing_entry(data_item: dict, data_source: str) -> dict:
    """Build a single routing dataset entry."""
    conversation = extract_conversation(data_item.get("instruction", ""))
    route = DATASET_TO_ROUTE.get(data_source, None)
    
    if route is None:
        return None
    
    prompt = PROMPT_TEMPLATE.replace("{FULL_CONVERSATION}", conversation)
    
    return {
        "instruction": prompt,
        "input": "",
        "output": f"ROUTE = {route}",
        "data_source": data_source
    }


def main():
    # Load dataset info
    with open(DATASET_INFO_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    routing_data = []
    stats = {}
    
    # Process each dataset that we have mapping for
    for dataset_name, route in DATASET_TO_ROUTE.items():
        if dataset_name not in dataset_info:
            print(f"Warning: {dataset_name} not found in dataset_info.json")
            continue
        
        file_path = dataset_info[dataset_name].get("file_name")
        if not file_path:
            print(f"Warning: No file_name for {dataset_name}")
            continue
        
        if not Path(file_path).exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing {dataset_name} -> ROUTE = {route}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            entry = build_routing_entry(item, dataset_name)
            if entry:
                routing_data.append(entry)
                count += 1
        
        stats[dataset_name] = count
        print(f"  Added {count} entries")
    
    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(routing_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Total entries: {len(routing_data)}")
    print(f"\nPer-route counts:")
    route_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for entry in routing_data:
        route = entry["output"].split()[-1]
        route_counts[route] += 1
    
    for route, count in sorted(route_counts.items()):
        print(f"  ROUTE = {route}: {count}")
    
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()



Build routing dataset from existing routeguard datasets.

Each entry will have:
- instruction: The routing prompt with full conversation
- input: Empty string
- output: "ROUTE = <LETTER>" based on the source dataset
- datasource: The original dataset name
"""

import json
import re
from pathlib import Path

# Dataset info path
DATASET_INFO_PATH = "/data/wenjie_jacky_mo/Debug_LM/data/dataset_info.json"
OUTPUT_PATH = "/data/wenjie_jacky_mo/Debug_LM/data/routing_dataset_train_single_original_source.json"

# Routing categories mapping
# A: Animal Abuse, Self Harm, Violence aiding and abetting incitement, Drug abuse weapons banned substance, Terrorism Organized Crime
# B: Child Abuse, Sexually explicit adult content
# C: Financial crime property crime theft, Privacy violation, Drug abuse weapons banned substance
# D: Misinformation regarding ethics laws and safety, Discrimination stereotype injustice, Hate speech offensive language, Controversial topics politics, Non violent unethical behavior

DATASET_TO_ROUTE = {
    # Category A
    "animal_abuse_train": "A",
    "animal_abuse_safe_train": "A",
    "self_harm_train": "A",
    "self_harm_safe_train": "A",
    "violence_aiding_and_abetting_incitement_train": "A",
    "violence_aiding_and_abetting_incitement_safe_train": "A",
    "drug_abuse_weapons_banned_substance_train": "A",
    "drug_abuse_weapons_banned_substance_safe_train": "A",
    "terrorism_organized_crime_train": "A",
    "terrorism_organized_crime_safe_train": "A",
    
    # Category B
    "child_abuse_train": "B",
    "child_abuse_safe_train": "B",
    "sexually_explicit_adult_content_train": "B",
    "sexually_explicit_adult_content_safe_train": "B",
    
    # Category C
    "financial_crime_property_crime_theft_train": "C",
    "financial_crime_property_crime_theft_safe_train": "C",
    "privacy_violation_train": "C",
    "privacy_violation_safe_train": "C",
    
    # Category D
    "misinformation_regarding_ethics_laws_and_safety_train": "D",
    "misinformation_regarding_ethics_laws_and_safety_safe_train": "D",
    "discrimination_stereotype_injustice_train": "D",
    "discrimination_stereotype_injustice_safe_train": "D",
    "hate_speech_offensive_language_train": "D",
    "hate_speech_offensive_language_safe_train": "D",
    "controversial_topics_politics_train": "D",
    "controversial_topics_politics_safe_train": "D",
    "non_violent_unethical_behavior_train": "D",
    "non_violent_unethical_behavior_safe_train": "D",
    
    # Datasets that need assignment (assigning based on closest match)
    # safe_train - general safe content, assign to D (most general category)
    "safe_train": "D",
    "safe_safe_train": "D",
    # mental_health relates to self harm
    "mental_health_over-reliance_crisis_safe_train": "A",
    # copyright_violations - closest to financial crime / non-violent unethical
    "copyright_violations_safe_train": "D",
    # sensitive_information - closest to privacy violation
    "sensitive_information_organization_government_safe_train": "C",
    # cyberattack - closest to financial crime
    "cyberattack_safe_train": "C",
}

PROMPT_TEMPLATE = """[Role]
You are RouteGuard, a conversation router.

Your task is NOT to answer the user.
Your task is to assign the conversation to exactly ONE routing category that best matches the user's topic and potential risk family.

Important:
- Do NOT decide whether the conversation is safe or unsafe.
- Every conversation must be routed to one category (no "SAFE" / "NONE" / "OTHER").
- Choose the category that is MOST RELEVANT to the conversation.

---

[Conversation]
{FULL_CONVERSATION}

---

[Routing Categories]
A. Animal Abuse, Self Harm, Violence aiding and abetting incitement, Drug abuse weapons banned substance, Terrorism Organized Crime
B. Child Abuse, Sexually explicit adult content
C. Financial crime property crime theft, Privacy violation, Drug abuse weapons banned substance
D. Misinformation regarding ethics laws and safety, Discrimination stereotype injustice, Hate speech offensive language, Controversial topics politics, Non violent unethical behavior 

---

[Routing Rules]
- Focus on the topic of the conversation.
- Route based on the dominant theme / risk family, even if it is only discussed abstractly.
- If multiple categories apply, pick the ONE that is most central to the user's message.
- Do NOT explain your reasoning.
- Output only the category letter.

---

[Output Format]ROUTE = <LETTER>"""


def extract_conversation(instruction: str) -> str:
    """Extract the conversation part from the instruction field."""
    # Pattern to match [Conversation] ... [Instructions]
    pattern = r'\[Conversation\]\s*(.*?)\s*\[Instructions\]'
    match = re.search(pattern, instruction, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to find just [Conversation] and take everything after
    pattern2 = r'\[Conversation\]\s*(.*?)$'
    match2 = re.search(pattern2, instruction, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    
    return instruction  # Return full instruction if pattern not found


def build_routing_entry(data_item: dict, data_source: str) -> dict:
    """Build a single routing dataset entry."""
    conversation = extract_conversation(data_item.get("instruction", ""))
    route = DATASET_TO_ROUTE.get(data_source, None)
    
    if route is None:
        return None
    
    prompt = PROMPT_TEMPLATE.replace("{FULL_CONVERSATION}", conversation)
    
    return {
        "instruction": prompt,
        "input": "",
        "output": f"ROUTE = {route}",
        "data_source": data_source
    }


def main():
    # Load dataset info
    with open(DATASET_INFO_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    routing_data = []
    stats = {}
    
    # Process each dataset that we have mapping for
    for dataset_name, route in DATASET_TO_ROUTE.items():
        if dataset_name not in dataset_info:
            print(f"Warning: {dataset_name} not found in dataset_info.json")
            continue
        
        file_path = dataset_info[dataset_name].get("file_name")
        if not file_path:
            print(f"Warning: No file_name for {dataset_name}")
            continue
        
        if not Path(file_path).exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing {dataset_name} -> ROUTE = {route}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            entry = build_routing_entry(item, dataset_name)
            if entry:
                routing_data.append(entry)
                count += 1
        
        stats[dataset_name] = count
        print(f"  Added {count} entries")
    
    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(routing_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Total entries: {len(routing_data)}")
    print(f"\nPer-route counts:")
    route_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for entry in routing_data:
        route = entry["output"].split()[-1]
        route_counts[route] += 1
    
    for route, count in sorted(route_counts.items()):
        print(f"  ROUTE = {route}: {count}")
    
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


