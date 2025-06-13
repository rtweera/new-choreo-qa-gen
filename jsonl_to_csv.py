import csv
import json
import yaml
import sys
from pathlib import Path

def load_field_mappings(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('field_mappings', {})

def invert_mappings(field_mappings):
    # Invert the mapping: output_field -> input_field
    return {v: k for k, v in field_mappings.items()}

def jsonl_to_csv(jsonl_path, csv_path, config_path):
    field_mappings = load_field_mappings(config_path)
    inverse_mappings = invert_mappings(field_mappings)
    rows = []
    all_fields = set()
    # Read all JSONL lines and collect all possible fields
    with open(jsonl_path, 'r', encoding='utf-8') as jsonlfile:
        for line in jsonlfile:
            obj = json.loads(line)
            # Map fields using inverse mapping if present, else keep as is
            row = {}
            for field in obj:
                csv_field = inverse_mappings.get(field, field)
                row[csv_field] = obj[field]
                all_fields.add(csv_field)
            rows.append(row)
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(all_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert JSONL to CSV with optional field mapping.')
    parser.add_argument('--jsonl', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--config', type=str, default='field_mappings.yaml', help='YAML config file for field mappings')
    args = parser.parse_args()

    jsonl_to_csv(args.jsonl, args.csv, args.config)
