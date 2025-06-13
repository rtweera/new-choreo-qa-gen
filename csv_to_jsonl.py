import csv
import json
import yaml
import sys
from pathlib import Path

def load_field_mappings(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('field_mappings', {})

def csv_to_jsonl(csv_path, jsonl_path, config_path):
    field_mappings = load_field_mappings(config_path)
    with open(csv_path, 'r', encoding='utf-8') as csvfile, open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            out = {}
            for field in row:
                # Use mapping if present, else keep field name as is
                out[field_mappings.get(field, field)] = row[field]
            jsonlfile.write(json.dumps(out, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert CSV to JSONL with optional field mapping.')
    parser.add_argument('--csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--jsonl', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--config', type=str, default='field_mappings.yaml', help='YAML config file for field mappings')
    args = parser.parse_args()

    csv_to_jsonl(args.csv, args.jsonl, args.config)
