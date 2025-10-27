import re
import os

from helper import get_relative_path

def clean_duplicate_platforms(file_path: str):
    regex_pattern = r'\(((?:[+,-]?\d+(?:,[+,-]?\d+)*)(?:\s*,\s*\(.*?\))*)\)'
    lines = []

    print(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    fixes = 0
    for position, line in enumerate(lines):
        input, exec_plan, latency = line.split(":")
        matches_iterator = re.finditer(regex_pattern, exec_plan)

        for match in matches_iterator:
            in_paranthesis = match.group()
            find = in_paranthesis.strip('(').strip(')')
            values = [int(num.strip()) for num in find.split(',')]
            platform_choices = values[43:43+9]
            if sum(platform_choices) > 1:
                print(f"Before: {platform_choices}")
                if platform_choices[5] == 1:
                    for i in range(len(platform_choices)):
                        if i != 5:
                            platform_choices[i] = 0
                print(f"After: {platform_choices}")
                assert sum(platform_choices) <= 1
                values[43:43+9] = platform_choices
                replacement = ','.join(map(str, values))
                new_exec_plan = exec_plan.replace(in_paranthesis, f"({replacement})")
                lines[position] = f"{input}:{new_exec_plan}:{latency}"
                fixes += 1

    with open(file_path, 'w') as file:
        print(fixes)
        file.writelines(lines)

def get_platform_choices(file_path: str, line: int):
    regex_pattern = r'\(((?:[+,-]?\d+(?:,[+,-]?\d+)*)(?:\s*,\s*\(.*?\))*)\)'
    lines = []

    print(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    input, exec_plan, latency = lines[line].split(":")
    matches_iterator = re.finditer(regex_pattern, exec_plan)

    for match in matches_iterator:
        in_paranthesis = match.group()
        find = in_paranthesis.strip('(').strip(')')
        values = [int(num.strip()) for num in find.split(',')]
        platform_choices = values[43:43+9]
        if sum(platform_choices) > 0:
            print(platform_choices)

def main():
    file_path = get_relative_path("retrain-1.txt", "Data/splits/imdb/training/")
    #clean_duplicate_platforms(file_path)
    get_platform_choices(file_path, 353)

if __name__ == "__main__":
    main()
