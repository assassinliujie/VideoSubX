from core.prompts import (
    generate_shared_prompt,
    get_prompt_faithfulness,
    get_prompt_expressiveness,
    get_prompt_single_pass,
)
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *

console = Console()


def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {
            "status": "error",
            "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}",
        }

    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {
                "status": "error",
                "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}",
            }

    return {"status": "success", "message": "Translation completed"}


def translate_lines(
    lines,
    previous_content_prompt,
    after_cotent_prompt,
    things_to_note_prompt,
    summary_prompt,
    index=0,
):
    shared_prompt = generate_shared_prompt(
        previous_content_prompt,
        after_cotent_prompt,
        summary_prompt,
        things_to_note_prompt,
    )
    line_count = len(lines.split("\n"))

    # Retry translation if the length of the original text and the translated text are not the same,
    # or if the specified key is missing.
    def retry_translation(prompt, length, step_name):
        def valid_faith(response_data):
            return valid_translate_result(
                response_data,
                [str(i) for i in range(1, length + 1)],
                ["direct"],
            )

        def valid_express(response_data):
            return valid_translate_result(
                response_data,
                [str(i) for i in range(1, length + 1)],
                ["free"],
            )

        def valid_single_pass(response_data):
            return valid_translate_result(
                response_data,
                [str(i) for i in range(1, length + 1)],
                ["direct", "free"],
            )

        for retry in range(3):
            if step_name == "faithfulness":
                result = ask_gpt(
                    prompt + retry * " ",
                    resp_type="json",
                    valid_def=valid_faith,
                    log_title=f"translate_{step_name}",
                )
            elif step_name == "expressiveness":
                result = ask_gpt(
                    prompt + retry * " ",
                    resp_type="json",
                    valid_def=valid_express,
                    log_title=f"translate_{step_name}",
                )
            elif step_name == "single_pass":
                result = ask_gpt(
                    prompt + retry * " ",
                    resp_type="json",
                    valid_def=valid_single_pass,
                    log_title="translate_single_pass",
                )
            else:
                raise ValueError(f"Unsupported translation step: {step_name}")

            if len(result) == line_count:
                return result
            if retry != 2:
                console.print(
                    f"[yellow]Warning: {step_name} translation of block {index} failed, retry...[/yellow]"
                )
        raise ValueError(
            f"{step_name.capitalize()} translation of block {index} failed after 3 retries. "
            "Please check `output/gpt_log/error.json` for more details."
        )

    # Single-pass mode: one API call with merged prompt.
    reflect_translate = load_key("reflect_translate")
    if not reflect_translate:
        prompt_single_pass = get_prompt_single_pass(lines, shared_prompt)
        single_pass_result = retry_translation(prompt_single_pass, line_count, "single_pass")

        for i in single_pass_result:
            single_pass_result[i]["direct"] = str(single_pass_result[i]["direct"]).replace("\n", " ")
            single_pass_result[i]["free"] = str(single_pass_result[i]["free"]).replace("\n", " ")

        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(single_pass_result):
            table.add_row(f"[cyan]Origin:  {single_pass_result[key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {single_pass_result[key]['direct']}[/magenta]")
            table.add_row(f"[green]Free:    {single_pass_result[key]['free']}[/green]")
            if i < len(single_pass_result) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        console.print(table)

        translate_result = "\n".join(
            [single_pass_result[i]["free"].strip() for i in single_pass_result]
        )
        if line_count != len(translate_result.split("\n")):
            console.print(
                Panel(
                    "[red]Translation failed (length mismatch). "
                    "Please check `output/gpt_log/translate_single_pass.json`[/red]"
                )
            )
            raise ValueError(f"Origin >>>{lines}<<<,\nbut got >>>{translate_result}<<<")
        return translate_result, lines

    # Step 1: Faithful to the original text.
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, line_count, "faithfulness")
    for i in faith_result:
        faith_result[i]["direct"] = str(faith_result[i]["direct"]).replace("\n", " ")

    # Step 2: Express smoothly.
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, line_count, "expressiveness")

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
    console.print(table)

    translate_result = "\n".join(
        [str(express_result[i]["free"]).replace("\n", " ").strip() for i in express_result]
    )
    if line_count != len(translate_result.split("\n")):
        console.print(
            Panel(
                f"[red]Translation of block {index} failed, length mismatch. "
                "Please check `output/gpt_log/translate_expressiveness.json`[/red]"
            )
        )
        raise ValueError(f"Origin >>>{lines}<<<,\nbut got >>>{translate_result}<<<")

    return translate_result, lines


if __name__ == "__main__":
    lines = """All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain."""
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(
        lines,
        previous_content_prompt,
        after_cotent_prompt,
        things_to_note_prompt,
        summary_prompt,
    )
