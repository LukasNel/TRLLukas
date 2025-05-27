from .utils import CLIParser
from .samplers.vllm.vllm_server import make_parser as make_vllm_serve_parser
from .samplers.vllm.vllm_server import main as vllm_serve_main

def cli_main():
    parser = CLIParser(prog="Deeptools CLI", usage="deepreasoningwithtools", allow_abbrev=False)

    # Add the subparsers
    subparsers = parser.add_subparsers(help="available commands", dest="command", parser_class=CLIParser)

    # Add the subparsers for every script
    make_vllm_serve_parser(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    if args.command == "vllm-serve":
        (script_args,) = parser.parse_args_and_config()
        vllm_serve_main(script_args)