from toolcallvllm.vllm_server import make_parser, main

if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)

