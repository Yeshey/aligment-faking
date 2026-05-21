{ ... }: {
  perSystem = { pkgs, lib, ... }: {
    devShells.vllm = pkgs.mkShell {
      name = "vllm";
      buildInputs = [ pkgs.python3 pkgs.python3Packages.pip pkgs.git ];
      shellHook = ''
        [ ! -d .venv ] && python -m venv .venv
        source .venv/bin/activate
        pip install -q vllm openai pandas tqdm
        echo "=== vllm (CPU / any hardware) ==="
        echo "Serve: vllm serve <model> --device cpu"
        echo "Run:   python src/gen.py && python src/eval.py"
      '';
    };
  };
}
