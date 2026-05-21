{ ... }: {
  perSystem = { pkgs, lib, config, ... }: {
    devShells.uv = pkgs.mkShell {
      name = "uv";
      buildInputs = [ pkgs.uv pkgs.python3 pkgs.git ];
      shellHook = ''
        export UV_PYTHON="${pkgs.python3}/bin/python"
        export UV_PROJECT_ENVIRONMENT=".venv"
        uv sync --frozen 2>/dev/null || uv sync
        source .venv/bin/activate
        echo "=== uv (no CUDA) ==="
        echo "Run: python src/gen.py && python src/eval.py"
      '';
    };
    # low priority — uv-cuda.nix overrides this if imported
    devShells.default = lib.mkDefault config.devShells.uv;
  };
}
